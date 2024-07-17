from functools import partial
import os
import contextlib
import signal
import io
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.data import HumanEvalDataset
from utils.preprocess import feedback_tokenize, collate_fn_inference
from utils.parameters import MAX_LENGTH, BATCH_SIZE_INFERENCE, MAX_NEW_TOKENS
from utils.path import BASE_DIR, FILENAME_INFERENCE

class TimeoutException(Exception):
    pass

@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)

class WriteOnlyStringIO(io.StringIO):
    """ StringIO that throws an exception when it's read from """

    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        """ Returns True if the IO object can be read. """
        return False

class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = 'stdin'

@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield

class Agent():
    """
    Q表的代理, 用以初始化Q表, 更新Q表, 得到新的T和a等等
    由于需要进行模型推理等操作, 所以需要两种tokenizer和dataloader
    """
    def __init__(self, model, tokenizer, tokenizer_inference, is_decoder=True, k=50, step=2, T=1.0):
        self.is_decoder = is_decoder
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer_inference = tokenizer_inference
        self.dataset_origin = None
        self.dataloader = None
        self.dataloader_inference = None

        # 初始T，让新T围绕初始T变化
        self.T = T

        self.step = step
        self.n = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Q表应该会有n个，每一个都是(MAX_LENGTH, k)
        self.k = k
        self.m = MAX_LENGTH

        # 初始化验证集
        filename = os.path.join(BASE_DIR, "dataset", "feedback", FILENAME_INFERENCE)
        self.initialize(filename)
        self.shape = (self.n, self.m, self.k)
        self.q_tables = torch.zeros(self.shape, dtype=float).to(self.device)
        # 除了Q表，还需要n个状态-最有可能动作集合表
        self.q_maps = torch.zeros(self.shape, dtype=float).to(self.device)
        # 记录Q表更新到了哪个表
        self.i = 0
        # 旧的q值，按照强化学习的思想，q值需要更高才行，保存模型时q值会作为主要依据，而不是loss
        self.q_old = torch.tensor(0.0)
        # 最好的q值
        self.q_best = torch.tensor(0.0)
        # 当前成功率
        self.success_rate = 0.0
        # 当前编译成功率
        self.compile_rate = 0.0
        self.initialize_q()

    def initialize(self, filename):
        """
        代理初始化, 代理使用的验证数据集其实是固定的, 并不需要传入
        """
        # 原始数据集，不做任何处理
        self.dataset_origin = HumanEvalDataset(filename, is_inference=True, is_test=True)
        # 前向传播用数据集
        dataset = HumanEvalDataset(filename, is_inference=True)
        # 推理用数据集
        dataset_inference = HumanEvalDataset(filename, is_inference=True)
        self.n = len(dataset)
        self.shape = (self.n, self.m, self.k)
        # 因为梯度在这里被锁住，所以不需要label
        tokenize_preprocessing = partial(
            feedback_tokenize,
            tokenizer=self.tokenizer,
            max_length=MAX_LENGTH,
            is_inference=True
        )
        dataset.map(tokenize_preprocessing)

        tokenize_preprocessing_inference = partial(
            feedback_tokenize,
            tokenizer=self.tokenizer_inference,
            max_length=MAX_LENGTH,
            is_inference=True
        )
        dataset_inference.map(tokenize_preprocessing_inference)

        batch_size = BATCH_SIZE_INFERENCE
        batch_size_inference = BATCH_SIZE_INFERENCE

        self.dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=collate_fn_inference
        )

        self.dataloader_inference = DataLoader(
            dataset=dataset_inference,
            batch_size=batch_size_inference,
            collate_fn=collate_fn_inference
        )

    def get_success_rate(self):
        return self.success_rate

    def get_compile_rate(self):
        return self.compile_rate

    def get_q(self):
        # 获取总的q值，根据强化学习的思想，应当使q值更高
        q = self.q_tables.sum() / (self.n * self.m)
        return q

    def is_rational(self):
        # 只有当当前q大于等于历史最大q值，才会保存模型
        q = self.get_q()
        if q >= self.q_best:
            return True
        else:
            return False

    def get_T(self, beta=100):
        """
        根据上一次的T, 获取当前T, beta需要根据epochs适当调整, epochs大, beta就大
        """
        q = self.get_q()
        # 如果q变高，说明这一次蒸馏有效，那么让T变小，学生模型就能继续学习突出的部分。如果q变低，说明匹配度不高
        T_new = self.T * torch.exp((self.q_old - q) * beta)
        self.q_old = q
        return max(0.1, min(T_new.item(), self.T))
    
    def _clear(self):
        # 在每一次求和后，重置Q表，重新初始化
        self.q_tables = torch.zeros(self.shape, dtype=float).to(self.device)
        self.q_maps = torch.zeros(self.shape, dtype=float).to(self.device)

    def reset(self):
        # 真正的重置Q表，初始化Q表
        # 初始化即获取每一条数据的每一个token，其对应的最有可能的前k个token
        self._clear()
        self.i = 0
        print("\tResetting the Q table...")
        # 初始化Q表，注意model()只接受padding_side="right"的，否则结果可能不正确
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(self.dataloader):
                input_ids = batch["input_ids"].to(self.device)
                attention_masks = batch["attention_masks"].to(self.device)

                output = None
                if self.is_decoder:
                    output = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_masks
                    )
                else:
                    # 由于梯度被禁用，decoder_input_ids并不会对结果产生影响，但是其本身要求有这个参数传入
                    output = self.model(
                        input_ids=input_ids,
                        decoder_input_ids=input_ids,
                        attention_mask=attention_masks
                    )
                logits = output.logits
                # logits仅仅是全连接层的最后一层输出，需要先softmax才能得到token序列的概率，再通过top_k得到前k个最大概率的id
                probs = F.softmax(logits, dim=-1)
                indices = torch.topk(probs, self.k, dim=-1).indices
                for j in range(len(indices)):
                    self.q_maps[self.i] = indices[j]
                    self.i += 1
        self.i = 0

    def update_single(self, text, is_valid, is_compiled=False, success_v=1.0, compile_v=0.1, failed_v=-0.2):
        """
        只更新一条数据
        """

        assert self.i < self.n

        # 先编码text，再更新，这里需要用训练用的tokenizer，毕竟q_tabls时根据model()得来的
        tokens = self.tokenizer(
            text,
            max_length=self.m,
            padding="max_length",
            truncation=True
        )["input_ids"]

        # 代码生成结果正确，将对应位置+1，代码生成结果不正确，将部分位置-1（因为导致错误的原因很可能只有一小部分）
        if is_valid:
            for j in range(len(tokens)):
                result = torch.where(self.q_maps[self.i][j] == tokens[j])
                if len(result[0]) == 1:
                    index = result[0].item()
                    self.q_tables[self.i][j][index] += success_v
        else:
            if is_compiled is True:
                for j in range(len(tokens)):
                    result = torch.where(self.q_maps[self.i][j] == tokens[j])
                    if len(result[0]) == 1:
                        index = result[0].item()
                        self.q_tables[self.i][j][index] += compile_v
            else:
                for j in random.sample(range(len(tokens)), self.m):
                    result = torch.where(self.q_maps[self.i][j] == tokens[j])
                    if len(result[0]) == 1:
                        index = result[0].item()
                        self.q_tables[self.i][j][index] += failed_v

    def update_all(self):
        """
        在一轮epoch结束后, 根据model生成的结果, 更新Q表, 不在微调的过程中进行
        """
        # max_new_tokens需要稍微高一些，但是不能太高，稍微高一些能收集较为准确的结果
        max_new_tokens = MAX_NEW_TOKENS
        print("\tUpdating the Q table...")
        total_correct = 0
        total_compiled = 0
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(self.dataloader_inference):
                input_ids = batch["input_ids"].to(self.device)
                attention_masks = batch["attention_masks"].to(self.device)
                
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_masks,
                    pad_token_id=self.tokenizer_inference.pad_token_id,
                    max_new_tokens=max_new_tokens
                )
                
                results = self.tokenizer_inference.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )

                # 最好不要变动results，保留原始结果
                results_clean = []

                if self.is_decoder is False:
                    for i in range(len(results)):
                        # 去除掉if __name__ == '__main__'
                        group = results[i].split('if __name')
                        results_clean.append(group[0])
                else:
                    for i in range(len(results)):
                        # 去掉多余的函数定义
                        group = results[i].rsplit("def", 1)
                        if group[0].find("def") != -1:
                            results_clean.append(group[0])
                        else:
                            results_clean.append(results[i])

                for j in range(len(results)):
                    # 判断生成的结果是否正确，并更新Q表
                    # 单元测试必须准确，否则后面都没有意义
                    check_globals = {}
                    check_program = None
                    if self.is_decoder is False:
                        # codet5p结果不包含prompt，要加上
                        check_program = (
                            self.dataset_origin[self.i]["prompt"] + results_clean[j] + "\n" +
                            self.dataset_origin[self.i]["test"] + "\n" +
                            f"check({self.dataset_origin[self.i]['entry_point']})"
                        )
                    else:
                        # codegen结果包含prompt，无需再加
                        check_program = (
                            results_clean[j] + "\n" +
                            self.dataset_origin[self.i]["test"] + "\n" +
                            f"check({self.dataset_origin[self.i]['entry_point']})"
                        )
                    with swallow_io():
                        with time_limit(3.0):
                            # 如果使用编译成功作为其中一个判别条件，会对q值有较大影响，多编译成功一个的影响远大于多通过一个单元测试
                            is_compiled = False
                            try:
                                c = compile(check_program, "<dynamic>", "exec")
                                if c is not None:
                                    is_compiled = True
                                    total_compiled += 1
                                exec(check_program, check_globals)
                                total_correct += 1
                                self.update_single(results_clean[j], True)
                            except Exception:
                                # 有可能是编译成功，但是运行错误
                                self.update_single(results_clean[j], False, is_compiled)
                    # 更新一条数据，那么self.i就应该+1
                    self.i += 1
        self.i = 0
        self.success_rate = total_correct / self.n
        self.compile_rate = total_compiled / self.n
        print("\tSuccess Rate: {0}".format(total_correct / self.n))
        print("\tCompile Rate: {0}".format(total_compiled / self.n))
        q = self.get_q()
        print("\tbest q: {0}, old q: {1}, current q: {2}".format(self.q_best, self.q_old, q))
        # 更新q_best
        if q > self.q_best:
            self.q_best = q

    def initialize_q(self):
        """
        初次使用agent, 需要调用此函数, 获取模型初始的q值, 设置q_old以及q_best
        """
        self.reset()
        self.update_all()
        self.q_old = self.get_q()
        self.q_best = self.q_old
        print("\tinitialize q, q_best: {0}".format(self.q_best))

    def is_update_available(self, epoch):
        """
        判断这一次是否需要更新
        """
        if epoch % self.step == 0:
            return True
        else:
            return False
