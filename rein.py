import contextlib
import signal
import io
import random

import torch
import torch.nn.functional as F
from tqdm import tqdm

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent():
    """
    Is it possible to make the structure of the Q table simpler?
    """
    def __init__(self, model, tokenizer, dataset, dataloader, validation_dataloader, args):
        self.model = model
        if torch.cuda.device_count() > 1:
            self.model = model.module
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.dataloader = dataloader
        self.validation_dataloader = validation_dataloader

        self.args = args
        self.m = args.max_new_tokens
        self.n = len(dataset)
        self.k = 1
        self.top_k = 50

        self.T = args.T

        self.shape = (self.n, self.m, self.top_k)
        self.q_tables = torch.zeros(self.shape, dtype=float).to(device)
        self.q_maps = torch.zeros(self.shape, dtype=float).to(device)

        self.q_best = -1.0

    def initialize(self):
        """
        initialize q tables
        """
        # 构造Q值表是否一定要right padding?
        i = 0
        print("\tInitialize the Q table...")
        self.q_tables = torch.zeros(self.shape, dtype=float).to(device)
        self.q_maps = torch.zeros(self.shape, dtype=float).to(device)
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(self.dataloader):
                input_ids = batch["input_ids"].to(device)
                attention_masks = batch["attention_masks"].to(device)

                # 如果是自回归模型，那么input_ids应该是prompt+label，这里的dataloader应该和微调用的保持一致
                output = None
                if self.args.is_decoder:
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
                indices = torch.topk(probs, self.top_k, dim=-1).indices
                for j in range(len(indices)):
                    self.q_maps[i] = indices[j]
                    i += 1

    def update_single(self, i, text, is_valid, is_compiled, success_v=1.0, compile_v=0.3, failed_v=-1.0):
        # 有些模型可能编译成功率过高，但是执行成功率很低，这种情况Q值会变高，是否合理？
        tokens = self.tokenizer(
            text,
            max_length=self.m,
            padding="max_length",
            truncation=True
        )["input_ids"]

        # 代码生成结果正确，将对应位置+1，代码生成结果不正确，将部分位置-1（因为导致错误的原因很可能只有一小部分）
        if is_valid:
            for j in range(len(tokens)):
                result = torch.where(self.q_maps[i][j] == tokens[j])
                if len(result[0]) == 1:
                    index = result[0].item()
                    self.q_tables[i][j][index] += success_v
        else:
            if is_compiled is True:
                for j in range(len(tokens)):
                    result = torch.where(self.q_maps[i][j] == tokens[j])
                    if len(result[0]) == 1:
                        index = result[0].item()
                        self.q_tables[i][j][index] += compile_v
            else:
                for j in random.sample(range(len(tokens)), self.m):
                    result = torch.where(self.q_maps[i][j] == tokens[j])
                    if len(result[0]) == 1:
                        index = result[0].item()
                        self.q_tables[i][j][index] += failed_v

    def update(self):
        self.initialize()
        print("\tUpdate the Q table...")
        self.model.eval()
        results = []
        with torch.no_grad():
            # 这里必须改成贪婪解码
            for batch in tqdm(self.validation_dataloader):
                input_ids = batch["input_ids"].to(device)
                attention_masks = batch["attention_masks"].to(device)

                # model.generate的输入为仅prompt，确保dataloader中的prompt正确
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_masks,
                    pad_token_id=self.tokenizer.pad_token_id,
                    max_new_tokens=self.m
                )

                result = self.tokenizer.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )
                task_ids = batch["task_ids"]
                for i in range(len(result)):
                    data = {
                        "task_id": task_ids[i//self.k],
                        "completion": result[i]
                    }
                    results.append(data)

        total_compiled = 0
        total_correct = 0
        for i in range(len(results)):
            idx = i // self.k
            task_id = results[i]["task_id"]
            completion = results[i]["completion"]
            # 去除completion中多余的部分，使得生成的结果无异常
            completion_clean = self.truncate_main(completion)
            completion_clean = self.truncate(completion_clean, is_decoder=self.args.is_decoder)
            completion_clean = self.clean_print(completion_clean, is_decoder=self.args.is_decoder)
            check_globals = {}
            check_program = None
            if self.args.validation_dataset not in ["mbpp", "mbpp_validation"]:
                # 不是mbpp数据集，prompt和code都是代码
                if self.args.is_decoder is False:
                    check_program = (
                        self.dataset[idx]["prompt"] + completion_clean + "\n" +
                        self.dataset[idx]["test"] + "\n" +
                        f"check({self.dataset[idx]['entry_point']})"
                    )
                else:
                    check_program = (
                        completion_clean + "\n" +
                        self.dataset[idx]["test"] + "\n" +
                        f"check({self.dataset[idx]['entry_point']})"
                    )
            else:
                if self.args.is_decoder is False:
                    check_program = (
                        self.dataset[idx]["prompt_code"] + completion_clean + "\n" +
                        self.dataset[idx]["test"] + "\n" +
                        f"check({self.dataset[idx]['entry_point']})"
                    )
                else:
                    check_program = (
                        self.dataset[idx]["prompt_code"] + self.truncate_function(completion_clean) + "\n" +
                        self.dataset[idx]["test"] + "\n" +
                        f"check({self.dataset[idx]['entry_point']})"
                    )
            is_compiled = False
            with swallow_io():
                with time_limit(3.0):
                    try:
                        c = compile(check_program, "<dynamic>", "exec")
                        if c is not None:
                            is_compiled = True
                            total_compiled += 1
                        exec(check_program, check_globals)
                        total_correct += 1
                        self.update_single(i//self.k, completion, is_valid=True, is_compiled=True)
                    except Exception:
                        self.update_single(i//self.k, completion, is_valid=False, is_compiled=is_compiled)

        print("\tCorrect Rate: {0}".format(total_correct / (self.n * self.k)))
        print("\tCompile Rate: {0}".format(total_compiled / (self.n * self.k)))
        q = self.get_q()
        print("\tbest q: {0}, current q: {1}".format(self.q_best, q))

    def get_q(self):
        q = self.q_tables.sum() / (self.m * self.n)
        return q

    def get_next_T(self):
        q = self.get_q()
        if q > self.q_best:
            self.T = max(0.1, self.T - 0.2)
        else:
            self.T = self.args.T
        return self.T
    
    def is_rational(self):
        q = self.get_q()
        if q > self.q_best:
            self.q_best = q
            return True
        else:
            return False

    def truncate(self, completion, is_decoder=False):
        # 去除掉最后一个def，因为最后一个def大概率不完整，并且保证至少有一个def
        group = completion.rsplit("def", 1)
        if group[0].find("def") != -1:
            return group[0]
        else:
            return completion

    def clean_print(self, completion, is_decoder=False):
        # 去除掉print的部分，但仅解码器的print可能在prompt中
        if is_decoder is False:
            index = completion.rfind("print")
            if index != -1:
                completion = completion[:index]
        else:
            index_def = completion.find("def")
            index = completion.rfind("print", index_def)
            if index != -1:
                completion = completion[:index]
        return completion

    def truncate_main(self, completion):
        index = completion.find("if __name")
        if index != -1:
            return completion[:index]
        return completion

    def truncate_function(self, completion):
        # 只保留def以后的部分，不包括def这一行
        lines = completion.split("\n")
        i = 0
        while i < len(lines):
            single = lines[i]
            if single.startswith("def"):
                break
            i = i + 1
        return "\n".join(lines[i+1:])