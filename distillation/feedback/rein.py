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
    def __init__(self, model, tokenizer, tokenizer_inference, is_decoder=True, k=50, step=2, T=1.0):
        self.is_decoder = is_decoder
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer_inference = tokenizer_inference
        self.dataset_origin = None
        self.dataloader = None
        self.dataloader_inference = None

        self.T = T

        self.step = step
        self.n = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.k = k
        self.m = MAX_LENGTH

        filename = os.path.join(BASE_DIR, "dataset", "feedback", FILENAME_INFERENCE)
        self.initialize(filename)
        self.shape = (self.n, self.m, self.k)
        self.q_tables = torch.zeros(self.shape, dtype=float).to(self.device)
        self.q_maps = torch.zeros(self.shape, dtype=float).to(self.device)
        self.i = 0
        self.q_old = torch.tensor(0.0)
        self.q_best = torch.tensor(0.0)
        self.success_rate = 0.0
        self.compile_rate = 0.0
        self.initialize_q()

    def initialize(self, filename):
        self.dataset_origin = HumanEvalDataset(filename, is_inference=True, is_test=True)
        dataset = HumanEvalDataset(filename, is_inference=True)
        dataset_inference = HumanEvalDataset(filename, is_inference=True)
        self.n = len(dataset)
        self.shape = (self.n, self.m, self.k)
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
        q = self.q_tables.sum() / (self.n * self.m)
        return q

    def is_rational(self):
        q = self.get_q()
        if q >= self.q_best:
            return True
        else:
            return False

    def get_T(self, beta=100):
        q = self.get_q()
        T_new = self.T * torch.exp((self.q_old - q) * beta)
        self.q_old = q
        return max(0.1, min(T_new.item(), self.T))
    
    def _clear(self):
        self.q_tables = torch.zeros(self.shape, dtype=float).to(self.device)
        self.q_maps = torch.zeros(self.shape, dtype=float).to(self.device)

    def reset(self):
        self._clear()
        self.i = 0
        print("\tResetting the Q table...")
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
                    output = self.model(
                        input_ids=input_ids,
                        decoder_input_ids=input_ids,
                        attention_mask=attention_masks
                    )
                logits = output.logits
                probs = F.softmax(logits, dim=-1)
                indices = torch.topk(probs, self.k, dim=-1).indices
                for j in range(len(indices)):
                    self.q_maps[self.i] = indices[j]
                    self.i += 1
        self.i = 0

    def update_single(self, text, is_valid, is_compiled=False, success_v=1.0, compile_v=0.1, failed_v=-0.2):

        assert self.i < self.n

        tokens = self.tokenizer(
            text,
            max_length=self.m,
            padding="max_length",
            truncation=True
        )["input_ids"]

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

                results_clean = []

                if self.is_decoder is False:
                    for i in range(len(results)):
                        group = results[i].split('if __name')
                        results_clean.append(group[0])
                else:
                    for i in range(len(results)):
                        group = results[i].rsplit("def", 1)
                        if group[0].find("def") != -1:
                            results_clean.append(group[0])
                        else:
                            results_clean.append(results[i])

                for j in range(len(results)):
                    check_globals = {}
                    check_program = None
                    if self.is_decoder is False:
                        check_program = (
                            self.dataset_origin[self.i]["prompt"] + results_clean[j] + "\n" +
                            self.dataset_origin[self.i]["test"] + "\n" +
                            f"check({self.dataset_origin[self.i]['entry_point']})"
                        )
                    else:
                        check_program = (
                            results_clean[j] + "\n" +
                            self.dataset_origin[self.i]["test"] + "\n" +
                            f"check({self.dataset_origin[self.i]['entry_point']})"
                        )
                    with swallow_io():
                        with time_limit(3.0):
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
                                self.update_single(results_clean[j], False, is_compiled)
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
        self.reset()
        self.update_all()
        self.q_old = self.get_q()
        self.q_best = self.q_old
        print("\tinitialize q, q_best: {0}".format(self.q_best))

    def is_update_available(self, epoch):
        if epoch % self.step == 0:
            return True
        else:
            return False
