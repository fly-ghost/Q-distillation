import os
import sys

# 要看命令在哪里执行
sys.path.append("/home/aistudio/work/libs")
sys.path.append("/home/aistudio/work/")

from functools import partial
import random

import transformers
import torch
from torch.utils.data import DataLoader
import numpy as np

from utils.data import FeedBackDataset
from utils.preprocess import feedback_tokenize, collate_fn
from utils.parameters import MAX_LENGTH, BATCH_SIZE
from utils.path import BASE_DIR, STUDENT_MODEL_NAME, STUDENT_TOKENIZER_NAME, TEACHER_MODEL_NAME, FILENAME
from utils.models import load_model, FeedBackDistillationLoss, feedback_train
from rein import Agent

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# humaneval数据集路径，微调需要使用包含教师模型生成label的数据集
filename = os.path.join(BASE_DIR, "dataset", "feedback", FILENAME)

# 模型蒸馏的条件是已经有训练好的模型，因此不需要分训练集和测试集
dataset = FeedBackDataset(filename, is_inference=False)

# 加载分词器
tokenizer_name = os.path.join(BASE_DIR, "models", STUDENT_TOKENIZER_NAME)
# 模型训练时需要采用right-padding，推理时采用left-padding
tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
# 部分分词器没有预先设置PAD和ATTENTION_MASK
if STUDENT_MODEL_NAME in ["codegen-mono-small", "codegen-multi-small"]:
    tokenizer.pad_token = tokenizer.eos_token

# 基于反馈的方法需要进行模型推理，所以padding_side需要设置成left
tokenizer_inference = transformers.AutoTokenizer.from_pretrained(
    tokenizer_name,
    padding_side="left"
    )
if STUDENT_MODEL_NAME in ["codegen-mono-small", "codegen-multi-small"]:
    tokenizer_inference.pad_token = tokenizer_inference.eos_token

# 对dataset进行处理，将文本变成token的形式
# 如果需要微调模型，那么tokenizer的max_length需要设置的大一些，防止不精确导致效果差
max_length = MAX_LENGTH

tokenize_preprocessing = partial(
    feedback_tokenize,
    tokenizer=tokenizer,
    max_length=max_length,
    is_inference=False
)
dataset.map(tokenize_preprocessing)

# batch_size不能设置的太大，不然模型无法训练
batch_size = BATCH_SIZE

dataloader = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    collate_fn=collate_fn
)

# 需要注意模型支持tokens的上限，比如codebert就是512个，超过这个值无法推理
student_model_name = STUDENT_MODEL_NAME
student_model = load_model(student_model_name, is_student=True)

teacher_model_name = TEACHER_MODEL_NAME
teacher_model = load_model(teacher_model_name, is_student=False)

epochs = 40
learning_rate = 5e-5
num_warmup_steps = epochs // 10
num_training_steps = epochs * len(dataloader)

T = 1.0
criterion = FeedBackDistillationLoss(T)

optimizer = torch.optim.AdamW(
    student_model.parameters(),
    lr=learning_rate
)
scheduler = transformers.get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

# 反馈代理类
is_decoder = True
if STUDENT_MODEL_NAME in ["codet5p-small"]:
    is_decoder = False

agent = Agent(student_model, tokenizer, tokenizer_inference, is_decoder=is_decoder, T=T, step=2)

feedback_train(student_model, teacher_model, dataloader, criterion, optimizer, scheduler, agent, epochs, is_decoder, beta=100)