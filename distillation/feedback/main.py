import os
import sys


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
from utils.models import load_model, FeedBackDistillationLoss, feedback_train, FeedBackWithRememberDistillationLoss, DistillationLoss
from rein import Agent

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

filename = os.path.join(BASE_DIR, "dataset", "feedback", FILENAME)

dataset = FeedBackDataset(filename, is_inference=False)

tokenizer_name = os.path.join(BASE_DIR, "models", STUDENT_TOKENIZER_NAME)
tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
if STUDENT_MODEL_NAME in ["codegen-mono-small", "codegen-multi-small"]:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer_inference = transformers.AutoTokenizer.from_pretrained(
    tokenizer_name,
    padding_side="left"
    )
if STUDENT_MODEL_NAME in ["codegen-mono-small", "codegen-multi-small"]:
    tokenizer_inference.pad_token = tokenizer_inference.eos_token

max_length = MAX_LENGTH

tokenize_preprocessing = partial(
    feedback_tokenize,
    tokenizer=tokenizer,
    max_length=max_length,
    is_inference=False
)
dataset.map(tokenize_preprocessing)

batch_size = BATCH_SIZE

dataloader = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    collate_fn=collate_fn
)

student_model_name = STUDENT_MODEL_NAME
student_model = load_model(student_model_name, is_student=True)

teacher_model_name = TEACHER_MODEL_NAME
teacher_model = load_model(teacher_model_name, is_student=False)

epochs = 20
learning_rate = 5e-5
num_warmup_steps = epochs // 10
num_training_steps = epochs * len(dataloader)

T = 1.0
criterion = FeedBackWithRememberDistillationLoss(T)

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

is_decoder = True
if STUDENT_MODEL_NAME in ["codet5p-small"]:
    is_decoder = False

agent = Agent(student_model, tokenizer, tokenizer_inference, is_decoder=is_decoder, T=T, step=2)

feedback_train(student_model, teacher_model, dataloader, criterion, optimizer, scheduler, agent, epochs, is_decoder, beta=100)