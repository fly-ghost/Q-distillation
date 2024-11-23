"""
Fine-tune the model directly
"""
import sys
sys.path.append("libs")

import os

import argparse
import random

import numpy as np
import torch
from transformers import get_scheduler

from utils.models import load_model, load_tokenizer, load_dataloader, load_dataset, feedback_distillation, FeedBackDistillationLoss, DistillationLoss
from rein import Agent

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(42)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    type=str,
    default="codet5p-small"
)
parser.add_argument(
    "--is_decoder",
    type=bool,
    default=False
)
parser.add_argument(
    "--tokenizer",
    type=str,
    default="codet5p-small"
)
parser.add_argument(
    "--teacher_tokenizer",
    type=str,
    default="codet5p-base"
)
parser.add_argument(
    "--teacher_model",
    type=str,
    default="codet5p-base"
)
parser.add_argument(
    "--dataset",
    type=str,
    default="humaneval",
    choices=["humaneval", "codet_humaneval", "mbpp_sanitized", "mbpp"]
)
parser.add_argument(
    "--validation_dataset",
    type=str,
    default="humaneval",
    choices=["humaneval", "codet_humaneval", "mbpp_sanitized", "mbpp", "mbpp_validation"]
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=8
)
parser.add_argument(
    "--max_length",
    type=int,
    default=256
)
parser.add_argument(
    "--epochs",
    type=int,
    default=10
)
parser.add_argument(
    "--lr",
    type=float,
    default=5e-5
)
parser.add_argument(
    "--scheduler",
    type=str,
    default="cosine_with_restarts",
    choices=["linear", "cosine", "cosine_with_restarts"]
)
parser.add_argument(
    "--T",
    type=float,
    default=1.0
)
parser.add_argument(
    "--max_new_tokens",
    type=int,
    default=256
)
parser.add_argument(
    "--k",
    type=int,
    default=10
)
parser.add_argument(
    "--temperature",
    type=float,
    default=0.95
)
parser.add_argument(
    "--top_p",
    type=float,
    default=1.0
)
parser.add_argument(
    "--model_save_dir",
    type=str,
    default="feedback"
)

args = parser.parse_args()

device_ids = [0]
teacher_device_ids = [0]

if torch.cuda.device_count() > 1:
    device_ids = [0, 1]
    teacher_device_ids = [2, 3]

# load student tokenizer
student_tokenizer = load_tokenizer(os.path.join("models", args.tokenizer), args.tokenizer, is_train=False)

# load teacher model
student_model = load_model(os.path.join("models", args.model), is_peft=False, is_train=True, is_decoder=args.is_decoder, device_ids=device_ids)

# load teacher tokenizer
teacher_tokenizer = load_tokenizer(os.path.join("models", args.teacher_tokenizer), args.teacher_tokenizer, is_train=False)

# load teacher model
teacher_model = load_model(os.path.join("models", args.teacher_model), is_peft=False, is_train=False, is_decoder=args.is_decoder, device_ids=teacher_device_ids)

# dataloader
dataloader = load_dataloader(os.path.join("dataset", args.dataset + ".jsonl"), args.dataset, student_tokenizer, max_length=args.max_length, batch_size=args.batch_size, is_train=True, is_decoder=args.is_decoder)

# dataset for agent
validation_dataset = load_dataset(os.path.join("dataset", args.validation_dataset + ".jsonl"), args.validation_dataset, is_train=True)

# dataloader for agent
validation_tokenizer = load_tokenizer(os.path.join("models", args.tokenizer), args.tokenizer, is_train=False)
validation_dataloader = load_dataloader(os.path.join("dataset", args.validation_dataset + ".jsonl"), args.validation_dataset, validation_tokenizer, max_length=args.max_length, batch_size=args.batch_size, is_train=False, is_decoder=args.is_decoder)

# distill the model
num_warmup_steps = args.epochs // 10
num_training_steps = args.epochs * len(dataloader)

criterion = FeedBackDistillationLoss(T=args.T)

optimizer = torch.optim.AdamW(
    student_model.parameters(),
    lr=args.lr
)
scheduler = get_scheduler(
    name=args.scheduler,
    optimizer=optimizer,
    num_warmup_steps=num_training_steps,
    num_training_steps=num_training_steps
)

model_save_dir = os.path.join("models", args.model, args.model_save_dir, args.dataset)
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

model_save_dir_temp = os.path.join("models", args.model, args.model_save_dir + "_temp", args.dataset)
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir_temp)

agent = Agent(student_model, validation_tokenizer, validation_dataset, dataloader, validation_dataloader, args)

feedback_distillation(
    student_model,
    teacher_model,
    dataloader,
    criterion,
    optimizer,
    scheduler,
    model_save_dir,
    model_save_dir_temp,
    agent,
    epochs=args.epochs,
    is_decoder=args.is_decoder,
    device_ids=device_ids,
    teacher_device_ids=teacher_device_ids
)