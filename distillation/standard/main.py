import os

from functools import partial

import transformers
import torch
from torch.utils.data import DataLoader

from utils.data import SoftLabelDataset
from utils.preprocess import soft_label_tokenize, collate_fn
from utils.parameters import MAX_LENGTH, BATCH_SIZE, T_PRESET, A_PRESET
from utils.path import BASE_DIR, STUDENT_MODEL_NAME, STUDENT_TOKENIZER_NAME, TEACHER_MODEL_NAME, FILENAME, PEFT_MODEL_NAME
from utils.models import load_model, DistillationLoss, train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

filename = os.path.join(BASE_DIR, "dataset", "standard", FILENAME)

dataset = SoftLabelDataset(filename, is_inference=False)

tokenizer_name = os.path.join(BASE_DIR, "models", STUDENT_TOKENIZER_NAME)
tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
if STUDENT_MODEL_NAME in ["codegen-mono-small", "codegen-multi-small"]:
    tokenizer.pad_token = tokenizer.eos_token

max_length = MAX_LENGTH
tokenize_preprocessing = partial(
    soft_label_tokenize,
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

epochs = 10
learning_rate = 1e-5
num_warmup_steps = epochs // 10
num_training_steps = epochs * len(dataloader)

T = T_PRESET
alpha = A_PRESET

criterion = DistillationLoss(T, alpha)

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

# 设置模型保存路径
save_dir = os.path.join(BASE_DIR, "models", PEFT_MODEL_NAME)

is_decoder = True
if STUDENT_MODEL_NAME in ["codet5p-small"]:
    is_decoder = False

train(student_model, teacher_model, dataloader, criterion, optimizer, scheduler, epochs=epochs, is_decoder=is_decoder)