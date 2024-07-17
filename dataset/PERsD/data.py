"""
个性化蒸馏数据预处理, 把label变成符合学生模型的label
"""
import sys

sys.path.append("/home/aistudio/work/libs")
sys.path.append("/home/aistudio/work/")

import os
from tqdm import tqdm
from functools import partial
import json

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import torch
from torch.utils.data import DataLoader

from utils.data import MbppDataset
from utils.preprocess import tokenize, collate_fn
from utils.parameters import MAX_LENGTH, MAX_NEW_TOKENS, BATCH_SIZE
from utils.path import ORIGIN_MODEL_NAME, ORIGIN_TOKENIZER_NAME, BASE_DIR
from utils.models import load_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer_name = os.path.join(BASE_DIR, "models", ORIGIN_TOKENIZER_NAME)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, padding_side="left")
# 部分分词器没有预先设置PAD和ATTENTION_MASK
if ORIGIN_MODEL_NAME in ["codegen-mono-small", "codegen-mono-base", "codegen-multi-small", "codegen-multi-base"]:
    tokenizer.pad_token = tokenizer.eos_token

model_name = os.path.join(BASE_DIR, "models", ORIGIN_MODEL_NAME)
model = load_model(model_name)

filename = os.path.join(BASE_DIR, "dataset", "mbpp-formatting.jsonl")
dataset = MbppDataset(filename, is_inference=True)

max_length = MAX_LENGTH
tokenize_preprocessing = partial(
    tokenize,
    tokenizer=tokenizer,
    max_length=max_length,
    is_inference=True
)
dataset.map(tokenize_preprocessing)

batch_size = 16

dataloader = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    collate_fn=collate_fn
)

def inference(model, tokenizer, dataloader):
    model.eval()
    results = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_masks = batch["attention_masks"].to(device)

            generated_ids = model.generate(
                input_ids,
                attention_mask=attention_masks,
                pad_token_id=tokenizer.pad_token_id,
                max_new_tokens=MAX_NEW_TOKENS
            )

            result = tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                spaces_between_special_tokens=False
            )
            results.append(result)
    return results

results = inference(model, tokenizer, dataloader)

# 最后将生成的结果写入文件中
results_clean = []
for result in results:
    results_clean.append(result)

# 生成完毕，需要将结果写入文件中以便评估
dataset = MbppDataset(filename)
dataset_clean = []
for i in range(len(results_clean)):
    # 每一条results_clean的数据都有k个结果
    dataset_clean.append({
        "task_id": dataset[i]["task_id"],
        "prompt": dataset[i]["prompt"],
        "label": results_clean[i]
    })

# 根据蒸馏的方式，需要修改文件名
output_filename = os.path.join(BASE_DIR, "dataset", "PERsD", "mbpp_persd.jsonl")
with open(output_filename, "w") as f:
    for data in dataset_clean:
        json_data = json.dumps(data)
        f.write(json_data + "\n")