import os
from tqdm import tqdm
from functools import partial
import json

from peft import AutoPeftModelForCausalLM, AutoPeftModelForSeq2SeqLM
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader

from utils.data import HumanEvalDataset
from utils.preprocess import tokenize, collate_fn_inference
from utils.parameters import MAX_LENGTH, MAX_NEW_TOKENS
from utils.path import PEFT_TOKENIZER_NAME, PEFT_MODEL_NAME, BASE_DIR, FILENAME_INFERENCE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer_name = os.path.join(BASE_DIR, "models", PEFT_TOKENIZER_NAME)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, padding_side="left")
# 部分分词器没有预先设置PAD和ATTENTION_MASK
if PEFT_MODEL_NAME in ["codegen-mono-peft", "codegen-multi-peft"]:
    tokenizer.pad_token = tokenizer.eos_token

model_name = os.path.join(BASE_DIR, "models", PEFT_MODEL_NAME)
model = None
if PEFT_MODEL_NAME in ["codet5p-peft"]:
    model = AutoPeftModelForSeq2SeqLM.from_pretrained(model_name)
else:
    model = AutoPeftModelForCausalLM.from_pretrained(model_name, is_decoder=True)
model = model.to(device)

filename = os.path.join(BASE_DIR, "dataset", FILENAME_INFERENCE)
dataset = HumanEvalDataset(filename, is_inference=True)

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
    collate_fn=collate_fn_inference
)

k = 10

def inference(model, tokenizer, dataloader):
    model.eval()
    results = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_masks = batch["attention_masks"].to(device)

            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_masks,
                pad_token_id=tokenizer.pad_token_id,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                num_beams=k,
                num_return_sequences=k,
                temperature=0.95,
                top_p=0.95
            )

            result = tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
            )
            results.append(result)
    return results

results = inference(model, tokenizer, dataloader)

results_clean = []
for result in results:
    for single in result:
        results_clean.append(single)

dataset = HumanEvalDataset(filename)
dataset_clean = []
for i in range(len(results_clean)):
    dataset_clean.append({
        "task_id": dataset[i//k]["task_id"],
        "completion": results_clean[i]
    })

output_filename = os.path.join(BASE_DIR, "evaluation", PEFT_MODEL_NAME, "mbpp_f_pass_" + str(k) + ".jsonl")
with open(output_filename, "w") as f:
    for data in dataset_clean:
        json_data = json.dumps(data)
        f.write(json_data + "\n")