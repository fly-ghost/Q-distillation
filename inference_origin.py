import os

from tqdm import tqdm
from functools import partial
import json

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import torch
from torch.utils.data import DataLoader

from utils.data import HumanEvalDataset
from utils.preprocess import tokenize
from utils.parameters import MAX_LENGTH, MAX_NEW_TOKENS
from utils.path import ORIGIN_MODEL_NAME, ORIGIN_TOKENIZER_NAME, BASE_DIR, FILENAME_INFERENCE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer_name = os.path.join(BASE_DIR, "models", ORIGIN_TOKENIZER_NAME)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, padding_side="left")

if ORIGIN_MODEL_NAME in ["codegen-mono-small", "codegen-mono-base", "codegen-multi-small", "codegen-multi-base"]:
    tokenizer.pad_token = tokenizer.eos_token

model_name = os.path.join(BASE_DIR, "models", ORIGIN_MODEL_NAME)
model = None
if ORIGIN_MODEL_NAME in ["codet5p-base", "codet5p-small", "codet5p-large"]:
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
else:
    model = AutoModelForCausalLM.from_pretrained(model_name, is_decoder=True)
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

def collate_fn(lines):
    input_ids = []
    attention_masks = []
    for row in lines:
        input_ids.append(row["input_ids"])
        attention_masks.append(row["attention_mask"])
    
    return {
        "input_ids": torch.tensor(input_ids),
        "attention_masks": torch.tensor(attention_masks)
    }

batch_size = 4

dataloader = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    collate_fn=collate_fn
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
                input_ids,
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
                spaces_between_special_tokens=False
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


output_filename = os.path.join(BASE_DIR, "evaluation", ORIGIN_MODEL_NAME, "samples_pass_" + str(k) + ".jsonl")
with open(output_filename, "w") as f:
    for data in dataset_clean:
        json_data = json.dumps(data)
        f.write(json_data + "\n")