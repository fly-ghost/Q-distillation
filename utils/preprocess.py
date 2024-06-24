import torch

def tokenize(data, tokenizer, max_length=256, is_inference=True):
    result = tokenizer(
        data["prompt"],
        max_length=max_length,
        padding="max_length",
        truncation=True
    )
    if is_inference is False:
        result["label"] = tokenizer(
            data["prompt"],
            data["label"],
            max_length=max_length,
            padding="max_length",
            truncation=True
        )["input_ids"]

        result["teacher_label"] = tokenizer(
            data["prompt"],
            data["teacher_label"],
            max_length=max_length,
            padding="max_length",
            truncation=True
        )["input_ids"]

    return result

def soft_label_tokenize(data, tokenizer, max_length=256, is_inference=True):
    result = tokenizer(
        data["prompt"],
        max_length=max_length,
        padding="max_length",
        truncation=True
    )
    if is_inference is False:
        result["label"] = tokenizer(
            data["prompt"],
            data["label"],
            max_length=max_length,
            padding="max_length",
            truncation=True
        )["input_ids"]

    return result

def feedback_tokenize(data, tokenizer, max_length=256, is_inference=True):
    result = tokenizer(
        data["prompt"],
        max_length=max_length,
        padding="max_length",
        truncation=True
    )
    if is_inference is False:
        result["label"] = tokenizer(
            data["prompt"],
            data["label"],
            max_length=max_length,
            padding="max_length",
            truncation=True
        )["input_ids"]

    return result

def collate_fn(lines):
    input_ids = []
    labels = []
    attention_masks = []
    for row in lines:
        input_ids.append(row["input_ids"])
        labels.append(row["label"])
        attention_masks.append(row["attention_mask"])
    
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_masks": torch.tensor(attention_masks)
    }

def collate_fn_inference(lines):
    input_ids = []
    attention_masks = []
    for row in lines:
        input_ids.append(row["input_ids"])
        attention_masks.append(row["attention_mask"])
    
    return {
        "input_ids": torch.tensor(input_ids),
        "attention_masks": torch.tensor(attention_masks)
    }