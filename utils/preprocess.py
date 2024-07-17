import torch

def tokenize(data, tokenizer, max_length=256, is_inference=True):
    # 对一行数据序列化，注意token的上限，以及填充
    # tokenizer本身会返回一个字典，字典中有input_ids字段
    # 对于一条数据，我们需要最终结果包含prompt(input_ids)，label
    # input_ids用于推理，label用于loss
    result = tokenizer(
        data["prompt"],
        max_length=max_length,
        padding="max_length",
        truncation=True
    )
    # 模型生成的其实是一个完整的结果，这里的label必须要完整
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
    # 对一行数据序列化，注意token的上限，以及填充
    # tokenizer本身会返回一个字典，字典中有input_ids字段
    # 对于一条数据，我们需要最终结果包含prompt(input_ids)，label
    # input_ids用于推理，label用于loss
    result = tokenizer(
        data["prompt"],
        max_length=max_length,
        padding="max_length",
        truncation=True
    )
    # 模型生成的其实是一个完整的结果，这里的label必须要完整
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
    # lines是batch_size个数据
    # 我们想通过batch["input_ids"]，获取batch_size个input_ids，并且都是tensor类型
    # 那么返回的input_ids一定是二维数组
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