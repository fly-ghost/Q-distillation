"""
模型相关代码
"""
from tqdm import tqdm
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from peft import get_peft_model, TaskType, LoraConfig, AutoPeftModelForCausalLM, AutoPeftModelForSeq2SeqLM

from utils.data import CustomDataset
from utils.preprocess import tokenize, collate_fn, collate_fn_inference

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path, is_peft=False, is_decoder=False, is_train=False, device_ids=[0]):
    model = None

    # is peft or not
    if is_peft is True:
        if is_decoder is True:
            model = AutoPeftModelForCausalLM.from_pretrained(model_path, is_decoder=True, trust_remote_code=True)
        else:
            model = AutoPeftModelForSeq2SeqLM.from_pretrained(model_path, trust_remote_code=True)
    else:
        if is_decoder is True:
            model = AutoModelForCausalLM.from_pretrained(model_path, is_decoder=True, trust_remote_code=True)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path, trust_remote_code=True)

    # is train or not
    if is_train is True:
        task_type = TaskType.CAUSAL_LM
        if is_decoder is False:
            task_type = TaskType.SEQ_2_SEQ_LM
        peft_config = LoraConfig(
            task_type=task_type, inference_mode=False, lora_alpha=32, lora_dropout=0.1
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    # 多卡环境需要分配device
    if len(device_ids) > 1:
        device = torch.device("cuda:{0}".format(device_ids[0]))
        model = nn.DataParallel(model, device_ids=device_ids)
    else:
        device = torch.device("cuda:0")
    model = model.to(device)

    return model

def load_tokenizer(tokenizer_path, tokenizer_name, is_train=False):
    padding_side = "left"
    if is_train is True:
        padding_side = "right"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side=padding_side)
    if tokenizer_name in ["codegen-mono-small", "codegen-mono-base", "codegen-multi-small", "codegen-multi-base"]:
        # tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_special_tokens({"pad_token":'<pad>'})

    return tokenizer

def load_dataset(dataset_path, dataset_name, is_train=False):
    dataset = CustomDataset(dataset_path, dataset_name, is_train=is_train)
    return dataset

def load_dataloader(dataset_path, dataset_name, tokenizer, max_length=256, batch_size=4, is_train=False, is_decoder=False):
    # 如果是仅解码器架构，是否需要<mask>?
    dataset = CustomDataset(dataset_path, dataset_name, is_train=is_train)
    tokenize_preprocessing = partial(
        tokenize,
        tokenizer=tokenizer,
        max_length=max_length,
        is_train=is_train,
        is_decoder=is_decoder
    )
    dataset.map(tokenize_preprocessing)
    dataloader = None
    if is_train is True:
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=collate_fn
        )
    else:
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=collate_fn_inference
        )
    return dataloader

def test_inference_greedy(model, tokenizer, dataloader, max_new_tokens=256, is_decoder=False):
    model.eval()
    results = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_masks = batch["attention_masks"].to(device)

            print(attention_masks)

            output = None
            if is_decoder is False:
                output = model(
                    input_ids=input_ids,
                    decoder_input_ids=input_ids,
                    attention_mask=attention_masks
                )
            else:
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_masks
                )

            # 观察logits能得到什么
            logits = output.logits
            probs = F.softmax(logits, dim=-1)
            generated_ids = torch.argmax(probs, dim=-1)

            result = tokenizer.batch_decode(
                generated_ids,
                spaces_between_special_tokens=False
            )
            task_ids = batch["task_ids"]
            for i in range(len(result)):
                data = {
                    "task_id": task_ids[i],
                    "completion": result[i]
                }
                results.append(data)
            break
    return results

def inference_greedy(model, tokenizer, dataloader, max_new_tokens=256):
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
                max_new_tokens=max_new_tokens
            )

            result = tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                spaces_between_special_tokens=False
            )
            task_ids = batch["task_ids"]
            for i in range(len(result)):
                data = {
                    "task_id": task_ids[i],
                    "completion": result[i]
                }
                results.append(data)

    return results

def inference(model, tokenizer, dataloader, max_new_tokens=256, k=10, temperature=0.95, top_p=0.95):
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
                max_new_tokens=max_new_tokens,
                do_sample=True,
                num_beams=k,
                num_return_sequences=k,
                temperature=temperature,
                top_p=top_p
            )

            result = tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                spaces_between_special_tokens=False
            )
            task_ids = batch["task_ids"]
            for i in range(len(result)):
                data = {
                    "task_id": task_ids[i//k],
                    "completion": result[i]
                }
                results.append(data)

    return results

class DistillationLoss(torch.nn.Module):
    """
    标准蒸馏损失函数
    """
    def __init__(self, T=1.0, a=0.5):
        super(DistillationLoss, self).__init__()
        self.first_loss_function = torch.nn.CrossEntropyLoss(reduction="mean")
        self.second_loss_function = torch.nn.KLDivLoss(log_target=True, reduction="batchmean")
        self.T = T
        self.a = a

    def get_T(self):
        return self.T

    def set_T(self, T):
        self.T = T

    def forward(self, logits, labels, teacher_logits):
        teacher_probs = F.log_softmax(teacher_logits / self.T, dim=-1)
        student_log_probs = F.log_softmax(logits, dim=-1)
        first_loss = self.first_loss_function(logits.permute(0, 2, 1), labels)
        second_loss = self.second_loss_function(student_log_probs, teacher_probs)
        total_loss = self.a * first_loss + (1 - self.a) * second_loss
        return total_loss

class FeedBackDistillationLoss(torch.nn.Module):
    def __init__(self, T=1.0):
        super(FeedBackDistillationLoss, self).__init__()
        self.second_loss_function = torch.nn.KLDivLoss(log_target=True, reduction="batchmean")
        # T用来控制平滑度，T越大，越平滑，概率差异越小
        self.T = T

    def get_T(self):
        return self.T

    def set_T(self, T):
        self.T = T

    def forward(self, logits, labels, teacher_logits):
        teacher_probs = F.log_softmax(teacher_logits / self.T, dim=-1)
        student_log_probs = F.log_softmax(logits, dim=-1)
        second_loss = self.second_loss_function(student_log_probs, teacher_probs)
        total_loss =  second_loss
        return total_loss

class CrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.loss_function = torch.nn.CrossEntropyLoss(reduction="mean")

    def forward(self, logits, labels):
        loss = self.loss_function(logits.permute(0, 2, 1), labels)
        return loss  

def fine_tune(model, dataloader, criterion, optimizer, scheduler, model_save_dir, epochs=10, is_decoder=True):
    for i in range(1, epochs + 1):
        print("start epoch: {0}".format(i))
        total_loss = 0.0
        min_loss = 0.0
        model.train()
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_masks = batch["attention_masks"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            output = None
            if is_decoder is True:
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_masks
                )
            else:
                output = model(
                    input_ids=input_ids,
                    decoder_input_ids=labels,
                    attention_mask=attention_masks
                )
            logits = output.logits

            loss = criterion(logits, labels)
            loss.backward()

            total_loss += loss.item()
            optimizer.step()
            scheduler.step()

        if min_loss == 0.0 or total_loss < min_loss:
            model.save_pretrained(model_save_dir)
        print("\tloss: {0}".format(total_loss / len(dataloader)))

def persd(model, dataloader, criterion, optimizer, scheduler, model_save_dir, agent, epochs=20, is_decoder=False, step=1):
    for i in range(1, epochs+1):
        print("start epoch: {0}".format(i))
        total_loss = 0.0
        model.train()
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_masks = batch["attention_masks"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            output = None
            if is_decoder is True:
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_masks
                )
            else:
                output = model(
                    input_ids=input_ids,
                    decoder_input_ids=labels,
                    attention_mask=attention_masks
                )
            logits = output.logits

            # loss = output.loss
            loss = criterion(logits, labels)
            loss.backward()

            total_loss += loss.item()
            optimizer.step()
            scheduler.step()
        
        if i % step == 0:
            agent.update()
        # modify the logic for saving the model
        if i % step == 0 and agent.is_rational():
            model.save_pretrained(model_save_dir)
        print("\tloss: {0}".format(total_loss / len(dataloader)))

def standard_distillation(student_model, teacher_model, dataloader, criterion, optimizer, scheduler, model_save_dir, epochs=10, is_decoder=False, device_ids=[0], teacher_device_ids=[0]):
    teacher_model.eval()
    for i in range(1, epochs+1):
        print("start epoch: {0}".format(i))
        total_loss = 0.0
        min_loss = 0.0
        student_model.train()
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device_ids[0])
            attention_masks = batch["attention_masks"].to(device_ids[0])
            labels = batch["labels"].to(device_ids[0])

            optimizer.zero_grad()

            student_output = None
            if is_decoder is True:
                student_output = student_model(
                    input_ids=input_ids,
                    attention_mask=attention_masks
                )
            else:
                student_output = student_model(
                    input_ids=input_ids,
                    decoder_input_ids=labels,
                    attention_mask=attention_masks
                )
            student_logits = student_output.logits

            input_ids = input_ids.to(teacher_device_ids[0])
            attention_masks = attention_masks.to(teacher_device_ids[0])
            labels = labels.to(teacher_device_ids[0])
            teacher_output = None
            with torch.no_grad():
                if is_decoder is True:
                    teacher_output = teacher_model(
                        input_ids=input_ids,
                        attention_mask=attention_masks
                    )
                else:
                    teacher_output = teacher_model(
                        input_ids=input_ids,
                        decoder_input_ids=labels,
                        attention_mask=attention_masks
                    )
            teacher_logits = teacher_output.logits


            student_logits = student_logits.to(teacher_device_ids[0])
            loss = criterion(student_logits, labels, teacher_logits)

            total_loss += loss.item()
            loss.backward()
            
            optimizer.step()
            scheduler.step()

        module = student_model
        if len(device_ids) > 1:
            module = student_model.module
        if min_loss == 0.0 or total_loss < min_loss:
            module.save_pretrained(model_save_dir)
        print("\tloss: {0}".format(total_loss / len(dataloader)))

def feedback_distillation(student_model, teacher_model, dataloader, criterion, optimizer, scheduler, model_save_dir, model_save_dir_temp, agent, epochs=10, is_decoder=True, device_ids=[0], teacher_device_ids=[0], step=1):
    teacher_model.eval()
    for i in range(1, epochs+1):
        print("start epoch: {0}".format(i))
        total_loss = 0.0
        student_model.train()
        for batch in tqdm(dataloader):
            # 多卡训练时，应当保证模型和数据在一张卡上才行
            input_ids = batch["input_ids"].to(device_ids[0])
            attention_masks = batch["attention_masks"].to(device_ids[0])
            labels = batch["labels"].to(device_ids[0])

            optimizer.zero_grad()

            student_output = None
            if is_decoder is True:
                student_output = student_model(
                    input_ids=input_ids,
                    attention_mask=attention_masks
                )
            else:
                student_output = student_model(
                    input_ids=input_ids,
                    decoder_input_ids=labels,
                    attention_mask=attention_masks
                )
            student_logits = student_output.logits

            # 张量和模型需要在一张卡上才能训练
            input_ids = input_ids.to(teacher_device_ids[0])
            attention_masks = attention_masks.to(teacher_device_ids[0])
            labels = labels.to(teacher_device_ids[0])
            teacher_output = None
            with torch.no_grad():
                if is_decoder is True:
                    teacher_output = teacher_model(
                        input_ids=input_ids,
                        attention_mask=attention_masks
                    )
                else:
                    teacher_output = teacher_model(
                        input_ids=input_ids,
                        decoder_input_ids=labels,
                        attention_mask=attention_masks
                    )
            teacher_logits = teacher_output.logits

            # 张量和张量之间也需要在一张卡上才能训练
            student_logits = student_logits.to(teacher_device_ids[0])
            loss = criterion(student_logits, labels, teacher_logits)

            total_loss += loss.item()
            loss.backward()
            
            optimizer.step()
            scheduler.step()
        
        if i % step == 0:
            agent.update()
            T = criterion.get_T()
            T_new = agent.get_next_T()
            criterion.set_T(T_new)
            print("\tT: {0} -> T_new: {1}".format(T, T_new))
        # modify the logic for saving the model
        module = student_model
        if len(device_ids) > 1:
            module = student_model.module
        if i % step == 0 and agent.is_rational():
            module.save_pretrained(model_save_dir)
        module.save_pretrained(model_save_dir_temp)
        print("\tloss: {0}".format(total_loss / len(dataloader)))