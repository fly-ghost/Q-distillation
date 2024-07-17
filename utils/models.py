"""
模型相关代码
"""
import os
from tqdm import tqdm
import json

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
from peft import get_peft_model, TaskType, LoraConfig

from utils.path import BASE_DIR, PEFT_MODEL_NAME, TEMP_MODEL_NAME

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def is_decoder_model(model_name):
    if model_name in ["codet5p-small", "codet5p-base"]:
        return False
    else:
        return True

def load_model(model_name, is_student=True):
    model_path = os.path.join(BASE_DIR, "models", model_name)
    model = None
    if is_decoder_model(model_name) is False:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, is_decoder=True)

    if is_student is True:
        task_type = TaskType.CAUSAL_LM
        if is_decoder_model(model_name) is False:
            task_type = TaskType.SEQ_2_SEQ_LM
        peft_config = LoraConfig(
            task_type=task_type, inference_mode=False, lora_alpha=32, lora_dropout=0.1
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    model = model.to(device)

    return model

def load_peft_model(model_name):
    pass

class DistillationLoss(torch.nn.Module):
    """
    标准蒸馏损失函数
    """
    def __init__(self, T=1.0, a=0.25):
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

class FeedBackWithRememberDistillationLoss(torch.nn.Module):
    """
    记忆化蒸馏, 实际上就是把学生模型生成的labels当作目标labels
    """
    def __init__(self, T=1.0, a=0.25):
        super(FeedBackWithRememberDistillationLoss, self).__init__()
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

class PERsDLoss(torch.nn.Module):
    def __init__(self, T=1.0):
        super(FeedBackWithRememberDistillationLoss, self).__init__()
        self.first_loss_function = torch.nn.CrossEntropyLoss(reduction="mean")
        self.T = T
        
    def get_T(self):
        return self.T

    def set_T(self, T):
        self.T = T

    def forward(self, logits, labels):
        first_loss = self.first_loss_function(logits.permute(0, 2, 1), labels)
        return first_loss
        
def train(student_model, teacher_model, dataloader, criterion, optimizer, scheduler, epochs=10, is_decoder=True):
    """
    一个标准蒸馏的过程, 每一次epoch都会保存一次模型
    """
    teacher_model.eval()
    for i in range(1, epochs+1):
        print("start epoch: {0}".format(i))
        total_loss = 0
        student_model.train()
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_masks = batch["attention_masks"].to(device)
            labels = batch["labels"].to(device)

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

            teacher_output = None
            with torch.no_grad():
                if is_decoder:
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

            loss = criterion(student_logits, labels, teacher_logits)

            total_loss += loss.item()
            loss.backward()
            
            optimizer.step()
            scheduler.step()
        student_model.save_pretrained(os.path.join(BASE_DIR, "models", PEFT_MODEL_NAME))
        print("\tloss: {0}".format(total_loss / len(dataloader)))

def standard_train(student_model, teacher_model, dataloader, criterion, optimizer, scheduler, agent, epochs=10, is_decoder=True, beta=100):
    """
    一个标准蒸馏的过程, 每一次epoch都会保存一次模型
    """
    # 记录温度T的变化
    T_list = []
    T_list.append(criterion.get_T())
    # 记录q值的变化
    Q_list = []
    Q_list.append(agent.get_q().item())
    # 记录loss的变化
    Loss_list = []
    with open(os.path.join(BASE_DIR, "values", PEFT_MODEL_NAME, "values_standard.jsonl"), "a+") as f:
        json_obj = json.dumps({
            "id": 0,
            "T": T_list[-1],
            "Q": Q_list[-1],
            "Loss": 0.0,
            "Success Rate": agent.get_success_rate(),
            "Compile Rate": agent.get_compile_rate()
        })
        f.write(json_obj + "\n")
    teacher_model.eval()
    for i in range(1, epochs+1):
        print("start epoch: {0}".format(i))
        total_loss = 0
        student_model.train()
        print("\tlr: ", scheduler.get_lr()[0])
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_masks = batch["attention_masks"].to(device)
            labels = batch["labels"].to(device)

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

            loss = criterion(student_logits, labels, teacher_logits)

            total_loss += loss.item()
            loss.backward()
            
            optimizer.step()
            scheduler.step()
        # 一轮训练结束，更新agent，根据agent的更新频率更新
        if agent.is_update_available(i):
            agent.reset()
            agent.update_all()
            # 更新损失函数的T值
            T = criterion.get_T()
            T_new = agent.get_T(beta=beta)
            criterion.set_T(T)
            print("\tT: {0} -> T_new: {1}".format(T, T_new))
            if agent.is_rational():
                student_model.save_pretrained(os.path.join(BASE_DIR, "models", PEFT_MODEL_NAME))
        student_model.save_pretrained(os.path.join(BASE_DIR, "models", TEMP_MODEL_NAME))
        print("\tloss: {0}".format(total_loss / len(dataloader)))
        T_list.append(criterion.get_T())
        Q_list.append(agent.get_q().item())
        Loss_list.append(total_loss / len(dataloader))
        # 一轮结束，将收集到的数据写入文件
        with open(os.path.join(BASE_DIR, "values", PEFT_MODEL_NAME, "values_standard.jsonl"), "a+") as f:
            json_obj = json.dumps({
                "id": i,
                "T": T_list[-1],
                "Q": Q_list[-1],
                "Loss": Loss_list[-1],
                "Success Rate": agent.get_success_rate(),
                "Compile Rate": agent.get_compile_rate()
            })
            f.write(json_obj + "\n")

def feedback_train(student_model, teacher_model, dataloader, criterion, optimizer, scheduler, agent, epochs=10, is_decoder=True, beta=100):
    """
    一个标准蒸馏的过程, 每一次epoch都会保存一次模型
    """
    # 记录温度T的变化
    T_list = []
    T_list.append(criterion.get_T())
    # 记录q值的变化
    Q_list = []
    Q_list.append(agent.get_q().item())
    # 记录loss的变化
    Loss_list = []
    with open(os.path.join(BASE_DIR, "values", PEFT_MODEL_NAME, "values_feedback.jsonl"), "a+") as f:
        json_obj = json.dumps({
            "id": 0,
            "T": T_list[-1],
            "Q": Q_list[-1],
            "Loss": 0.0,
            "Success Rate": agent.get_success_rate(),
            "Compile Rate": agent.get_compile_rate()
        })
        f.write(json_obj + "\n")
    teacher_model.eval()
    for i in range(1, epochs+1):
        print("start epoch: {0}".format(i))
        total_loss = 0
        student_model.train()
        print("\tlr: ", scheduler.get_lr()[0])
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_masks = batch["attention_masks"].to(device)
            labels = batch["labels"].to(device)

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

            loss = criterion(student_logits, labels, teacher_logits)

            total_loss += loss.item()
            loss.backward()
            
            optimizer.step()
            scheduler.step()
        # 一轮训练结束，更新agent，根据agent的更新频率更新
        if agent.is_update_available(i):
            agent.reset()
            agent.update_all()
            # 更新损失函数的T值
            T = criterion.get_T()
            T_new = agent.get_T(beta=beta)
            criterion.set_T(T_new)
            print("\tT: {0} -> T_new: {1}".format(T, T_new))
            if agent.is_rational():
                student_model.save_pretrained(os.path.join(BASE_DIR, "models", PEFT_MODEL_NAME))
        student_model.save_pretrained(os.path.join(BASE_DIR, "models", TEMP_MODEL_NAME))
        # torch.save(optimizer.state_dict(), os.path.join(BASE_DIR, "optimizers", PEFT_MODEL_NAME))
        print("\tloss: {0}".format(total_loss / len(dataloader)))
        T_list.append(criterion.get_T())
        Q_list.append(agent.get_q().item())
        Loss_list.append(total_loss / len(dataloader))
        # 一轮结束，将收集到的数据写入文件
        with open(os.path.join(BASE_DIR, "values", PEFT_MODEL_NAME, "values_feedback.jsonl"), "a+") as f:
            json_obj = json.dumps({
                "id": i,
                "T": T_list[-1],
                "Q": Q_list[-1],
                "Loss": Loss_list[-1],
                "Success Rate": agent.get_success_rate(),
                "Compile Rate": agent.get_compile_rate()
            })
            f.write(json_obj + "\n")

def persd_train(student_model, dataloader, criterion, optimizer, scheduler, agent, epochs=10, is_decoder=True, beta=100):
    # 记录温度T的变化
    T_list = []
    T_list.append(criterion.get_T())
    # 记录q值的变化
    Q_list = []
    Q_list.append(agent.get_q().item())
    # 记录loss的变化
    Loss_list = []
    with open(os.path.join(BASE_DIR, "values", PEFT_MODEL_NAME, "values_persd.jsonl"), "a+") as f:
        json_obj = json.dumps({
            "id": 0,
            "T": T_list[-1],
            "Q": Q_list[-1],
            "Loss": 0.0,
            "Success Rate": agent.get_success_rate(),
            "Compile Rate": agent.get_compile_rate()
        })
        f.write(json_obj + "\n")
    for i in range(1, epochs+1):
        print("start epoch: {0}".format(i))
        total_loss = 0
        student_model.train()
        print("\tlr: ", scheduler.get_lr()[0])
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_masks = batch["attention_masks"].to(device)
            labels = batch["labels"].to(device)

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

            loss = criterion(student_logits, labels)

            total_loss += loss.item()
            loss.backward()
            
            optimizer.step()
            scheduler.step()
        # 一轮训练结束，更新agent，根据agent的更新频率更新
        if agent.is_update_available(i):
            agent.reset()
            agent.update_all()
            # 更新损失函数的T值
            T = criterion.get_T()
            T_new = agent.get_T(beta=beta)
            criterion.set_T(T)
            print("\tT: {0} -> T_new: {1}".format(T, T_new))
            if agent.is_rational():
                student_model.save_pretrained(os.path.join(BASE_DIR, "models", PEFT_MODEL_NAME))
        student_model.save_pretrained(os.path.join(BASE_DIR, "models", TEMP_MODEL_NAME))
        # torch.save(optimizer.state_dict(), os.path.join(BASE_DIR, "optimizers", PEFT_MODEL_NAME))
        print("\tloss: {0}".format(total_loss / len(dataloader)))
        T_list.append(criterion.get_T())
        Q_list.append(agent.get_q().item())
        Loss_list.append(total_loss / len(dataloader))
        # 一轮结束，将收集到的数据写入文件
        with open(os.path.join(BASE_DIR, "values", PEFT_MODEL_NAME, "values_persd.jsonl"), "a+") as f:
            json_obj = json.dumps({
                "id": i,
                "T": T_list[-1],
                "Q": Q_list[-1],
                "Loss": Loss_list[-1],
                "Success Rate": agent.get_success_rate(),
                "Compile Rate": agent.get_compile_rate()
            })
            f.write(json_obj + "\n")