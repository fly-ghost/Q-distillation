import json
from torch.utils.data import Dataset

class HumanEvalDataset(Dataset):
    def __init__(self, filename, is_inference=True, is_test=False):
        self.dataset = []
        # humaneval数据集，每一行都是一个json
        with open(filename, "r") as f:
            for line in f:
                json_obj = json.loads(line)
                # 有了json_obj，就可以通过关键字访问，这里只需要task_id, prompt
                data = {
                    "task_id": json_obj["task_id"],
                    "prompt": json_obj["prompt"],
                }
                if is_inference is False:
                    data["label"] = json_obj["label"]
                if is_test is True:
                    data["test"] = json_obj["test"]
                    data["entry_point"] = json_obj["entry_point"]
                self.dataset.append(data)

    def __getitem__(self, index):
        return self.dataset[index]
    
    def __len__(self):
        return len(self.dataset)
    
    def map(self, func):
        # 按照函数修改每一行数据，对于humaneval就是将文本变成token
        for i in range(len(self.dataset)):
            self.dataset[i] = func(self.dataset[i])

class MbppDataset(Dataset):
    def __init__(self, filename, is_inference=True):
        self.dataset = []
        # humaneval数据集，每一行都是一个json
        with open(filename, "r") as f:
            for line in f:
                json_obj = json.loads(line)
                # 有了json_obj，就可以通过关键字访问，这里只需要task_id, prompt
                data = {
                    "task_id": json_obj["task_id"],
                    "prompt": json_obj["prompt"],
                }
                if is_inference is False:
                    data["label"] = json_obj["label"]
                self.dataset.append(data)

    def __getitem__(self, index):
        return self.dataset[index]
    
    def __len__(self):
        return len(self.dataset)
    
    def map(self, func):
        # 按照函数修改每一行数据，对于humaneval就是将文本变成token
        for i in range(len(self.dataset)):
            self.dataset[i] = func(self.dataset[i])

class SoftLabelDataset(Dataset):
    def __init__(self, filename, is_inference=False):
        self.dataset = []
        with open(filename, "r") as f:
            for line in f:
                json_obj = json.loads(line)
                data = {
                    "task_id": json_obj["task_id"],
                    "prompt": json_obj["prompt"]
                }
                if is_inference is False:
                    data["label"] = json_obj["label"]
                self.dataset.append(data)

    def __getitem__(self, index):
        return self.dataset[index]
    
    def __len__(self):
        return len(self.dataset)
    
    def map(self, func):
        # 按照函数修改每一行数据，对于humaneval就是将文本变成token
        for i in range(len(self.dataset)):
            self.dataset[i] = func(self.dataset[i])

class FeedBackDataset(Dataset):
    def __init__(self, filename, is_inference=False):
        self.dataset = []
        with open(filename, "r") as f:
            for line in f:
                json_obj = json.loads(line)
                data = {
                    "task_id": json_obj["task_id"],
                    "prompt": json_obj["prompt"]
                }
                if is_inference is False:
                    data["label"] = json_obj["label"]
                self.dataset.append(data)

    def __getitem__(self, index):
        return self.dataset[index]
    
    def __len__(self):
        return len(self.dataset)
    
    def map(self, func):
        for i in range(len(self.dataset)):
            self.dataset[i] = func(self.dataset[i])
