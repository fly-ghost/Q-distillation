"""
对mbpp生成的内容进行处理 如果是decoder-only会产生额外的自然语言
"""

import json
import os

def clean_nl(completion):
    # 去除代码中提示的那一部分，提示不会有\n
    index = completion.find("\n")
    if index != -1:
        return completion[index+1:]
    return completion

def truncate(completion):
    # 去除掉最后一个不完整的def，并且保证至少有一个def
    group = completion.rsplit("def", 1)
    if group[0].find("def") != -1:
        return group[0]
    else:
        return completion
    
def clean_print(completion):
    # 去除最后一个print，因为大概率不完整，而且也不需要print
    index = completion.rfind("print")
    if index != -1:
        completion = completion[:index]
    return completion

def truncate_main(completion):
    index = completion.find("if __name")
    if index != -1:
        return completion[:index]
    return completion

model_name = "deepseek-small"
method = "standard"

filepath = os.path.join(model_name, method, "mbpp", "1", "samples-0.95-10-256")

results = []
with open(filepath, "r") as f:
    for line in f:
        json_obj = json.loads(line)
        completion = json_obj["completion"]
        if model_name.startswith("codet5p") is False:
            completion = clean_nl(completion)
        completion = truncate_main(completion)
        completion = truncate(completion)
        completion = clean_print(completion)
        data = {
            "task_id": json_obj["task_id"],
            "completion": completion
        }
        results.append(data)

output_filepath = os.path.join(model_name, method, "mbpp", "1", "samples-0.95-10-256-clean")
with open(output_filepath, "w") as f:
    for data in results:
        json_obj = json.dumps(data)
        f.write(json_obj + "\n")

