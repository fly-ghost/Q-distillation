"""
替换label为个性化的结果
"""

import json
import os

def truncate(completion):
    group = completion.rsplit("def", 1)
    if group[0].find("def") != -1:
        return group[0]
    else:
        return completion

def truncate_main(completion):
    index = completion.find("if __name")
    if index != -1:
        return completion[:index]
    return completion

filename = "humaneval.jsonl"
data_list = []
with open(filename, "r") as f:
    for line in f:
        json_obj = json.loads(line)
        data_list.append(json_obj)

result_path = os.path.join("persd", "codet5p", "samples-256")
with open(result_path, "r") as f:
    i = 0
    for line in f:
        json_obj = json.loads(line)
        completion = json_obj["completion"]
        completion = truncate_main(completion)
        completion = truncate(completion)
        data_list[i]["canonical_solution"] = completion
        i += 1

output_path = os.path.join("persd", "codet5p", "humaneval.jsonl")
with open(output_path, "w") as f:
    for data in data_list:
        json_obj = json.dumps(data)
        f.write(json_obj + "\n")