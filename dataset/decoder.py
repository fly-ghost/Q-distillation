"""
将prompt和label都变成prompt+label
"""
import json

filename = "humaneval.jsonl"

data_list = []
with open(filename, "r") as f:
    for line in f:
        json_obj = json.loads(line)
        combine = json_obj["prompt"] + json_obj["canonical_solution"]
        json_obj["prompt"] = combine
        json_obj["canonical_solution"] = combine
        data_list.append(json_obj)

output_filename = "humaneval-decoder.jsonl"
with open(output_filename, "w") as f:
    for data in data_list:
        f.write(json.dumps(data) + "\n")