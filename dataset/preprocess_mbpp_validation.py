"""
将mbpp改成humaneval的格式 方便使用humaneval的方式进行评估
prompt分为prompt(函数及以上部分), prompt_nl(自然语言提示以及函数及以上部分one-shot)
"""

import json
import re

filename = "mbpp.jsonl"

results = []
with open(filename, "r") as f:
    for line in f:
        json_obj = json.loads(line)
        code = json_obj["code"]
        prompt = json_obj["text"]
        data = {
            "task_id": json_obj["task_id"],
            "prompt": json_obj["text"],
            "canonical_solution": json_obj["code"],
            "entry_point": "",
            "test": "def check(candidate):\n    " + "\n    ".join(json_obj["test_list"][:1])
        }
        # 提取函数签名，得到entry_point
        lines = code.split("\n")
        for single in lines:
            single = single.strip()
            if single.startswith("def"):
                left_kuot_index = single.find("(")
                function_signature = single[4:left_kuot_index]
                data["entry_point"] = function_signature
                break
        # 提取函数签名及以前的部分，合成prompt
        function_and_before = []
        i = 0
        while i < len(lines):
            function_and_before.append(lines[i])
            single = lines[i].strip()
            if single.startswith("def"):
                break
            i = i + 1
        data["prompt"] = data["prompt"] + "\n" + "\n".join(function_and_before) + "\n"
        data["prompt_code"] = "\n".join(function_and_before) + "\n"
        data["canonical_solution"] = "\n".join(lines[i+1:])
        results.append(data)

total = 0
for data in results:
    if data["entry_point"] != "":
        total += 1
print(total)

output_filename = "mbpp_validation.jsonl"
with open(output_filename, "w") as f:
    for data in results:
        json_obj = json.dumps(data)
        f.write(json_obj + "\n")
