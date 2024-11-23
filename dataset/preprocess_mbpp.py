"""
把mbpp数据集的prompt更改为one-shot的形式 即自然语言加上一部分代码
同时把code变成去除掉函数签名及以上剩余的部分 去掉的部分单独列出来
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
        # 提取函数签名，得到entry_point
        lines = code.split("\n")
        # 提取函数签名及以前的部分，合成prompt
        function_and_before = []
        i = 0
        while i < len(lines):
            function_and_before.append(lines[i])
            single = lines[i].strip()
            if single.startswith("def"):
                break
            i = i + 1

        data = json_obj
        data["text"] = data["text"] + "\n" + "\n".join(function_and_before) + "\n"
        data["code"] = "\n".join(lines[i+1:])
        results.append(data)

output_filename = "mbpp_inference.jsonl"
with open(output_filename, "w") as f:
    for data in results:
        json_obj = json.dumps(data)
        f.write(json_obj + "\n")
