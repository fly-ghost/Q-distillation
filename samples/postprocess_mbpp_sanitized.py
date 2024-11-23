import json
import os

model_name = "codet5p-small-finetuned"
dataset = "mbpp_sanitized"

filepath = os.path.join(model_name, dataset, "2", "samples-0.95-10-256")

def truncate(completion):
    group = completion.rsplit("def", 1)
    if group[0].find("def") != -1:
        return group[0]
    else:
        return completion
    
def clean_print(completion):
    index = completion.rfind("print")
    if index != -1:
        completion = completion[:index]
    return completion

def truncate_main(completion):
    index = completion.find("if __name")
    if index != -1:
        return completion[:index]
    return completion


results = []
with open(filepath, "r") as f:
    for line in f:
        json_obj = json.loads(line)
        completion = json_obj["completion"]
        completion = truncate(completion)
        # completion = clean_print(completion)
        completion = truncate_main(completion)
        data = {
            "task_id": json_obj["task_id"],
            "completion": completion
        }
        results.append(data)

output_filepath = os.path.join(model_name, dataset, "2", "samples-0.95-10-256-clean")
with open(output_filepath, "w") as f:
    for data in results:
        json_obj = json.dumps(data)
        f.write(json_obj + "\n")