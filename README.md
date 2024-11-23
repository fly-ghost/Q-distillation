# Title
Distilling large language models based on code execution feedback

# Introduction
We use the relevant indicators of code generation to distillate the large language model, construct Q table in a novel way, dynamically adjust the distillation temperature T, and complete the whole distillation process in a more efficient way.

# Steps
## Environment setting
```sh
pip3 install torch --index-url https://download.pytorch.org/whl/cu118
```
```sh
pip3 install transformers
```
```sh
pip install -r requirements.txt
```
## Download pre-training models
### codet5p-220m
https://huggingface.co/Salesforce/codet5p-220m/tree/main
Place the downloaded files into the models/codet5p-small folder
### codet5p-770m
https://huggingface.co/Salesforce/codet5p-770m/tree/main
Place the downloaded files into the models/codet5p-base folder

## Start distillation
```sh
python feedback_distillation.py --model="codet5p-small" --tokenizer="codet5p-small" --teacher_model="codet5p-base" --teacher_tokenizer="codet5p-base" --dataset="humaneval"
```
The models are saved to ```models/codet5p-small/temp``` for each iteration, with the best models saved to ```models/codet5p-small/feedback```.
# Evaluation
```sh
python main.py --model="codet5p-small" --tokenizer="codet5p-small" --method="feedback"
```
By running this command, the distilled student model is evaluated and the results are saved in ```samples/codet5p-small/feedback```.