import os


CURRENT_DIR = os.path.dirname(__file__)
# 项目根目录
BASE_DIR = os.path.dirname(CURRENT_DIR)

# 模型相关
# 原始模型，此名用于原始模型的推理
ORIGIN_MODEL_NAME = "codet5p-base"
ORIGIN_TOKENIZER_NAME = "codet5p-base"

# 学生模型名，此名用作微调
STUDENT_MODEL_NAME = "codet5p-small"
# 教师模型名，此名仅用于推理
TEACHER_MODEL_NAME = "codet5p-base"
# 微调后的模型名，此名仅用于推理
PEFT_MODEL_NAME = "codet5p-peft"

# 学生Tokenizer
STUDENT_TOKENIZER_NAME = "codet5p-small"
# 教师Tokenizer
TEACHER_TOKENIZER_NAME = "codet5p-base"
# 微调后的Tokenizer名，一般和原来的学生Tokenizer一样
PEFT_TOKENIZER_NAME = "codet5p-small"

# 数据集路径，用于微调
FILENAME = "mbpp-feedback.jsonl"
# 数据集路径，用于推理和验证
FILENAME_INFERENCE = "human-eval-formatting.jsonl"

# 用于暂时保存模型，或者最新模型的模型名
TEMP_MODEL_NAME = "codet5p-temp"
