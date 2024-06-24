import os


CURRENT_DIR = os.path.dirname(__file__)

BASE_DIR = os.path.dirname(CURRENT_DIR)


ORIGIN_MODEL_NAME = "codet5p-small"
ORIGIN_TOKENIZER_NAME = "codet5p-small"


STUDENT_MODEL_NAME = "codet5p-small"

TEACHER_MODEL_NAME = "codet5p-base"

PEFT_MODEL_NAME = "codet5p-peft"


STUDENT_TOKENIZER_NAME = "codet5p-small"

TEACHER_TOKENIZER_NAME = "codet5p-base"

PEFT_TOKENIZER_NAME = "codet5p-small"


FILENAME = "humaneval-personalized.jsonl"

FILENAME_INFERENCE = "humaneval-formatting.jsonl"


TEMP_MODEL_NAME = "codet5p-temp"
