# tokenizer的最大token数
MAX_LENGTH = 256
# model.generate最多额外的token数，会极大影响生成时间
MAX_NEW_TOKENS = 128
# batch的大小，影响微调时间，设置越大占用显存越多，微调时间越快
BATCH_SIZE = 8
# 推理的batch大小
BATCH_SIZE_INFERENCE = 16
# 模型训练的温度T，T越高越平滑
T_PRESET = 1.0
# 损失函数的权值
A_PRESET = 0.25