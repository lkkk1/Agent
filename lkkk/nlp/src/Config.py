from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent.parent
CURRENT_ROOT_DIR = ROOT_DIR/"lkkk"/"nlp"

# 路径
RAW_DATA_DIR = CURRENT_ROOT_DIR /"data"/"raw"
PROCESSED_DATA_DIR = CURRENT_ROOT_DIR /"data"/"processed"
LOGS_DIR = CURRENT_ROOT_DIR /"logs"
MODELS_DIR = CURRENT_ROOT_DIR /"models"

# 训练参数
BATCH_SIZE = 64
LEARN_RATE = 0.001
EPOCHS = 50
MAX_SEQ_LEN = 128

# 模型结构
DIM_MODEL = 128
NUM_HEADS = 4
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2