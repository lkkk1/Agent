from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent.parent
CURRENT_ROOT_DIR = ROOT_DIR/"lkkk"/"pre_train_bertcls"

# 路径
RAW_DATA_DIR = CURRENT_ROOT_DIR /"data"/"raw"
PROCESSED_DATA_DIR = CURRENT_ROOT_DIR /"data"/"processed"
LOGS_DIR = CURRENT_ROOT_DIR /"logs"
MODELS_DIR = CURRENT_ROOT_DIR /"models"

# 训练参数
BATCH_SIZE = 16
LEARN_RATE = 1e-5
EPOCHS = 1
MAX_SEQ_LEN = 128