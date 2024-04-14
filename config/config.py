from yacs.config import CfgNode as CN

_C = CN()

# Data Sequence
_C.DATA = CN()
_C.DATA.GPS_GALLERY = "D:/kerner-lab/geo-clip-paper-implementation/coordinates_100K.csv"
_C.DATA.TRAIN_DATASET_PATH = "D:/kerner-lab/datasets/mp16/"
_C.DATA.EVAL_DATASET_PATH = "D:/kerner-lab/datasets/yfcc4k/"

# Model
_C.MODEL = CN()
_C.MODEL.CHECKPOINT_PATH = "D:/Kerner-Lab/geo-clip-paper-implementation/lightning_logs/version_0/checkpoints/epoch=8-step=5400.ckpt"
_C.MODEL.GPS_QUEUE_SIZE = 992
_C.MODEL.SEED_VALUE = 43

# Training
_C.TRAINING = CN()
_C.TRAINING.LEARNING_RATE = 3e-5
_C.TRAINING.WEIGHT_DECAY = 1e-6
_C.TRAINING.BATCH_SIZE = 32
_C.TRAINING.MAX_EPOCHS = 7
_C.TRAINING.NUM_WORKERS = 8
_C.TRAINING.SWA_LRS = 1e-3
_C.TRAINING.TRAIN_SPLIT = 0.8

# Validation
_C.VALIDATION = CN()
_C.VALIDATION.BATCH_SIZE = 24
_C.VALIDATION.NUM_WORKERS = 4