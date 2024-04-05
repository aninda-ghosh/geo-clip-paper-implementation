from yacs.config import CfgNode as CN

_C = CN()

# Data Sequence
_C.DATA = CN()
_C.DATA.GPS_GALLERY = "D:/Kerner-Lab/geo-clip-paper-implementation/coordinates_100K.csv"
_C.DATA.DATASET_FILE = "D:/Kerner-Lab/user03/sampled_export/photo_metadata_resampled.csv"

# Model
_C.MODEL = CN()
_C.MODEL.GPS_QUEUE_SIZE = 1000
_C.MODEL.SEED_VALUE = 43

# Training
_C.TRAINING = CN()
_C.TRAINING.LEARNING_RATE = 3e-5
_C.TRAINING.WEIGHT_DECAY = 1e-6
_C.TRAINING.BATCH_SIZE = 25
_C.TRAINING.MAX_EPOCHS = 10
_C.TRAINING.NUM_WORKERS = 8
_C.TRAINING.SWA_LRS = 1e-3
_C.TRAINING.TRAIN_SPLIT = 0.75

# Validation
_C.VALIDATION = CN()
_C.VALIDATION.BATCH_SIZE = 16
_C.VALIDATION.NUM_WORKERS = 4