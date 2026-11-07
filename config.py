import torch

# data
BLOCK_SIZE = 256
BATCH_SIZE = 32

# model
N_EMBED = 256
N_HEAD = 8
N_LAYER = 8
DROPOUT = 0
POS = 'rope'    # 'rope' | 'learned'
NORM = 'rms'    # 'rms' | 'layer'

# training
MAX_STEPS = 20000
EVAL_ITERS = 200
EVAL_INTERVAL = 1000
MAX_LR = .001
MIN_LR = .0001
WARMUP_STEPS = 200
SEED = 1

# sampling
TEMPERATURE = .8
TOP_K = 50
SAMPLE_TOKENS = 500

# paths
TRAIN_TXT = 'data/TinyStoriesV2-GPT4-train.txt'
VAL_TXT = 'data/TinyStoriesV2-GPT4-valid.txt'
TRAIN = 'data/train.bin'
VAL = 'data/val.bin'
CHECKPOINT = 'checkpoint.pt'

# derived
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
