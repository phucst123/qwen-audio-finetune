import torch
from transformers import BitsAndBytesConfig

# Debug flag
DEBUG = False

# Model configuration
MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"
TORCH_DTYPE = torch.bfloat16  # Use torch.float16 for T4 GPUs
DEVICE = "cuda"
QUANT_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=TORCH_DTYPE,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# Data configuration
TRAIN_PATH = "train.csv"
VAL_PATH = "val.csv"
TEST_PATH = "test.csv"
TASK_PROMPT = "Describe the audio in detail"
MAX_LENGTH = 512

# Training configuration
LEARNING_RATE = 2e-5
EPOCHS = 1
SCHEDULE = "constant"  # Options: "constant", "cosine"
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 1
WARMUP_STEPS = 0  # 50
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 0.1
EVAL_STEPS = 0.2
SAVE_STEPS = 250
SAVE_TOTAL_LIMIT = 1

# LoRA configuration
LORA_R = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
USE_RSLORA = True
TARGET_MODULES = "all-linear"
BIAS = "none"
TASK_TYPE = "CAUSAL_LM"

# Output directories
OUTPUT_DIR = "./finetuned-model"
LOGGING_DIR = "./logs"

# Environment setup
import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
