import torch

# Model configuration
MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"
TORCH_DTYPE = torch.bfloat16  # torch.float16 for T4 GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEBUG = False

# Training configuration
TASK_PROMPT = "FILL IN THE BLANKS: "
LEARNING_RATE = 2e-5
EPOCHS = 1
SCHEDULE = "constant"  # "constant" or "cosine"
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 1
WARMUP_STEPS = 50
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 0.1
EVAL_STEPS = 0.2
SAVE_STEPS = 250
SAVE_TOTAL_LIMIT = 1

# LoRA configuration
LORA_R = 32
LORA_ALPHA = 32
USE_RSLORA = True
TARGET_MODULES = "all-linear"
LORA_DROPOUT = 0.1

# Output directories
OUTPUT_DIR = "./finetuned-model"
LOGGING_DIR = "./logs"
