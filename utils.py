import os
import gc
import torch
from huggingface_hub import notebook_login


def setup_environment():
    """Set up environment variables and login to HuggingFace."""
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    notebook_login()


def clear_gpu_memory():
    """Clear GPU memory cache."""
    gc.collect()
    torch.cuda.empty_cache()
