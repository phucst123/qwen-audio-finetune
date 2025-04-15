from typing import Dict
import torch
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model
from src.config import (
    MODEL_ID,
    TORCH_DTYPE,
    DEVICE,
    QUANT_CONFIG,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    USE_RSLORA,
    TARGET_MODULES,
    BIAS,
    TASK_TYPE,
)


def initialize_model_and_processor():
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=TORCH_DTYPE,
        # quantization_config=QUANT_CONFIG,
    ).to(DEVICE)
    return model, processor


def apply_lora(model):
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        use_rslora=USE_RSLORA,
        target_modules=TARGET_MODULES,
        bias=BIAS,
        task_type=TASK_TYPE,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def inspect_batch(batch: Dict[str, torch.Tensor], processor) -> None:
    print("\n===== DETAILED BATCH INSPECTION =====\n")
    for key, tensor in batch.items():
        print(f"\n{key}")
        print(f"Shape: {tensor.shape}")
        print(f"Type: {tensor.dtype}")
        print(f"Device: {tensor.device}")
        if key not in ["input_features", "feature_attention_mask"]:
            print(f"First row: {tensor[0].tolist()}")

    if "input_ids" in batch:
        print("\n===== DECODED Input_ids =====\n")
        first_input = batch["input_ids"][0].tolist()
        decoded_input = processor.tokenizer.decode(
            first_input, skip_special_tokens=False
        )
        print(f"Decoded input (First row):\n{decoded_input}")

    if "labels" in batch:
        print("\n===== DECODED Labels =====\n")
        first_label = batch["labels"][0].tolist()
        first_label = [token for token in first_label if token != -100]
        decoded_label = processor.tokenizer.decode(
            first_label, skip_special_tokens=False
        )
        print(f"Decoded label (First row):\n{decoded_label}")
