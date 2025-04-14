import torch
from transformers import (
    Qwen2AudioForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model


def get_quant_config(torch_dtype):
    """Create quantization configuration."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


def load_model_and_processor(model_id, device, torch_dtype, use_quantization=False):
    """Load model and processor."""
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    model_kwargs = {
        "torch_dtype": torch_dtype,
    }

    if use_quantization:
        model_kwargs["quantization_config"] = get_quant_config(torch_dtype)

    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_id, **model_kwargs
    ).to(device)

    return model, processor


def add_lora_to_model(model, config):
    """Add LoRA adapters to the model."""
    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        use_rslora=config.USE_RSLORA,
        target_modules=config.TARGET_MODULES,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    print("Trainable parameters:")
    model.print_trainable_parameters()

    return model


def inspect_tokens(processor):
    """Print tokenizer special tokens information."""
    print(
        f"BOS token: {processor.tokenizer.bos_token} - ID: {processor.tokenizer.bos_token_id}"
    )
    print(
        f"EOS token: {processor.tokenizer.eos_token} - ID: {processor.tokenizer.eos_token_id}"
    )
    print(
        f"PAD token: {processor.tokenizer.pad_token} - ID: {processor.tokenizer.pad_token_id}"
    )
