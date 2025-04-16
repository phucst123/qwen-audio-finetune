from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel
from torch.utils.data import DataLoader
from config import *

from data_collator import AudioDataCollator
from evaluation import run_model_evaluation
from custom_datasets import AudioDataset

from typing import Dict
import torch


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


def setup_trainer(model, processor, train_dataset, val_dataset, data_collator):

    training_args = TrainingArguments(
        label_names=["labels"],
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        warmup_steps=WARMUP_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        logging_steps=LOGGING_STEPS,
        output_dir=OUTPUT_DIR,
        eval_strategy="no",
        eval_steps=EVAL_STEPS,
        lr_scheduler_type=SCHEDULE,
        # save_strategy="steps",
        # save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        # fp16=True,  # if using Colab, but then you need to use bitsandbytes quantization
        bf16=True,
        remove_unused_columns=False,
        report_to="tensorboard",
        run_name=RUN_NAME,
        logging_dir=f"{LOGGING_DIR}/{RUN_NAME}",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    return trainer


def main():

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    base_model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=TORCH_DTYPE,
        # attn_implementation="flash_attention_2", # Do not use this for quantization, sometimes it will cause error
        # quantization_config=QUANT_CONFIG,
    ).to(DEVICE)

    # Apply LoRA
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        use_rslora=USE_RSLORA,
        target_modules=TARGET_MODULES,
        bias=BIAS,
        task_type=TASK_TYPE,
    )

    if PREVIOUS_MODEL_PATH:
        prev_lora_model = PeftModel.from_pretrained(base_model, PREVIOUS_MODEL_PATH)

        # Merge weights to incorporate previous LoRA adaptations
        model = prev_lora_model.merge_and_unload()

        # Apply new LoRA config to the merged model
    else:
        model = base_model

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Datasets
    train_dataset = AudioDataset(TRAIN_PATH)
    val_dataset = AudioDataset(VAL_PATH)
    test_dataset = AudioDataset(TEST_PATH)

    # Data collator
    data_collator = AudioDataCollator(processor=processor)

    # Inspect the batch
    if DEBUG:
        dataloader = DataLoader(
            train_dataset, batch_size=1, collate_fn=data_collator, shuffle=True
        )
        for batch in dataloader:
            inspect_batch(batch, processor)
            break

    # Trainer
    trainer = setup_trainer(model, processor, val_dataset, test_dataset, data_collator)

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

    # Evaluate the model
    model.eval()
    run_model_evaluation(test_dataset, model, processor, DEVICE)


if __name__ == "__main__":
    main()
