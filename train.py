import os
import torch
from transformers import TrainingArguments, Trainer

from config import *
from utils import setup_environment, clear_gpu_memory
from model.model_utils import (
    load_model_and_processor,
    add_lora_to_model,
    inspect_tokens,
)
from data.dataset import load_datasets, create_dataloader
from data.data_collator import AudioDataCollator, inspect_batch
from evaluation import run_model_evaluation


def main():
    # Setup
    setup_environment()

    # Load model and processor
    print(f"Loading model and processor from {MODEL_ID}...")
    model, processor = load_model_and_processor(
        MODEL_ID, DEVICE, TORCH_DTYPE, use_quantization=True
    )

    # Inspect tokenizer
    inspect_tokens(processor)

    # Add LoRA adapters
    print("Adding LoRA adapters to model...")
    model = add_lora_to_model(model, config=import_module("config"))

    # Load datasets
    print("Loading datasets...")
    df_train, df_val = load_datasets("train.csv", "val.csv")

    # Create data collator
    data_collator = AudioDataCollator(
        processor=processor, task_prompt=TASK_PROMPT, debug=DEBUG
    )

    # Test data collator with a small batch
    if DEBUG:
        print("Testing data collator with a small batch...")
        test_loader = create_dataloader(df_train.iloc[:2], data_collator, batch_size=1)
        for batch in test_loader:
            inspect_batch(batch, processor, debug=DEBUG)
            break

    # Prepare run name
    run_name = f"Qwen2-Audio-7B-Instruct-Lora-{EPOCHS}_ep-{SCHEDULE}_schedule-lr_{LEARNING_RATE}-bs_{BATCH_SIZE}"

    # Setup training arguments
    training_args = TrainingArguments(
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=WARMUP_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        logging_steps=LOGGING_STEPS,
        output_dir=OUTPUT_DIR,
        evaluation_strategy="steps",
        eval_steps=EVAL_STEPS,
        lr_scheduler_type=SCHEDULE,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        fp16=False,  # We're using BF16 instead
        bf16=True,
        remove_unused_columns=False,
        report_to="tensorboard",
        run_name=run_name,
        logging_dir=os.path.join(LOGGING_DIR, run_name),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        },  # For Qwen compatibility
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=df_train,
        eval_dataset=df_val,
        data_collator=data_collator,
    )

    # Start training
    print("Starting training...")
    trainer.train()

    # Save the model
    print(f"Saving model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

    # Evaluate the model
    print("Evaluating the model...")
    model.eval()
    run_model_evaluation(df_val, model, processor, DEVICE, TASK_PROMPT)


if __name__ == "__main__":
    try:
        from importlib import import_module

        main()
    except Exception as e:
        print(f"Error during execution: {e}")
    finally:
        clear_gpu_memory()
