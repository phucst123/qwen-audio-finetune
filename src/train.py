from transformers import Trainer, TrainingArguments
from torch.utils.data import DataLoader
from src.config import (
    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    SCHEDULE,
    WARMUP_STEPS,
    WEIGHT_DECAY,
    LOGGING_STEPS,
    EVAL_STEPS,
    SAVE_STEPS,
    SAVE_TOTAL_LIMIT,
    OUTPUT_DIR,
    LOGGING_DIR,
    TRAIN_DIR,
    VAL_DIR,
    DEVICE,
    TASK_PROMPT,
)
from src.datasets import AudioDataset
from src.data_collator import AudioDataCollator
from src.utils import initialize_model_and_processor, apply_lora, inspect_batch
from src.evaluation import run_model_evaluation
from huggingface_hub import notebook_login


def setup_trainer(model, processor, train_dataset, val_dataset, data_collator):
    run_name = (
        f"Qwen2-Audio-7B-Instruct-Lora-{EPOCHS}_ep-"
        f"{SCHEDULE}_schedule-lr_{LEARNING_RATE}-bs_{BATCH_SIZE}"
    )

    training_args = TrainingArguments(
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        warmup_steps=WARMUP_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        logging_steps=LOGGING_STEPS,
        output_dir=OUTPUT_DIR,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        lr_scheduler_type=SCHEDULE,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        bf16=True,
        remove_unused_columns=False,
        report_to="tensorboard",
        run_name=run_name,
        logging_dir=f"{LOGGING_DIR}/{run_name}",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    return trainer


def main():
    # Login to Hugging Face
    notebook_login()

    # Load model and processor
    model, processor = initialize_model_and_processor()

    # Apply LoRA
    model = apply_lora(model)

    # Load datasets
    train_dataset = AudioDataset(TRAIN_DIR)
    val_dataset = AudioDataset(VAL_DIR)

    # Initialize data collator
    data_collator = AudioDataCollator(processor=processor)

    # Inspect a batch (optional, for debugging)
    if False:  # Set to True to inspect
        dataloader = DataLoader(
            train_dataset, batch_size=1, collate_fn=data_collator, shuffle=True
        )
        for batch in dataloader:
            inspect_batch(batch, processor)
            break

    # Setup trainer
    trainer = setup_trainer(model, processor, train_dataset, val_dataset, data_collator)

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

    # Evaluate the model
    model.eval()
    run_model_evaluation(val_dataset, model, processor, DEVICE)


if __name__ == "__main__":
    main()
