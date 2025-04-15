from inference import run_inference
from config import TASK_PROMPT, OUTPUT_DIR, TORCH_DTYPE, TEST_PATH, DEVICE
from custom_datasets import AudioDataset
from peft import PeftModel, PeftConfig
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor


def run_model_evaluation(dataset, model, processor, device):
    for record in dataset:
        audio = record["url"]
        results = run_inference(audio, model, processor, device)
        print(f"URL: {audio}")
        print(f"Ground truth: {record['description']}")
        print(f"Response: {results}")
        print("\n")


def main():
    config = PeftConfig.from_pretrained(OUTPUT_DIR)
    base_model = Qwen2AudioForConditionalGeneration.from_pretrained(
        config.base_model_name_or_path,
        torch_dtype=TORCH_DTYPE,
        trust_remote_code=True
    ).to(DEVICE)
    
    # Load the LoRA model
    model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
    
    # Load the processor
    processor = AutoProcessor.from_pretrained(OUTPUT_DIR, trust_remote_code=True)
    
    # Set to evaluation mode
    model.eval()

    test_dataset = AudioDataset(TEST_PATH)
    run_model_evaluation(test_dataset, model, processor, DEVICE)


if __name__ == "__main__":
    main()