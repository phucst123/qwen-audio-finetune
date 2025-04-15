from src.inference import run_inference
from src.config import TASK_PROMPT


def run_model_evaluation(dataset, model, processor, device: str):
    for _, row in dataset.data.iterrows():
        audio = row["url"]
        results = run_inference(audio, model, processor, device)
        print(f"Ground truth answer: {row['description']}")
        print("\n")
