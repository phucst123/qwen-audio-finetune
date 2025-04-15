from inference import run_inference
from config import TASK_PROMPT


def run_model_evaluation(dataset):
    for record in dataset:
        audio = record["url"]
        results = run_inference(audio)
        print(f"Ground truth: {record['description']}")
        print("\n")
