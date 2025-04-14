from inference import run_inference


def run_model_evaluation(dataset, model, processor, device, task_prompt):
    """Run evaluation on a dataset and print results."""
    results = []

    print("\n===== MODEL EVALUATION =====\n")

    for idx, row in dataset.iterrows():
        print(f"\nExample {idx+1}/{len(dataset)}:")
        print(f"ID: {row.get('id', 'N/A')}")

        audio_url = row["url"]
        ground_truth = row["description"]

        print(f"Processing audio: {audio_url}")
        response = run_inference(audio_url, processor, model, device, task_prompt)

        print(f"Ground truth: {ground_truth}")
        print(f"Model output: {response}")

        results.append(
            {
                "id": row.get("id", idx),
                "ground_truth": ground_truth,
                "prediction": response,
            }
        )

        print("-" * 50)

    return results
