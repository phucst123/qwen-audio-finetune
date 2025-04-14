# Qwen2 Audio Fine-tuning

A framework for fine-tuning the Qwen2-Audio-7B-Instruct model for audio understanding tasks.

## Setup

```bash
pip install -r requirements.txt
```

## Directory Structure

- `config.py`: Configuration parameters
- `data/`: Dataset loading and processing
- `model/`: Model loading and utilities
- `inference.py`: Inference functionality
- `evaluation.py`: Evaluation script
- `train.py`: Training script
- `utils.py`: Common utilities

## Usage

### Training

```bash
python train.py
```

### Inference

```python
from model.model_utils import load_model_and_processor
from inference import run_inference
from config import MODEL_ID, DEVICE, TORCH_DTYPE, TASK_PROMPT

# Load model and processor
model, processor = load_model_and_processor(MODEL_ID, DEVICE, TORCH_DTYPE)

# Run inference
audio_url = "https://example.com/audio.wav"
response = run_inference(audio_url, processor, model, DEVICE, TASK_PROMPT)
print(response)
```

## Data Format

The training and validation data should be CSV files with at least the following columns:
- `id`: Unique identifier for the example
- `url`: URL of the audio file
- `description`: The target text (ground truth)

## Model

This code is designed for fine-tuning the Qwen2-Audio-7B-Instruct model using LoRA adapters. The model is designed to take audio input and produce text output for tasks like fill-in-the-blanks.
