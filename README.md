# Qwen2 Audio Fine-tuning

A framework for fine-tuning the Qwen2-Audio-7B-Instruct model for audio understanding tasks.

## Setup

```bash
apt update && apt install ffmpeg -y
pip install -r requirements.txt
```

## Directory Structure

- `config.py`: Configuration parameters
- `custom_datasets.py`: Dataset loading 
- `data_collator.py`: Data loader and processing
- `inference.py`: Inference script
- `evaluation.py`: Evaluation script
- `train.py`: Training script

## Usage

Modify the path and config within the config.py file

### Training

```bash
python train.py
```

### Evaluation

```bash
python evaluation.py
```

## Data Format

The training and validation data should be CSV files with at least the following columns:
- `SourceAudioID`: Unique identifier for the example
- `url`: URL of the audio file
- `Description`: The target text (ground truth)

## Model

This code is designed for fine-tuning the Qwen2-Audio-7B-Instruct model using LoRA adapters. The model is designed to take audio input and produce text output for tasks like fill-in-the-blanks.
