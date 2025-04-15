import os
import numpy as np
import torch
from typing import Dict, List
import librosa
from io import BytesIO
from urllib.request import urlopen
from src.config import DEBUG, TASK_PROMPT, MAX_LENGTH


class AudioDataCollator:
    def __init__(self, processor):
        self.processor = processor
        self.task_prompt = TASK_PROMPT
        self.max_length = MAX_LENGTH or processor.feature_extractor.n_samples
        self.sampling_rate = processor.feature_extractor.sampling_rate

    def process_audio(self, audio_url: str) -> np.ndarray:
        try:
            if os.path.exists(audio_url):
                audio, sr = librosa.load(audio_url, sr=self.sampling_rate)
            else:
                audio, sr = librosa.load(
                    BytesIO(urlopen(audio_url).read()), sr=self.sampling_rate
                )
            if len(audio) > self.max_length:
                if DEBUG:
                    print(
                        f"Audio length {len(audio)} exceeds max length {self.max_length}. Truncating..."
                    )
                audio = audio[: self.max_length]
            return audio.astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"Failed to process audio from {audio_url}: {str(e)}")

    def __call__(self, examples):
        valid_examples = []
        audios = []
        combined_texts = []

        # Process each example
        for example in examples:
            try:
                # Process the audio
                audio = self.process_audio(example["url"])
                audios.append(audio)

                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio", "audio_url": example["url"]},
                            {"type": "text", "text": self.task_prompt},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": example["description"],
                    },
                ]
                combined_text = self.processor.apply_chat_template(
                    conversation, add_generation_prompt=False, tokenize=False
                )
                combined_texts.append(combined_text)

                valid_examples.append(example)
            except Exception as e:
                print(f"Failed to process example {example['id']}: {str(e)}")

        if not valid_examples:
            raise ValueError("No valid examples found in the batch.")

        if DEBUG:
            print(f"\n===== DEBUGGING INPUT =====\n")
            print(f"Number of combined texts: {len(combined_texts)}")
            print(f"Number of audios: {len(audios)}")

        try:
            inputs = self.processor(
                text=list(combined_texts),
                audio=list(audios),
                return_tensors="pt",
                padding=True,
            )
        except Exception as e:
            print(f"Processor error: {str(e)}")
            raise

        # Prepare the labels
        labels = inputs.input_ids.clone()

        if DEBUG:
            print("\n===== DEBUGGING INPUT =====\n")
            print(f"Input IDs shape: {inputs.input_ids.shape}")
            print(f"Input IDs: {inputs.input_ids}")
            print(f"Input IDs type: {type(inputs.input_ids)}")

        # Mask the prompt portion dynamically for each example
        for i in range(len(combined_texts)):
            try:
                # Ensure we handle indexing correctly
                input_ids_row = inputs["input_ids"][i]
                if DEBUG:
                    print(f"\nProcessing example {i}:")
                    print(f"Input IDs before masking: {input_ids_row}")

                # Find the tokenized sequence for "<|im_start|>assistant"
                assistant_start_tokens = self.processor.tokenizer.encode(
                    "<|im_start|>assistant", add_special_tokens=False
                )
                if DEBUG:
                    print(f"Assistant start token IDs: {assistant_start_tokens}")

                # Search for the sequence of tokens in the input
                assistant_start_idx = -1
                for j in range(len(input_ids_row) - len(assistant_start_tokens) + 1):
                    if torch.equal(
                        input_ids_row[j : j + len(assistant_start_tokens)],
                        torch.tensor(
                            assistant_start_tokens, device=input_ids_row.device
                        ),
                    ):
                        assistant_start_idx = j
                        break

                if DEBUG:
                    print(f"Assistant start index: {assistant_start_idx}")

                if assistant_start_idx != -1:  # Check if the sequence exists
                    # Mask everything before the assistant start token
                    labels[i, : assistant_start_idx + len(assistant_start_tokens)] = (
                        -100
                    )
                else:
                    # Fallback if the sequence is not found
                    print(
                        f"Warning: '<|im_start|>assistant' not found in input IDs for example {i}."
                    )
                    labels[i, :] = -100  # Mask all tokens if not found
            except Exception as e:
                print(f"Error processing example {i}: {str(e)}")

        if DEBUG:
            print("\n===== DEBUGGING PROCESSOR OUTPUT =====\n")
            print(f"Input IDs Shape: {inputs['input_ids'].shape}")
            print(f"Attention Mask Shape: {inputs['attention_mask'].shape}")
            print(f"Labels Shape: {labels.shape}")
            print(f"First label row (masked): {labels[0]}")

        return {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "input_features": inputs.input_features,
            "feature_attention_mask": inputs.feature_attention_mask,
            "labels": labels,
        }
