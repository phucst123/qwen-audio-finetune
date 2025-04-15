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
            audio, sr = librosa.load(
                BytesIO(urlopen(audio_url).read()), sr=self.sampling_rate
            )
            if len(audio) > self.max_length:
                if DEBUG:
                    print(
                        f"Audio length {len(audio)} exceeds max length {self.max_length}. Truncating."
                    )
                audio = audio[: self.max_length]
            return audio.astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"Failed to process audio from {audio_url}: {str(e)}")

    def __call__(self, examples):
        valid_examples = []
        audios = []
        combined_texts = []

        for example in examples:
            try:
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
                    {"role": "assistant", "content": example["description"]},
                ]
                combined_text = self.processor.apply_chat_template(
                    conversation, add_generation_prompt=True, tokenize=False
                )
                combined_texts.append(combined_text)
                valid_examples.append(example)
            except Exception as e:
                print(f"Failed to process example {example['id']}: {str(e)}")

        if not valid_examples:
            raise ValueError("No valid examples found in the batch.")

        if DEBUG:
            print(f"Number of combined texts: {len(combined_texts)}")
            print(f"Number of audios: {len(audios)}")

        inputs = self.processor(
            text=combined_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        labels = inputs.input_ids.clone()
        for i in range(len(combined_texts)):
            input_ids_row = inputs["input_ids"][i]
            assistant_start_tokens = self.processor.tokenizer.encode(
                "<|im_start|>assistant", add_special_tokens=False
            )
            assistant_start_idx = -1
            for j in range(len(input_ids_row) - len(assistant_start_tokens) + 1):
                if torch.equal(
                    input_ids_row[j : j + len(assistant_start_tokens)],
                    torch.tensor(assistant_start_tokens),
                ):
                    assistant_start_idx = j
                    break
            if assistant_start_idx != -1:
                labels[i, : assistant_start_idx + len(assistant_start_tokens)] = -100
            else:
                print(f"Warning: '<|im_start|>assistant' not found in example {i}.")
                labels[i, :] = -100

        return {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "input_features": inputs.input_features,
            "feature_attention_mask": inputs.feature_attention_mask,
            "labels": labels,
        }
