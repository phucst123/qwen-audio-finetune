import torch
import numpy as np
import librosa
from urllib.request import urlopen
from io import BytesIO


class AudioDataCollator:
    def __init__(self, processor, task_prompt, max_length=None, debug=False):
        self.processor = processor
        self.task_prompt = task_prompt
        self.max_length = max_length or processor.feature_extractor.n_samples
        self.sampling_rate = processor.feature_extractor.sampling_rate
        self.debug = debug

    def process_audio(self, audio_url):
        """Process and load an audio file from URL."""
        try:
            audio, sr = librosa.load(
                BytesIO(urlopen(audio_url).read()), sr=self.sampling_rate
            )

            # Truncate if needed
            if len(audio) > self.max_length:
                if self.debug:
                    print(
                        f"Audio length {len(audio)} is longer than max length {self.max_length}. Truncating."
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

        if self.debug:
            print(f"\n===== DEBUGGING INPUT =====\n")
            print(f"Number of combined texts: {len(combined_texts)}")
            print(f"Number of audios: {len(audios)}")

        try:
            inputs = self.processor(
                text=combined_texts,
                audio=audios,
                return_tensors="pt",
                padding=True,
            )
        except Exception as e:
            print(f"Processor error: {str(e)}")
            raise

        # Prepare the labels
        labels = inputs.input_ids.clone()

        if self.debug:
            print("\n===== DEBUGGING INPUT =====\n")
            print(f"Input IDs shape: {inputs.input_ids.shape}")

        # Mask the prompt portion dynamically for each example
        for i in range(len(combined_texts)):
            try:
                # Ensure we handle indexing correctly
                input_ids_row = inputs["input_ids"][i]
                if self.debug:
                    print(f"\nProcessing example {i}:")

                # Find the tokenized sequence for "<|im_start|>assistant"
                assistant_start_tokens = self.processor.tokenizer.encode(
                    "<|im_start|>assistant", add_special_tokens=False
                )
                if self.debug:
                    print(f"Assistant start token IDs: {assistant_start_tokens}")

                # Search for the sequence of tokens in the input
                assistant_start_idx = -1
                for j in range(len(input_ids_row) - len(assistant_start_tokens) + 1):
                    if torch.equal(
                        input_ids_row[j : j + len(assistant_start_tokens)],
                        torch.tensor(assistant_start_tokens),
                    ):
                        assistant_start_idx = j
                        break

                if self.debug:
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

        return {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "input_features": inputs.input_features,
            "feature_attention_mask": inputs.feature_attention_mask,
            "labels": labels,
        }


def inspect_batch(batch, processor, debug=False):
    """Inspect a batch of data for debugging."""
    if not debug:
        return

    print("\n===== DETAILED BATCH INSPECTION =====\n")

    for key, tensor in batch.items():
        print(f"\n{key}")
        print(f"Shape: {tensor.shape}")
        print(f"Type: {tensor.dtype}")
        print(f"Device: {tensor.device}")
        if key not in ["input_features", "feature_attention_mask"]:
            print(f"First row: {tensor[0].tolist()}")

    if "input_ids" in batch:
        print("\n===== DECODED Input_ids =====\n")
        first_input = batch["input_ids"][0].tolist()
        decoded_input = processor.tokenizer.decode(
            first_input, skip_special_tokens=False
        )
        print(f"Decoded input (First row):\n{decoded_input}")

    if "labels" in batch:
        print("\n===== DECODED Labels =====\n")
        first_label = batch["labels"][0].tolist()
        # Filter out -100 values for decoding
        first_label = [token for token in first_label if token != -100]
        # Decode the labels
        decoded_label = processor.tokenizer.decode(
            first_label, skip_special_tokens=False
        )
        print(f"Decoded label (First row):\n{decoded_label}")
