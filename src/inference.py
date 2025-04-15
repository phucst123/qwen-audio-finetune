import torch
import librosa
from io import BytesIO
from urllib.request import urlopen
from src.config import DEBUG, TASK_PROMPT, MAX_LENGTH


def run_inference(audio_url: str, model, processor, device: str):
    try:
        audio, sr = librosa.load(BytesIO(urlopen(audio_url).read()), sr=None)

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": audio_url},
                    {"type": "text", "text": TASK_PROMPT},
                ],
            },
        ]
        text = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )

        if DEBUG:
            print("Template text:\n", text)

        audios = []
        sr_target = int(processor.feature_extractor.sampling_rate)
        audio_data, sr_loaded = librosa.load(
            BytesIO(urlopen(audio_url).read()), sr=sr_target
        )
        audios.append(audio_data)

        inputs = processor(
            text=text,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        ).to(device)

        generate_ids = model.generate(**inputs, max_new_tokens=256)
        generate_ids = generate_ids[:, inputs.input_ids.size(1) :]
        response = processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]

        print(f"Response: {response}")
        return response
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
