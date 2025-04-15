import os
import librosa
from io import BytesIO
from urllib.request import urlopen
from src.config import DEBUG, TASK_PROMPT, MAX_LENGTH


def run_inference(audio_url: str, model, processor, device: str):
    try:
        # if os.path.exists(audio_url):
        #     audio, sr = librosa.load(audio_url, sr=None)
        # else:
        #     audio, sr = librosa.load(BytesIO(urlopen(audio_url).read()), sr=None)

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

        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if ele["type"] == "audio" and "audio_url" in ele:
                        try:
                            sr_loaded = int(processor.feature_extractor.sampling_rate)

                            # Load audio based on path type
                            audio_path = ele["audio_url"]
                            if os.path.exists(audio_path):
                                audio_data, _ = librosa.load(audio_path, sr=sr_loaded)
                            else:
                                audio_data, _ = librosa.load(
                                    BytesIO(urlopen(audio_path).read()), sr=sr_loaded
                                )

                            if DEBUG:
                                print(f"Audio loaded with sample rate: {sr_loaded}")

                            audios.append(audio_data)
                        except Exception as e:
                            print(
                                f"Failed to process audio from {ele['audio_url']}: {e}"
                            )

        inputs = processor(
            text=text,
            audio=audios,
            return_tensors="pt",
            padding=True,
            # max_length=512,
        ).to(device)

        generate_ids = model.generate(**inputs, max_new_tokens=256)
        if DEBUG:
            print(
                f"Input shape: {inputs.input_ids.size(1)}, Generated shape: {generate_ids.shape}"
            )
        generate_ids = generate_ids[:, inputs.input_ids.size(1) :]
        response = processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]

        return response
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
