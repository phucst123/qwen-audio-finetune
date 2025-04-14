from IPython.display import Audio, display
from urllib.request import urlopen
from io import BytesIO
import librosa
import torch


def display_audio(audio_url):
    """Display an audio player for the given URL."""
    try:
        audio, sr = librosa.load(BytesIO(urlopen(audio_url).read()), sr=None)
        display(Audio(audio, rate=sr))
        return audio, sr
    except Exception as e:
        print(f"Failed to load audio from {audio_url}: {e}")
        return None, None


def run_inference(audio_url, processor, model, device, task_prompt):
    """Run inference with the model on an audio input."""
    try:
        # Load and display the audio file
        audio, sr = display_audio(audio_url)
        if audio is None:
            return None

        # Create conversation format
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": audio_url},
                    {"type": "text", "text": task_prompt},
                ],
            },
        ]

        # Process the conversation
        text = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )

        # Process audio
        audios = []
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if ele["type"] == "audio" and "audio_url" in ele:
                        try:
                            sr = int(processor.feature_extractor.sampling_rate)
                            audio_data, sr_loaded = librosa.load(
                                BytesIO(urlopen(ele["audio_url"]).read()), sr=sr
                            )
                            audios.append(audio_data)
                        except Exception as e:
                            print(
                                f"Failed to process audio from {ele['audio_url']}: {e}"
                            )

        # Create model inputs
        inputs = processor(
            text=text,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        # Generate response
        with torch.no_grad():
            generate_ids = model.generate(**inputs, max_new_tokens=256)

        # Extract only the newly generated tokens
        generate_ids = generate_ids[:, inputs.input_ids.size(1) :]
        response = processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]

        print(f"Response: {response}")
        return response

    except Exception as e:
        print(f"An error occurred during inference: {e}")
        return None
