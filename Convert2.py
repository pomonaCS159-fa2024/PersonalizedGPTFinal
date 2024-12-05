import whisper
import os
import ssl
import numpy as np
from pydub import AudioSegment
from pydub.silence import split_on_silence

audio_folder = "/Users/avga2021/Desktop/NLP_Final2/speech_to_text/Audio"
output_folder = "/Users/avga2021/Desktop/NLP_Final2/Output2"
preprocessed_folder = "/Users/avga2021/Desktop/NLP_Final2/PreprocessedAudio"

ssl._create_default_https_context = ssl._create_unverified_context

model = whisper.load_model("base")  # Balanced for speed and accuracy
os.environ["PATH"] += os.pathsep + os.path.expanduser("~/ffmpeg")


os.makedirs(output_folder, exist_ok=True)
os.makedirs(preprocessed_folder, exist_ok=True)

# Preprocess audio to WAV, 16kHz, mono
def preprocess_audio(file_path, output_folder):
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_path = os.path.join(output_folder, f"{base_name}_processed.wav")
    command = f"ffmpeg -i '{file_path}' -ar 16000 -ac 1 '{output_path}'"
    os.system(command)
    return output_path

# Split audio by silence and fallback to fixed-length chunks
def split_audio(file_path, silence_thresh=-35, min_silence_len=800, chunk_duration_ms=30000):
    print(f"Splitting {file_path} into chunks by silence...")
    audio = AudioSegment.from_file(file_path)
    chunks = split_on_silence(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)

    if len(chunks) < 2:  # If silence split produces too few chunks
        print("Silence-based splitting produced few chunks. Falling back to fixed-length chunks.")
        chunks = [audio[i:i + chunk_duration_ms] for i in range(0, len(audio), chunk_duration_ms)]

    chunk_paths = []
    for idx, chunk in enumerate(chunks):
        if len(chunk) < 3000:  # Skip chunks shorter than 3 seconds
            continue
        chunk_path = os.path.join(output_folder, f"{os.path.basename(file_path)}_chunk_{idx}.wav")
        chunk.export(chunk_path, format="wav")
        chunk_paths.append(chunk_path)
    return chunk_paths

def generate_natural_prompts(transcriptions):
    prompts = []
    for transcription in transcriptions:
        key_phrase = transcription.split(".")[0]
        if "why" in transcription.lower():
            prompts.append(f"Why is {key_phrase.lower()} important?")
        elif "how" in transcription.lower():
            prompts.append(f"How does {key_phrase.lower()} happen?")
        elif "what" in transcription.lower():
            prompts.append(f"What do you mean by {key_phrase.lower()}?")
        else:
            prompts.append(f"Can you explain {key_phrase.lower()} further?")
    return prompts

print(f"Detected files in {audio_folder}: {os.listdir(audio_folder)}")

for audio_file in os.listdir(audio_folder):
    if audio_file.lower().endswith((".wav", ".mp3")):
        file_path = os.path.join(audio_folder, audio_file)
        print(f"Processing {file_path}...")

        try:
            processed_path = preprocess_audio(file_path, preprocessed_folder)

            segments = split_audio(processed_path)

            transcriptions = []
            for segment_path in segments:
                print(f"Transcribing segment: {segment_path}...")
                try:
                    result = model.transcribe(segment_path, language="en", task="transcribe")
                    transcription = result["text"]
                    transcriptions.append(transcription)
                    # Save transcription
                    output_file = os.path.join(output_folder, f"{os.path.basename(segment_path)}.txt")
                    with open(output_file, "w") as f:
                        f.write(transcription)
                    print(f"Transcription saved to {output_file}")
                except Exception as e:
                    print(f"Error transcribing segment {segment_path}: {e}")

            # Generate prompts
            if transcriptions:
                prompts = generate_natural_prompts(transcriptions)
                prompt_file = os.path.join(output_folder, f"{os.path.basename(file_path)}_prompts.jsonl")
                with open(prompt_file, "w") as f:
                    for prompt, response in zip(prompts, transcriptions):
                        f.write(f'{{"prompt": "{prompt}", "response": "{response}"}}\n')
                print(f"Prompts saved to {prompt_file}")

        except Exception as e:
            print(f"Error processing")
    else:
       

        print("Proce1")
