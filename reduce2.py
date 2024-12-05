import os
from pydub import AudioSegment
os.environ["PATH"] += os.pathsep + os.path.expanduser("~/ffmpeg")

def reduce_audio_quality(input_file, output_file, max_size_mb=300):
 

    audio = AudioSegment.from_file(input_file)

    temp_output = "temp_output.mp3"
    audio.export(temp_output, format="mp3", bitrate="128k")

    temp_size = os.path.getsize(temp_output) / (1024 * 1024)  # Size in MB

    if temp_size <= max_size_mb:
        os.rename(temp_output, output_file)
        print(f"File '{input_file}' is under {max_size_mb} MB, saved as {output_file}")
        return

    print(f"File '{input_file}' exceeds {max_size_mb} MB, splitting into smaller files.")

    num_parts = min(3, len(audio) // (max_size_mb * 1024 * 1024 // audio.frame_rate // 1000))  # No more than 3 parts
    split_duration_ms = len(audio) // num_parts

    for i in range(num_parts):
        start_time = i * split_duration_ms
        end_time = (i + 1) * split_duration_ms if i < num_parts - 1 else len(audio)
        part = audio[start_time:end_time]
        part_output_file = f"{output_file}_part{i+1}.mp3"
        part.export(part_output_file, format="mp3", bitrate="128k")
        print(f"Exported part {i+1} of '{input_file}' to {part_output_file}")

    os.remove(temp_output)

def process_folder(input_folder, max_size_mb=300):

    for filename in os.listdir(input_folder):
        input_file = os.path.join(input_folder, filename)

  
        if os.path.isfile(input_file) and filename.lower().endswith(('.mp3', '.wav', '.flac', '.ogg')):
            print(f"Processing file: {filename}")
          
            output_file = os.path.join(input_folder, f"processed_{filename}")
            reduce_audio_quality(input_file, output_file, max_size_mb)

input_folder = "/Users/avga2021/Desktop/NLP_Final2/speech_to_text/Audio copy"  # Replace with the path to your folder
process_folder(input_folder)
