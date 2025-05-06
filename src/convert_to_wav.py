#reference code: https://gist.github.com/arjunsharma97/0ecac61da2937ec52baf61af1aa1b759#file-m4atowav-py
# Original Author: arjunsharma97
# This code is adapted from the reference above, for non-commercial purposes

import os
import shutil
from pydub import AudioSegment
from pydub.utils import which
import stat

# Path
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
os.chdir(project_root)

# Relative paths (edit if needed)
FFMPEG_BIN = os.path.join("ffmpeg", "bin")  # put your ffmpeg binaries here
SOURCE_DIR = os.path.join("data", "m4a_input")
TARGET_DIR = os.path.join("data", "wav_output")


# Add ffmpeg directory to the environment variable PATH
if FFMPEG_BIN not in os.environ["PATH"]:
    os.environ["PATH"] += os.pathsep + FFMPEG_BIN

print("ffmpeg  =", which("ffmpeg"))
print("ffprobe =", which("ffprobe"))

# Check if the input directory exsits
if not os.path.exists(SOURCE_DIR):
    print(f"Input data doesn't exist:{SOURCE_DIR}")
    exit(1)

# Create output directory
os.makedirs(TARGET_DIR, exist_ok=True)

# Start to convert
converted = 0
failed = 0

for filename in os.listdir(SOURCE_DIR):
    if filename.lower().endswith(".m4a"):
        input_path = os.path.join(SOURCE_DIR, filename)
        output_filename = os.path.splitext(filename)[0] + ".wav"
        output_path = os.path.join(TARGET_DIR, output_filename)

        try:
            # Check if readable
            if not os.access(input_path, os.R_OK):
                raise PermissionError("The file is unreadable, might be occupied by other programs")

            print(f"Converting: {filename}")
            audio = AudioSegment.from_file(input_path, format="m4a")
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(output_path, format="wav")
            converted += 1
        except PermissionError as pe:
            print(f"Permission error (skip): {filename} → {pe}")
            failed += 1
        except Exception as e:
            print(f"Other error (skip): {filename} → {e}")
            failed += 1

# Outcome
print("\nBatch conversion completed")
print(f"Convert success: {converted}")
print(f"Convert failed: {failed}")
print(f"WAV file is saved at: {TARGET_DIR}")

