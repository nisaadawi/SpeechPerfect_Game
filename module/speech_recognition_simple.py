import os

# Set the path to ffmpeg for Whisper
ffmpeg_bin_dir = r"C:\Users\user\OneDrive\Documents\FYP ND\RenPy_Game\SpeechPerfect\ffmpeg\bin"
os.environ["IMAGEIO_FFMPEG_EXE"] = os.path.join(ffmpeg_bin_dir, "ffmpeg.exe")
os.environ["PATH"] = ffmpeg_bin_dir + os.pathsep + os.environ.get("PATH", "")

import whisper

# Make sure this audio file exists
audio_file = r"C:\Users\user\OneDrive\Documents\FYP ND\RenPy_Game\SpeechPerfect\module\test.wav"

model = whisper.load_model("base")
result = model.transcribe(audio_file, language="en", fp16=False)

print("📝 Transcript:")
print(result["text"])
