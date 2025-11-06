import os
import time
import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# 1️⃣ Setup FFmpeg for Whisper
# ===============================
ffmpeg_bin_dir = r"C:\Users\user\OneDrive\Documents\FYP ND\RenPy_Game\SpeechPerfect\ffmpeg\bin"
os.environ["IMAGEIO_FFMPEG_EXE"] = os.path.join(ffmpeg_bin_dir, "ffmpeg.exe")
os.environ["PATH"] = ffmpeg_bin_dir + os.pathsep + os.environ.get("PATH", "")

# ===============================
# 2️⃣ Record audio
# ===============================
fs = 16000  # Sample rate (Hz)
seconds = 10  # Recording duration
timestamp = int(time.time())
output_file = rf"C:\Users\user\OneDrive\Documents\FYP ND\RenPy_Game\SpeechPerfect\module\recorded_{timestamp}.wav"

print("🎙️ Recording... Start speaking!")
recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='int16')
sd.wait()
write(output_file, fs, recording)
print(f"✅ Recording saved as {output_file}\n")

# ===============================
# 3️⃣ Transcribe audio with Whisper
# ===============================
print("🧠 Transcribing audio using Whisper...")
model = whisper.load_model("base")
result = model.transcribe(output_file, language="en", fp16=False)
transcript = result["text"]

print("\n📝 Transcript:")
print(transcript)

# ===============================
# 4️⃣ Count filler words
# ===============================
FILLER_WORDS = {"um", "uh", "erm", "hmm", "like", "you know", "so", "well", "basically", "actually"}
words = transcript.lower().split()
filler_count = sum(1 for w in words if w in FILLER_WORDS)

# ===============================
# 5️⃣ Voice timbre analysis
# ===============================
y, sr = librosa.load(output_file, sr=None)

# Spectral centroid, bandwidth, RMS
centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
rms = np.mean(librosa.feature.rms(y=y))

# MFCCs
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# Compute MFCC mean and variance for each coefficient
mfcc_mean = np.mean(mfccs, axis=1)
mfcc_var = np.var(mfccs, axis=1)

# Convert MFCC mean & variance to labels (example)
mean_label = "Normal"
if np.mean(mfcc_mean) < -100: mean_label = "Dark / Dull"
elif np.mean(mfcc_mean) > 50: mean_label = "Bright / Clear"

var_label = "Moderate"
if np.mean(mfcc_var) < 50: var_label = "Monotone"
elif np.mean(mfcc_var) > 200: var_label = "Expressive"

# ===============================
# 6️⃣ Display summary
# ===============================
print("\n🧮 Speech Analysis Summary:")
print(f"   • Total words: {len(words)}")
print(f"   • Filler words: {filler_count}")
if len(words) > 0:
    print(f"   • Filler rate: {(filler_count/len(words))*100:.1f}%")

    # 🕒 Words per minute (WPM)
    duration_minutes = librosa.get_duration(y=y, sr=sr) / 60  # more accurate duration
    wpm = len(words) / duration_minutes

    # 🎯 Speech rate classification
    if wpm < 40 or wpm > 160:
        rate_label = "🟥 Severe (Too slow / Too fast)"
    elif 40 <= wpm < 60 or 140 < wpm <= 160:
        rate_label = "🟧 Moderate"
    elif 60 <= wpm < 80 or 120 < wpm <= 140:
        rate_label = "🟨 Okay"
    else:
        rate_label = "🟩 Optimal"

    print(f"   • Speaking rate: {wpm:.1f} words per minute → {rate_label}")

print("\n🎵 Voice Timbre Metrics:")
print(f"   • Spectral Centroid (brightness): {centroid:.2f} Hz")
print(f"   • Spectral Bandwidth (spread): {bandwidth:.2f} Hz")
print(f"   • RMS (loudness): {rms:.4f}")

print("\n🎶 MFCC-Based Voice Analysis:")
print(f"   • Average Timbre: {mean_label} (mean of MFCCs: {np.mean(mfcc_mean):.2f})")
print(f"   • Voice Expressiveness: {var_label} (mean of MFCC variance: {np.mean(mfcc_var):.2f})")

# ===============================
# 7️⃣ Plot MFCC Heatmap
# ===============================
plt.figure(figsize=(12, 4))
librosa.display.specshow(mfccs, x_axis='time', sr=sr, cmap='coolwarm')
plt.colorbar(format='%+2.0f dB')
plt.title('MFCC Heatmap')
plt.ylabel('MFCC Coefficients')
plt.xlabel('Time (s)')
plt.tight_layout()
plt.show()

# ===============================
# 8️⃣ Plot MFCC Mean & Variance
# ===============================
coeffs = np.arange(1, mfccs.shape[0]+1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.bar(coeffs, mfcc_mean, color='skyblue')
plt.xlabel('MFCC Coefficient')
plt.ylabel('Mean Value')
plt.title('MFCC Coefficient Mean')

plt.subplot(1, 2, 2)
plt.bar(coeffs, mfcc_var, color='salmon')
plt.xlabel('MFCC Coefficient')
plt.ylabel('Variance')
plt.title('MFCC Coefficient Variance (Expressiveness)')

plt.tight_layout()
plt.show()

input("\nPress Enter to exit...")
