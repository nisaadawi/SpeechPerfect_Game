import os
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
# 2️⃣ Select and load WAV file
# ===============================
file_path = input("Enter path to WAV file: ").strip('"')

if not os.path.exists(file_path):
    print("❌ File not found.")
    exit()

print(f"\n🎧 Analyzing file: {file_path}\n")

# ===============================
# 3️⃣ Transcribe using Whisper
# ===============================
print("🧠 Transcribing audio using Whisper...")
model = whisper.load_model("base")
result = model.transcribe(file_path, language="en", fp16=False)
transcript = result["text"]

print("\n📝 Transcript:")
print(transcript)

# ===============================
# 4️⃣ Filler Words Detection
# ===============================
FILLER_WORDS = {
    "um", "uh", "erm", "hmm", "ah", "eh", "oh",
    "like", "you know", "i mean", "well", "so", "okay",
    "right", "basically", "actually", "literally",
    "sort of", "kind of", "maybe", "probably", "i guess", "perhaps"
}

words = transcript.lower().split()
detected_fillers = [w for w in words if w in FILLER_WORDS]
filler_count = len(detected_fillers)

print("\n💬 Detected Filler Words:")
if filler_count == 0:
    print("   • Excellent! No fillers detected 👏")
else:
    print(f"   • Total fillers: {filler_count}")
    print("   • List:", ", ".join(detected_fillers))

# Calculate filler frequency per minute
y, sr = librosa.load(file_path, sr=None)
duration_sec = librosa.get_duration(y=y, sr=sr)
duration_min = duration_sec / 60
fillers_per_min = filler_count / duration_min if duration_min > 0 else 0

# ===============================
# 5️⃣ Speech Rate (WPM)
# ===============================
y, sr = librosa.load(file_path, sr=None)
duration_sec = librosa.get_duration(y=y, sr=sr)
duration_min = duration_sec / 60
wpm = len(words) / duration_min if duration_min > 0 else 0

# Classify WPM
if 80 <= wpm <= 120:
    wpm_label = "🟩 Optimal"
elif (60 <= wpm < 80) or (120 < wpm <= 140):
    wpm_label = "🟨 Okay"
elif (40 <= wpm < 60) or (140 < wpm <= 160):
    wpm_label = "🟧 Moderate"
else:
    wpm_label = "🟥 Severe"

# ===============================
# 6️⃣ Filler Rate per Minute
# ===============================
fillers_per_min = filler_count / duration_min if duration_min > 0 else 0

if fillers_per_min <= 2:
    filler_label = "🟩 Excellent"
elif 3 <= fillers_per_min <= 5:
    filler_label = "🟨 Medium"
else:
    filler_label = "🟥 Needs Improvement"

# ===============================
# 7️⃣ Voice Timbre (MFCC & Spectral)
# ===============================
centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
rms = np.mean(librosa.feature.rms(y=y))

mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
mfcc_mean = np.mean(mfccs, axis=1)
mfcc_var = np.var(mfccs, axis=1)

mean_label = "Normal"
if np.mean(mfcc_mean) < -100:
    mean_label = "Dark / Dull"
elif np.mean(mfcc_mean) > 50:
    mean_label = "Bright / Clear"

var_label = "Moderate"
if np.mean(mfcc_var) < 50:
    var_label = "Monotone"
elif np.mean(mfcc_var) > 300:
    var_label = "Expressive"

# ===============================
# 8️⃣ Pause Duration Detection (Adaptive)
# ===============================

rms_energy = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
frames = np.arange(len(rms_energy))
times = librosa.frames_to_time(frames, sr=sr, hop_length=512)

# Adaptive threshold (sensitive to recording volume)
mean_rms = np.mean(rms_energy)
sensitivity = 0.5  # adjust 0.3–0.7 if needed
threshold = max(mean_rms * sensitivity, 0.005)

silent_frames = rms_energy < threshold

pause_durations = []
pause_start = None

for i, is_silent in enumerate(silent_frames):
    if is_silent and pause_start is None:
        pause_start = times[i]
    elif not is_silent and pause_start is not None:
        pause_end = times[i]
        duration = pause_end - pause_start
        if duration > 0.2:  # ignore micro pauses
            pause_durations.append(duration)
        pause_start = None

if pause_start is not None:
    duration = times[-1] - pause_start
    if duration > 0.2:
        pause_durations.append(duration)

avg_pause = np.mean(pause_durations) if pause_durations else 0
total_pause_time = np.sum(pause_durations)
pause_count = len(pause_durations)

# Classify average pause
if avg_pause <= 1:
    pause_label = "🟩 Optimal"
elif 1 < avg_pause <= 2:
    pause_label = "🟨 Moderate"
else:
    pause_label = "🟥 Awkward"

# ===============================
# 9️⃣ Display Summary
# ===============================
print("\n🧮 Speech Analysis Summary:")
print(f"   • Total duration: {duration_sec:.2f} sec")
print(f"   • Total words: {len(words)}")
print(f"   • Filler words: {filler_count}")
print(f"   • Filler rate: {(filler_count/len(words))*100:.1f}%")
print(f"   • Filler frequency: {fillers_per_min:.2f} fillers/min → {filler_label}")
print(f"   • Speech rate: {wpm:.1f} words/min → {wpm_label}")

print("\n⏸️ Pause Analysis:")
print(f"   • Total pauses detected: {pause_count}")
print(f"   • Average pause duration: {avg_pause:.2f} sec → {pause_label}")
print(f"   • Total silence time: {total_pause_time:.2f} sec")
print(f"   • Adaptive threshold used: {threshold:.4f}")

print("\n🎵 Voice Timbre Metrics:")
print(f"   • Spectral Centroid (brightness): {centroid:.2f} Hz")
print(f"   • Spectral Bandwidth (spread): {bandwidth:.2f} Hz")
print(f"   • RMS (loudness): {rms:.4f}")

print("\n🎶 MFCC-Based Voice Analysis:")
print(f"   • Average Timbre: {mean_label} (mean of MFCCs: {np.mean(mfcc_mean):.2f})")
print(f"   • Voice Expressiveness: {var_label} (mean of MFCC variance: {np.mean(mfcc_var):.2f})")


# ===============================
# 🔟 Visualizations
# ===============================

# --- (A) RMS Energy with Pauses ---
plt.figure(figsize=(12, 4))
plt.plot(times, rms_energy, label="RMS Energy", color="blue")
plt.axhline(y=threshold, color="red", linestyle="--", label=f"Silence Threshold = {threshold:.4f}")

for i, is_silent in enumerate(silent_frames):
    if is_silent:
        plt.axvspan(times[i], times[i] + (times[1] - times[0]), color='lightgrey', alpha=0.4)

plt.title("RMS Energy & Detected Pauses")
plt.xlabel("Time (s)")
plt.ylabel("Energy")
plt.legend()
plt.tight_layout()
plt.show()

# --- (B) MFCC Heatmap ---
plt.figure(figsize=(12, 4))
librosa.display.specshow(mfccs, x_axis='time', sr=sr, cmap='coolwarm')
plt.colorbar(format='%+2.0f dB')
plt.title('MFCC Heatmap')
plt.ylabel('MFCC Coefficients')
plt.xlabel('Time (s)')
plt.tight_layout()
plt.show()

# --- (C) MFCC Mean & Variance ---
coeffs = np.arange(1, mfccs.shape[0] + 1)
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
