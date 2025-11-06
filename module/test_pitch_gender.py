import librosa
import numpy as np

# === CONFIGURATION ===
filename = r"C:\Users\user\OneDrive\Documents\FYP ND\RenPy_Game\SpeechPerfect\module\audio\recorded.wav"
gender = "female"  # or "male"

# === LOAD AUDIO ===
y, sr = librosa.load(filename)

# === EXTRACT PITCH (F0) USING PYIN ===
f0, voiced_flag, voiced_probs = librosa.pyin(
    y, 
    fmin=librosa.note_to_hz('C2'),
    fmax=librosa.note_to_hz('C7')
)

# Remove NaN (unvoiced frames)
valid = ~np.isnan(f0)
f0 = f0[valid]

# Fix time vector to match pitch vector
frame_times = librosa.times_like(f0, sr=sr)

# === PITCH STATISTICS ===
mean_pitch = np.mean(f0)
std_pitch = np.std(f0)
pitch_range = np.max(f0) - np.min(f0)

# === SLOPE CALCULATION ===
pitch_diff = np.diff(f0)
time_diff = np.diff(frame_times)
slope = pitch_diff / time_diff
avg_slope = np.mean(np.abs(slope))
slope_std = np.std(slope)

# === PEAK ANALYSIS ===
from scipy.signal import find_peaks
peaks, _ = find_peaks(f0, prominence=20)
valleys, _ = find_peaks(-f0, prominence=20)
num_peaks_valleys = len(peaks) + len(valleys)
duration = frame_times[-1] if len(frame_times) > 0 else 0
peaks_per_sec = num_peaks_valleys / duration if duration > 0 else 0

# === GENDER-BASED BASELINES ===
if gender.lower() == "male":
    normal_pitch_range = (85, 180)  # Hz
else:
    normal_pitch_range = (165, 255)  # Hz

# === INTERPRETATION ===
if std_pitch < 30:
    expressiveness = "Monotonous"
elif std_pitch < 80:
    expressiveness = "Moderate"
else:
    expressiveness = "Expressive"

if avg_slope > 800 or peaks_per_sec > 4:
    behavior = "Likely nervous / jittery speech"
else:
    behavior = "Good / expressive intonation"

print(f"Gender: {gender.capitalize()}")
print(f"Mean Pitch: {mean_pitch:.2f} Hz (Expected {normal_pitch_range[0]}-{normal_pitch_range[1]} Hz)")
print(f"Pitch Variation (SD): {std_pitch:.2f} Hz → {expressiveness}")
print(f"Pitch Range: {pitch_range:.2f} Hz")
print(f"Average slope: {avg_slope:.2f} Hz/sec")
print(f"Slope SD: {slope_std:.2f} Hz/sec")
print(f"Number of significant peaks/valleys: {num_peaks_valleys}")
print(f"Peaks per second: {peaks_per_sec:.2f}")
print(f"Overall assessment: {behavior}")
