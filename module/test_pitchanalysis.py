import parselmouth
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# ===== Load audio file =====
audio_file = r"C:\Users\user\OneDrive\Documents\FYP ND\RenPy_Game\SpeechPerfect\module\audio\recorded.wav"
sound = parselmouth.Sound(audio_file)

# ===== Extract pitch using RAPT (default method) =====
pitch = sound.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=400)  # 10ms step
f0_values = pitch.selected_array['frequency']  # Hz
times = pitch.xs()  # time points

# ===== Remove unvoiced frames (0 Hz) =====
f0_nonzero = f0_values[f0_values > 0]

# ===== 1️⃣ Pitch Variation (Standard Deviation) =====
pitch_std = np.std(f0_nonzero)

# ===== 2️⃣ Pitch Slope / Contour =====
slopes = np.diff(f0_nonzero) / 0.01  # delta F0 / delta time (Hz/sec)
avg_slope = np.mean(np.abs(slopes))

# ===== 3️⃣ Peaks & Valleys =====
peaks, _ = find_peaks(f0_nonzero)
valleys, _ = find_peaks(-f0_nonzero)
num_peaks_valleys = len(peaks) + len(valleys)

# ===== Simple Intonation Assessment =====
intonation_score = 0

# Pitch variation
if pitch_std < 25:
    variation_label = "Monotone"
elif pitch_std < 35:
    variation_label = "Moderate"
else:
    variation_label = "Expressive"
    intonation_score += 1

# Slope
if avg_slope < 5:
    slope_label = "Flat"
elif avg_slope < 20:
    slope_label = "Moderate"
else:
    slope_label = "Dynamic"
    intonation_score += 1

# Peaks/Valleys
if num_peaks_valleys < 10:
    peak_label = "Few"
elif num_peaks_valleys < 50:
    peak_label = "Moderate"
else:
    peak_label = "Many"
    intonation_score += 1

# Overall assessment
if intonation_score == 0:
    overall = "Monotone / Flat"
elif intonation_score == 1:
    overall = "Moderate Intonation"
else:
    overall = "Good / Expressive Intonation"

# ===== Print results =====
print(f"Pitch Variation (SD): {pitch_std:.2f} Hz → {variation_label}")
print(f"Average Slope: {avg_slope:.2f} Hz/sec → {slope_label}")
print(f"Number of Peaks/Valleys: {num_peaks_valleys} → {peak_label}")
print(f"Overall Intonation Assessment: {overall}")

# ===== Plot Pitch Contour =====
plt.figure(figsize=(12, 4))
plt.plot(times, f0_values, label="Pitch Contour")
plt.xlabel("Time (s)")
plt.ylabel("Pitch (Hz)")
plt.title("Pitch Contour (F0)")
plt.grid(True)
plt.show()
