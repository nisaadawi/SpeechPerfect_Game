import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import librosa
import numpy as np

# -----------------------------
# Load Wav2Vec2 processor + model (A/V/D)
# -----------------------------
MODEL_NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME)

# -----------------------------
# Load audio
# -----------------------------
audio_path = input("Enter path to WAV file: ")
y, sr = librosa.load(audio_path, sr=16000)  # model expects 16kHz
input_values = processor(y, sampling_rate=sr, return_tensors="pt").input_values

# -----------------------------
# Predict A/V/D
# -----------------------------
with torch.no_grad():
    logits = model(input_values).logits

avd_raw = logits.squeeze().cpu().numpy()

# Handle different output shapes
if avd_raw.ndim > 1:
    avd_raw = avd_raw.flatten()

# Ensure we have 3 values (arousal, valence, dominance)
if len(avd_raw) >= 3:
    avd_raw = avd_raw[:3]
else:
    raise ValueError(f"Expected 3 AVD values, got {len(avd_raw)}")

print(f"\n📊 Raw AVD Logits: {avd_raw}")
print(f"   Min: {avd_raw.min():.3f}, Max: {avd_raw.max():.3f}, Mean: {avd_raw.mean():.3f}")

# Proper normalization: The model outputs logits that need to be scaled
# MSP-Podcast dataset typically uses ranges around -1 to 1 or 0 to 1
# Try multiple normalization methods and choose the best one

# Method 1: Sigmoid (for unbounded logits)
avd_tensor = torch.tensor(avd_raw).unsqueeze(0)
avd_sigmoid = torch.sigmoid(avd_tensor).squeeze().numpy()

# Method 2: Tanh normalization (for values in roughly -3 to 3 range)
avd_tanh = (np.tanh(avd_raw) + 1) / 2

# Method 3: Linear scaling if values are already in a reasonable range
# Check if values are already roughly in 0-1 or -1 to 1 range
if avd_raw.min() >= -1.5 and avd_raw.max() <= 1.5:
    # Values seem to be in -1 to 1 range, scale to 0-1
    avd_linear = (avd_raw + 1) / 2
    norm_method = "linear_scale"
elif avd_raw.min() >= 0 and avd_raw.max() <= 1:
    # Already in 0-1 range
    avd_linear = avd_raw
    norm_method = "no_scale"
else:
    # Use sigmoid or tanh
    avd_linear = avd_sigmoid
    norm_method = "sigmoid_fallback"

# Choose the best normalization method
# Prefer method that gives values in middle range (not all extremes)
sigmoid_range = avd_sigmoid.max() - avd_sigmoid.min()
tanh_range = avd_tanh.max() - avd_tanh.min()

# Use the method with better range (not all 0s or 1s)
if 0.1 < sigmoid_range < 0.9 and not (np.any(avd_sigmoid < 0.01) or np.any(avd_sigmoid > 0.99)):
    arousal, valence, dominance = avd_sigmoid
    norm_method = "sigmoid"
elif 0.1 < tanh_range < 0.9:
    arousal, valence, dominance = avd_tanh
    norm_method = "tanh"
else:
    # Use linear scaling or direct values
    arousal, valence, dominance = avd_linear
    if norm_method == "no_scale":
        norm_method = "direct"

print(f"   Normalization method: {norm_method}")
print(f"\n📊 AVD Emotion Dimensions (normalized to 0-1):")
print(f"Arousal: {arousal:.3f} (Energy/Activation: 0=calm, 1=highly activated)")
print(f"Valence: {valence:.3f} (Positivity: 0=negative, 1=positive)")
print(f"Dominance: {dominance:.3f} (Control: 0=submissive, 1=dominant)")

# Calculate speech rate and rushed speech indicators
duration_sec = len(y) / sr

# Use actual audio features to detect rushed speech
# High energy variation, fast tempo, short pauses indicate rushed speech
# Calculate RMS energy to detect speech rate and pauses
rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
energy_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)

# Detect pauses (low energy regions)
mean_rms = np.mean(rms)
threshold = mean_rms * 0.3
silent_frames = rms < threshold

# Calculate pause ratio (lower = more rushed)
pause_ratio = np.sum(silent_frames) / len(silent_frames) if len(silent_frames) > 0 else 0

# Calculate energy variation (high variation = rushed, inconsistent pace)
energy_std = np.std(rms)
energy_cv = energy_std / (mean_rms + 1e-6)  # Coefficient of variation

# Calculate tempo/rhythm (using onset detection)
onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
if len(onsets) > 1:
    onset_intervals = np.diff(onsets)
    avg_onset_interval = np.mean(onset_intervals)
    onset_rate = 1.0 / (avg_onset_interval + 1e-6)  # Onsets per second
    # Fast onset rate = rushed speech
    tempo_score = min(1.0, onset_rate / 5.0)  # Normalize (5 onsets/sec = very fast)
else:
    tempo_score = 0.5  # Default if can't detect onsets

# Distinguish between fast-paced but controlled (TED talks) vs rushed/stressed speech
# Key insight: Fast pace + High dominance + Positive valence = Confident, energetic (NOT rushed)
#              Fast pace + Low dominance + Negative valence = Anxious, rushed (STRESSED)

# Calculate base rushed indicators (fast pace features)
base_rushed = (
    arousal * 0.2 +                    # High arousal contributes
    (1 - pause_ratio) * 0.3 +          # Few pauses = rushed
    tempo_score * 0.3 +                 # Fast tempo = rushed
    min(1.0, energy_cv * 2) * 0.2      # High variation = rushed
)

# Control factor: High dominance + positive valence = controlled, confident speech
# This REDUCES the rushed score (fast but controlled = good, not rushed)
control_factor = (dominance * 0.6) + (valence * 0.4)  # How controlled/confident

# Adjust rushed score: If speech is controlled (high dominance + positive valence),
# reduce the rushed score even if pace is fast
# TED talks: Fast pace but controlled = low rushed score
# Anxious speech: Fast pace but uncontrolled = high rushed score
rushed_score = base_rushed * (1 - control_factor * 0.5)  # Reduce by up to 50% if controlled
rushed_score = min(100, max(0, rushed_score * 100))

# Calculate speech quality score (for good speeches like TED talks)
# High arousal + High dominance + Positive valence = Excellent speech
speech_quality = (arousal * 0.3) + (dominance * 0.4) + (valence * 0.3)
speech_quality_score = speech_quality * 100

print(f"\n⚡ Speech Rate & Quality Analysis:")
print(f"   Duration: {duration_sec:.2f} seconds")
print(f"   Pause Ratio: {pause_ratio:.2%} (Normal: 15-25%)")
print(f"   Energy Variation (CV): {energy_cv:.3f}")
if len(onsets) > 1:
    print(f"   Onset Rate: {onset_rate:.2f} onsets/sec")
print(f"   Speech Quality Score: {speech_quality_score:.2f}/100")
print(f"   Control Factor: {control_factor:.3f} (Higher = more controlled/confident)")
print(f"   Rushed Speech Score: {rushed_score:.2f}/100")

# Interpretation based on AVD + pace
if control_factor > 0.6 and arousal > 0.5:
    if rushed_score < 40:
        print(f"   ✅ CONFIDENT & ENERGETIC - Fast-paced but well-controlled speech")
        print(f"      (Typical of professional presentations, TED talks)")
    else:
        print(f"   ⚠️  Fast-paced speech with some rushed elements")
elif rushed_score > 60:
    print(f"   ⚠️  RUSHED - Fast-paced, hurried speech pattern detected")
    print(f"      (Indicates stress, anxiety, or time pressure)")
elif rushed_score > 40:
    print(f"   ⚠️  Moderately rushed - Some hurried elements detected")
elif pause_ratio < 0.1:
    print(f"   ⚠️  Very few pauses detected - indicates rushed speech")
elif pause_ratio > 0.25 and control_factor > 0.5:
    print(f"   ✅ Well-paced and controlled speech")

# -----------------------------
# Calculate Stress/Anxiety from AVD
# -----------------------------
# Stress/Anxiety indicators:
# - High Arousal + Low Valence = Negative high activation (anxiety/stress)
# - Low Dominance = Feeling out of control (stress indicator)
# - High Arousal + Low Valence + Low Dominance = High stress/anxiety

# Method 1: Weighted combination (improved to distinguish controlled vs rushed)
# Key: High arousal + Low dominance + Low valence = stress/anxiety
#      High arousal + High dominance + High valence = confident, energetic (NOT stressed)
# Only count rushed score if speech is NOT well-controlled
anxiety_base = (arousal * 0.3) + ((1 - valence) * 0.35) + ((1 - dominance) * 0.35)

# Adjust based on control: If well-controlled, reduce anxiety even if fast-paced
# If uncontrolled and rushed, increase anxiety
if control_factor > 0.6:
    # Well-controlled speech (like TED talks) - reduce anxiety contribution from pace
    anxiety_score = anxiety_base * 0.7 + (rushed_score / 100 * 0.1)
else:
    # Uncontrolled speech - full contribution from rushed elements
    anxiety_score = anxiety_base * 0.7 + (rushed_score / 100 * 0.3)

anxiety_score = min(100, anxiety_score * 100)  # Scale to 0-100

# Method 2: Stress in 2D Arousal-Valence space
# Stress/Anxiety typically occurs in: High Arousal + Low Valence quadrant
# Distance from calm state (low arousal, high valence)
stress_2d = np.sqrt((arousal - 0.0)**2 + (valence - 1.0)**2) * 100  # Distance from ideal calm state
stress_2d = min(100, stress_2d)

# Method 3: Combined stress score (weighted average of both methods)
stress_score = (anxiety_score * 0.6) + (stress_2d * 0.4)

# Determine stress/anxiety levels
def get_stress_level(score):
    if score < 25:
        return "🟢 Low", "Calm and relaxed"
    elif score < 40:
        return "🟡 Low-Moderate", "Slightly stressed"
    elif score < 55:
        return "🟠 Moderate", "Moderately stressed/anxious"
    elif score < 70:
        return "🟠 High-Moderate", "Noticeably stressed/anxious"
    elif score < 85:
        return "🔴 High", "Highly stressed/anxious"
    else:
        return "🔴 Very High", "Extremely stressed/anxious"

anxiety_level, anxiety_desc = get_stress_level(anxiety_score)
stress_level, stress_desc = get_stress_level(stress_score)

print(f"\n😰 Stress/Anxiety Analysis:")
print(f"   Rushed Speech Score: {rushed_score:.2f}/100")
print(f"   Anxiety Score (AVD + Rushed): {anxiety_score:.2f}/100 → {anxiety_level} {anxiety_desc}")
print(f"   Stress Score (2D distance): {stress_2d:.2f}/100")
print(f"   Combined Stress Score: {stress_score:.2f}/100 → {stress_level} {stress_desc}")

# Interpretation guide
print(f"\n📖 Interpretation:")
if control_factor > 0.6 and arousal > 0.5 and valence > 0.5:
    print("   ✅ CONFIDENT & ENERGETIC SPEECH")
    print("      High arousal + High dominance + Positive valence")
    print("      This indicates well-controlled, engaging speech (like TED talks)")
    if rushed_score < 40:
        print("      Speech is fast-paced but controlled - excellent delivery!")
elif control_factor < 0.4 and arousal > 0.6:
    print("   ⚠️  UNCONTROLLED FAST SPEECH")
    print("      High arousal but low control - indicates rushed/stressed speech")
elif arousal > 0.6 and valence < 0.4:
    print("   ⚠️  High arousal with negative valence - indicates stress/anxiety")
elif arousal > 0.7 and control_factor > 0.5:
    print("   ✅ Very energetic and controlled - confident, engaging speech")
elif arousal < 0.3:
    print("   ℹ️  Low arousal - calm/relaxed speech")
if valence < 0.3:
    print("   ⚠️  Very negative valence - strong negative emotions detected")
elif valence > 0.6:
    print("   ✅ Positive valence - positive, engaging emotional state")
if dominance < 0.3:
    print("   ⚠️  Low dominance - feeling of lack of control (stress indicator)")
elif dominance > 0.6:
    print("   ✅ High dominance - feeling in control, confident delivery")
if anxiety_score < 30 and rushed_score < 40:
    print("   ✅ Overall: Calm, relaxed, and well-controlled speech pattern")
elif anxiety_score < 50 and control_factor > 0.6:
    print("   ✅ Overall: Confident, energetic speech (good quality)")
elif anxiety_score > 60 or (rushed_score > 60 and control_factor < 0.5):
    print("   ⚠️  Overall: Stressed, anxious, or rushed speech pattern detected")
else:
    print("   ℹ️  Overall: Moderate speech pattern with some areas for improvement")
