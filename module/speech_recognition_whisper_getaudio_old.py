import os
import sys
import csv
from typing import Any, Dict, Iterable, List, Optional

import whisper
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2Model

# ===============================
# 1️⃣ Setup FFmpeg for Whisper
# ===============================
ffmpeg_bin_dir = r"C:\Users\user\OneDrive\Documents\FYP ND\RenPy_Game\SpeechPerfect\ffmpeg\bin"
os.environ["IMAGEIO_FFMPEG_EXE"] = os.path.join(ffmpeg_bin_dir, "ffmpeg.exe")
os.environ["PATH"] = ffmpeg_bin_dir + os.pathsep + os.environ.get("PATH", "")

# ===============================
# Global configuration
# ===============================
FILLER_WORDS = {
    "um",
    "uh",
    "erm",
    "hmm",
    "ah",
    "eh",
    "oh",
    "like",
    "you know",
    "i mean",
    "well",
    "so",
    "okay",
    "right",
    "basically",
    "actually",
    "literally",
    "sort of",
    "kind of",
    "maybe",
    "probably",
    "i guess",
    "perhaps",
}

CSV_FIELDS = [
    "file_name",
    "file_path",
    "transcript",
    "duration_sec",
    "duration_min",
    "word_count",
    "speech_rate_wpm",
    "wpm_label",
    "filler_count",
    "detected_fillers",
    "filler_rate_percent",
    "fillers_per_min",
    "filler_label",
    "pause_count",
    "avg_pause_sec",
    "total_pause_time_sec",
    "pause_label",
    "pause_durations",
    "pause_threshold",
    "spectral_centroid_hz",
    "spectral_bandwidth_hz",
    "rms",
    "mfcc_mean_mean",
    "mfcc_variance_mean",
    "mfcc_mean_label",
    "mfcc_variance_label",
    "mfcc_std_mean_raw_before_cmvn",
    "mfcc_std_mean_after_cmvn",
    "mfcc_std_label",
]

# ===============================
# MFCC Variance Thresholds (old values)
# ===============================
MFCC_VARIANCE_MONOTONE_THRESHOLD = 50
MFCC_VARIANCE_EXPRESSIVE_THRESHOLD = 300

# ===============================
# MFCC Standard Deviation Thresholds (Raw values before CMVN)
# ===============================
# Based on data distribution and literature mapping:
# < 14.0 → Monotone (Bad)
# 14.0 - 21.0 → Moderate (Okay)
# > 21.0 → Expressive (Good)
RAW_STD_MONOTONE_THRESHOLD = 14.0      # Threshold between Monotone and Moderate
RAW_STD_MODERATE_THRESHOLD = 21.0      # Threshold between Moderate and Expressive
RAW_STD_EXPRESSIVE_THRESHOLD = 26.0    # High threshold for very expressive (optional)


# ===============================
# Global StudentNet Model (loaded lazily)
# ===============================
STUDENTNET_MODEL = None
STUDENTNET_W2V_MODEL = None
STUDENTNET_DEVICE = None
STUDENTNET_PATH = r"C:\Users\user\OneDrive\Documents\FYP ND\RenPy_Game\SpeechPerfect\voice-based-stress-recognition"


# ===============================
# Helper functions
# ===============================
def load_studentnet_model():
    """Load the StudentNet stress detection model."""
    global STUDENTNET_MODEL, STUDENTNET_W2V_MODEL, STUDENTNET_DEVICE
    
    if STUDENTNET_MODEL is None:
        print("🔄 Loading StudentNet stress detection model...")
        try:
            # Add StudentNet repo to Python path
            sys.path.append(STUDENTNET_PATH)
            
            from models import StudentNet
            
            # Setup device
            STUDENTNET_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Initialize StudentNet model
            STUDENTNET_MODEL = StudentNet().to(STUDENTNET_DEVICE)
            
            # Load pretrained weights
            weights_path = os.path.join(STUDENTNET_PATH, "pytorch_model.bin")
            state = torch.load(weights_path, map_location=STUDENTNET_DEVICE)
            STUDENTNET_MODEL.load_state_dict(state)
            STUDENTNET_MODEL.eval()
            
            # Load Wav2Vec2 feature extractor
            STUDENTNET_W2V_MODEL = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(STUDENTNET_DEVICE)
            STUDENTNET_W2V_MODEL.eval()
            
            print("✅ StudentNet model loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load StudentNet model: {e}")
            import traceback
            traceback.print_exc()
            STUDENTNET_MODEL = None
            STUDENTNET_W2V_MODEL = None
            STUDENTNET_DEVICE = None
    
    return STUDENTNET_MODEL, STUDENTNET_W2V_MODEL, STUDENTNET_DEVICE


def detect_stress_studentnet(file_path: str, y: np.ndarray, sr: int) -> Dict[str, Any]:
    """
    Detect stress using StudentNet pretrained model.
    
    Returns:
    - logit_0: Logit value for "Not Stressed" class
    - logit_1: Logit value for "Stressed" class
    - probability: Stress probability (0-1)
    - predicted_label: "Stressed" or "Not Stressed"
    """
    model, w2v_model, device = load_studentnet_model()
    
    if model is None or w2v_model is None or device is None:
        return {
            "studentnet_logit_0": None,
            "studentnet_logit_1": None,
            "studentnet_probability": None,
            "studentnet_predicted_label": "Model unavailable",
        }
    
    try:
        # Resample to 16kHz if needed (model requirement)
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sr = 16000
        
        # Extract embedding using Wav2Vec2
        waveform = torch.tensor(y).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = w2v_model(waveform, return_dict=True)
            hidden_states = outputs.last_hidden_state
            embedding = hidden_states.mean(dim=1)  # mean pooling over time
            embedding = embedding[:, :512]  # truncate to 512-dim
        
        # Predict stress using StudentNet
        with torch.no_grad():
            logits, _ = model(embedding)
            
            # Extract raw logit values
            logit_not_stress = logits[0, 0].item()
            logit_stress = logits[0, 1].item()
            
            # Convert logits to probabilities (softmax)
            probs = F.softmax(logits, dim=1)
            stress_prob = probs[0, 1].item()  # index 1 = stressed
        
        # Decide label (threshold = 0.5)
        threshold = 0.5
        stress_label = "Stressed" if stress_prob >= threshold else "Not Stressed"
        
        return {
            "studentnet_logit_0": round(float(logit_not_stress), 4),
            "studentnet_logit_1": round(float(logit_stress), 4),
            "studentnet_probability": round(float(stress_prob), 4),
            "studentnet_predicted_label": stress_label,
        }
        
    except Exception as e:
        print(f"⚠️ Error in StudentNet stress detection: {e}")
        import traceback
        traceback.print_exc()
        return {
            "studentnet_logit_0": None,
            "studentnet_logit_1": None,
            "studentnet_probability": None,
            "studentnet_predicted_label": f"Error: {str(e)}",
        }


def display_summary(
    file_path: str,
    transcript: str,
    detected_fillers: List[str],
    filler_label: str,
    wpm_label: str,
    pause_label: str,
    metrics: Dict[str, Optional[float]],
) -> None:
    print(f"\n🎧 Analyzing file: {file_path}\n")

    print("🧠 Transcript:")
    print(transcript or "(empty)")

    print("\n💬 Detected Filler Words:")
    if not detected_fillers:
        print("   • Excellent! No fillers detected 👏")
    else:
        print(f"   • Total fillers: {len(detected_fillers)}")
        print("   • List:", ", ".join(detected_fillers))

    print("\n🧮 Speech Analysis Summary:")
    print(f"   • Total duration: {metrics['duration_sec']:.2f} sec")
    print(f"   • Total words: {metrics['word_count']}")
    print(f"   • Filler words: {metrics['filler_count']}")
    print(f"   • Filler rate: {metrics['filler_rate_percent']:.1f}%")
    print(f"   • Filler frequency: {metrics['fillers_per_min']:.2f} fillers/min → {filler_label}")
    print(f"   • Speech rate: {metrics['speech_rate_wpm']:.1f} words/min → {wpm_label}")

    print("\n⏸️ Pause Analysis:")
    print(f"   • Total pauses detected: {metrics['pause_count']}")
    print(f"   • Average pause duration: {metrics['avg_pause_sec']:.2f} sec → {pause_label}")
    print(f"   • Total silence time: {metrics['total_pause_time_sec']:.2f} sec")
    print(f"   • Adaptive threshold used: {metrics['pause_threshold']:.4f}")

    print("\n🎵 Voice Timbre Metrics:")
    print(f"   • Spectral Centroid (brightness): {metrics['spectral_centroid_hz']:.2f} Hz")
    print(f"   • Spectral Bandwidth (spread): {metrics['spectral_bandwidth_hz']:.2f} Hz")
    print(f"   • RMS (loudness): {metrics['rms']:.4f}")

    print("\n🎶 MFCC-Based Voice Analysis:")
    print(f"   • Average Timbre: {metrics['mfcc_mean_label']} (mean of MFCCs: {metrics['mfcc_mean_mean']:.2f})")
    print(
        f"   • Voice Expressiveness (Variance): {metrics['mfcc_variance_label']} "
        f"(mean of MFCC variance: {metrics['mfcc_variance_mean']:.2f})"
    )
    print(
        f"   • Voice Expressiveness (Std Dev - Before CMVN): {metrics['mfcc_std_mean_raw_before_cmvn']:.2f} → {metrics['mfcc_std_label']}"
    )
    print(
        f"   • Voice Expressiveness (Std Dev - After CMVN): {metrics['mfcc_std_mean_after_cmvn']:.6f}"
    )

    print("\n🎯 Stress Detection (StudentNet Pretrained Model):")
    if metrics.get("studentnet_probability") is not None:
        logit_0 = metrics.get("studentnet_logit_0")
        logit_1 = metrics.get("studentnet_logit_1")
        probability = metrics.get("studentnet_probability")
        label = metrics.get("studentnet_predicted_label", "Unknown")
        
        print(f"   • Logit 0 (Not Stressed): {logit_0:.4f}")
        print(f"   • Logit 1 (Stressed): {logit_1:.4f}")
        print(f"   • Stress Probability: {probability*100:.2f}%")
        print(f"   • Predicted Label: {label}")
    else:
        print("   • StudentNet model unavailable or error occurred")


def expand_paths_to_audio_files(raw_input: str) -> List[str]:
    raw_input = raw_input.strip()
    if not raw_input:
        return []

    pieces = [p.strip().strip('"') for p in raw_input.split(",") if p.strip()]
    audio_paths: List[str] = []
    valid_ext = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}

    for piece in pieces:
        expanded = os.path.expanduser(piece)
        if os.path.isdir(expanded):
            for entry in sorted(os.listdir(expanded)):
                full_path = os.path.join(expanded, entry)
                if os.path.isfile(full_path) and os.path.splitext(full_path)[1].lower() in valid_ext:
                    audio_paths.append(full_path)
        elif os.path.isfile(expanded):
            if os.path.splitext(expanded)[1].lower() in valid_ext:
                audio_paths.append(expanded)
        else:
            matched = [
                path
                for path in sorted(glob_paths(expanded))
                if os.path.splitext(path)[1].lower() in valid_ext
            ]
            audio_paths.extend(matched)

    unique_paths = []
    seen = set()
    for path in audio_paths:
        if path not in seen:
            unique_paths.append(path)
            seen.add(path)
    return unique_paths


def glob_paths(pattern: str) -> Iterable[str]:
    try:
        import glob

        return glob.glob(pattern)
    except Exception:
        return []


def prompt_yes_no(message: str, default: bool = False) -> bool:
    default_label = "y" if default else "n"
    response = input(f"{message} [y/n] (default {default_label}): ").strip().lower()
    if not response:
        return default
    if response in {"y", "yes"}:
        return True
    if response in {"n", "no"}:
        return False
    print("Invalid choice, using default.")
    return default


def prompt_csv_path(default_path: str) -> str:
    response = input(f"Enter output CSV path [{default_path}]: ").strip().strip('"')
    if not response:
        return default_path
    expanded = os.path.expanduser(response)
    if os.path.isdir(expanded) or expanded.endswith(os.sep):
        expanded = os.path.join(expanded, os.path.basename(default_path))
    if not expanded.lower().endswith(".csv"):
        expanded = f"{expanded}.csv"
    directory = os.path.dirname(expanded)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    return expanded


def analyze_audio(file_path: str, model: Any, show_plots: bool) -> Dict[str, Optional[float]]:
    result = model.transcribe(file_path, language="en", fp16=False)
    transcript = result.get("text", "").strip()

    words = transcript.lower().split()
    detected_fillers_list = [w for w in words if w in FILLER_WORDS]
    filler_count = len(detected_fillers_list)

    y, sr = librosa.load(file_path, sr=None)
    duration_sec = float(librosa.get_duration(y=y, sr=sr))
    duration_min = duration_sec / 60 if duration_sec > 0 else 1.0
    word_count = len(words)

    wpm = word_count / duration_min if duration_min > 0 else 0.0
    if 80 <= wpm <= 120:
        wpm_label = "🟩 Optimal" #1
    elif (60 <= wpm < 80) or (120 < wpm <= 140):
        wpm_label = "🟨 Okay" #1
    elif (40 <= wpm < 60) or (140 < wpm <= 160):
        wpm_label = "🟧 Moderate" #0.5
    else:
        wpm_label = "🟥 Severe" #0

    fillers_per_min = filler_count / duration_min if duration_min > 0 else 0.0
    if fillers_per_min <= 2:
        filler_label = "🟩 Excellent"
    elif 3 <= fillers_per_min <= 5:
        filler_label = "🟨 Medium"
    else:
        filler_label = "🟥 Needs Improvement"

    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    rms = float(np.mean(librosa.feature.rms(y=y)))

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # Calculate raw std before CMVN
    mfcc_std_before_cmvn = np.std(mfccs, axis=1)  # Raw std values before CMVN
    mfcc_std_mean_raw_before_cmvn = float(np.mean(mfcc_std_before_cmvn))
    
    # Apply CMVN (Cepstral Mean and Variance Normalization)
    # Normalize across time frames (axis=1) for each coefficient
    # This removes channel/speaker bias and is standard in speech processing
    eps = 1e-8  # Small epsilon to avoid division by zero
    mfcc_mean_per_coeff = np.mean(mfccs, axis=1, keepdims=True)  # Shape: (13, 1)
    mfcc_std_per_coeff = np.std(mfccs, axis=1, keepdims=True)  # Shape: (13, 1)
    mfccs_cmvn = (mfccs - mfcc_mean_per_coeff) / (mfcc_std_per_coeff + eps)
    
    # Calculate statistics from CMVN-normalized MFCCs
    mfcc_mean = np.mean(mfccs_cmvn, axis=1)
    mfcc_var = np.var(mfccs_cmvn, axis=1)
    mfcc_std_after_cmvn = np.std(mfccs_cmvn, axis=1)  # Raw std values after CMVN
    
    # Use raw std value after CMVN (no scaling, no categorization)
    mfcc_std_mean_after_cmvn = float(np.mean(mfcc_std_after_cmvn))
    
    mfcc_mean_value = float(np.mean(mfcc_mean))
    mfcc_var_value = float(np.mean(mfcc_var))

    mean_label = "Normal"
    if mfcc_mean_value < -100:
        mean_label = "Dark / Dull"
    elif mfcc_mean_value > 50:
        mean_label = "Bright / Clear"

    # MFCC variance thresholds (old values)
    var_label = "Moderate"
    if mfcc_var_value < MFCC_VARIANCE_MONOTONE_THRESHOLD:
        var_label = "Monotone"
    elif mfcc_var_value > MFCC_VARIANCE_EXPRESSIVE_THRESHOLD:
        var_label = "Expressive"

    # MFCC standard deviation categorization based on raw values before CMVN
    # Using thresholds: < 14.0 = Monotone, 14.0-21.0 = Moderate, > 21.0 = Expressive
    if mfcc_std_mean_raw_before_cmvn < RAW_STD_MONOTONE_THRESHOLD:
        mfcc_std_label = "Monotone ❌ Bad"
    elif mfcc_std_mean_raw_before_cmvn < RAW_STD_MODERATE_THRESHOLD:
        mfcc_std_label = "Moderate ⚠️ Okay"
    else:  # >= RAW_STD_MODERATE_THRESHOLD (i.e., >= 21.0)
        mfcc_std_label = "Expressive ✅ Good"

    rms_energy = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    frames = np.arange(len(rms_energy))
    energy_times = librosa.frames_to_time(frames, sr=sr, hop_length=512)

    mean_rms = float(np.mean(rms_energy))
    threshold = max(mean_rms * 0.5, 0.005)

    silent_frames = rms_energy < threshold
    pause_durations: List[float] = []
    pause_start: Optional[float] = None

    for i, is_silent in enumerate(silent_frames):
        if is_silent and pause_start is None:
            pause_start = float(energy_times[i])
        elif not is_silent and pause_start is not None:
            pause_end = float(energy_times[i])
            duration = pause_end - pause_start
            if duration > 0.2:
                pause_durations.append(duration)
            pause_start = None

    if pause_start is not None:
        duration = float(energy_times[-1]) - pause_start
        if duration > 0.2:
            pause_durations.append(duration)

    avg_pause = float(np.mean(pause_durations)) if pause_durations else 0.0
    total_pause_time = float(np.sum(pause_durations)) if pause_durations else 0.0
    pause_count = len(pause_durations)

    if avg_pause <= 1:
        pause_label = "🟩 Optimal"
    elif 1 < avg_pause <= 2:
        pause_label = "🟨 Moderate"
    else:
        pause_label = "🟥 Awkward"

    # StudentNet-based Stress Detection using pretrained model
    studentnet_results = detect_stress_studentnet(file_path, y, sr)

    metrics: Dict[str, Optional[float]] = {
        "file_name": os.path.basename(file_path),
        "file_path": file_path,
        "transcript": transcript,
        "duration_sec": duration_sec,
        "duration_min": duration_min,
        "word_count": word_count,
        "speech_rate_wpm": wpm,
        "wpm_label": wpm_label,
        "filler_count": filler_count,
        "detected_fillers": ";".join(detected_fillers_list),
        "filler_rate_percent": (filler_count / word_count * 100) if word_count else 0.0,
        "fillers_per_min": fillers_per_min,
        "filler_label": filler_label,
        "pause_count": pause_count,
        "avg_pause_sec": avg_pause,
        "total_pause_time_sec": total_pause_time,
        "pause_label": pause_label,
        "pause_durations": ";".join(f"{d:.3f}" for d in pause_durations),
        "pause_threshold": threshold,
        "spectral_centroid_hz": centroid,
        "spectral_bandwidth_hz": bandwidth,
        "rms": rms,
        "mfcc_mean_mean": mfcc_mean_value,
        "mfcc_variance_mean": mfcc_var_value,
        "mfcc_mean_label": mean_label,
        "mfcc_variance_label": var_label,
        "mfcc_std_mean_raw_before_cmvn": mfcc_std_mean_raw_before_cmvn,  # Raw MFCC std value before CMVN
        "mfcc_std_mean_after_cmvn": mfcc_std_mean_after_cmvn,  # Raw MFCC std value after CMVN normalization
        "mfcc_std_label": mfcc_std_label,  # Quality label based on raw std thresholds
        "studentnet_logit_0": studentnet_results.get("studentnet_logit_0"),
        "studentnet_logit_1": studentnet_results.get("studentnet_logit_1"),
        "studentnet_probability": studentnet_results.get("studentnet_probability"),
        "studentnet_predicted_label": studentnet_results.get("studentnet_predicted_label", ""),
    }

    display_summary(
        file_path=file_path,
        transcript=transcript,
        detected_fillers=detected_fillers_list,
        filler_label=filler_label,
        wpm_label=wpm_label,
        pause_label=pause_label,
        metrics=metrics,
    )

    if show_plots:
        plt.figure(figsize=(12, 4))
        plt.plot(energy_times, rms_energy, label="RMS Energy", color="blue")
        plt.axhline(y=threshold, color="red", linestyle="--", label=f"Silence Threshold = {threshold:.4f}")
        for i, is_silent in enumerate(silent_frames):
            if is_silent:
                plt.axvspan(
                    energy_times[i],
                    energy_times[i] + (energy_times[1] - energy_times[0]),
                    color="lightgrey",
                    alpha=0.4,
                )
        plt.title("RMS Energy & Detected Pauses")
        plt.xlabel("Time (s)")
        plt.ylabel("Energy")
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 4))
        librosa.display.specshow(mfccs, x_axis="time", sr=sr, cmap="coolwarm")
        plt.colorbar(format="%+2.0f dB")
        plt.title("MFCC Heatmap")
        plt.ylabel("MFCC Coefficients")
        plt.xlabel("Time (s)")
        plt.tight_layout()
        plt.show()

        coeffs = np.arange(1, mfccs.shape[0] + 1)
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.bar(coeffs, mfcc_mean, color="skyblue")
        plt.xlabel("MFCC Coefficient")
        plt.ylabel("Mean Value")
        plt.title("MFCC Coefficient Mean")
        plt.subplot(1, 2, 2)
        plt.bar(coeffs, mfcc_var, color="salmon")
        plt.xlabel("MFCC Coefficient")
        plt.ylabel("Variance")
        plt.title("MFCC Coefficient Variance (Expressiveness)")
        plt.tight_layout()
        plt.show()

    return metrics


def write_results_to_csv(results: List[Dict[str, Optional[float]]], csv_path: str) -> None:
    """
    Write results to CSV file. If file is locked, try alternative filename.
    """
    import time
    
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for row in results:
                sanitized = {key: row.get(key, "") for key in CSV_FIELDS}

                for key, value in sanitized.items():
                    if isinstance(value, (np.floating, np.integer)):
                        sanitized[key] = float(value)

                writer.writerow(sanitized)
    except PermissionError:
        # File is likely open in another program (Excel, etc.)
        # Try with a timestamped filename
        base_path = os.path.splitext(csv_path)[0]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        new_csv_path = f"{base_path}_{timestamp}.csv"
        print(f"\n⚠️  Cannot write to {csv_path} (file may be open in another program)")
        print(f"   Trying alternative filename: {os.path.basename(new_csv_path)}")
        
        try:
            with open(new_csv_path, "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
                writer.writeheader()
                for row in results:
                    sanitized = {key: row.get(key, "") for key in CSV_FIELDS}

                    for key, value in sanitized.items():
                        if isinstance(value, (np.floating, np.integer)):
                            sanitized[key] = float(value)

                    writer.writerow(sanitized)
            print(f"✅ Successfully saved to: {new_csv_path}")
        except Exception as e:
            print(f"❌ Failed to write CSV file: {e}")
            print(f"   Please close any programs that may have the CSV file open and try again.")
            raise


def batch_process_audio_folder(audio_folder: str = None, output_csv: str = None, show_plots: bool = False) -> None:
    """
    Batch process all audio files in the audio folder and save to CSV.
    
    Args:
        audio_folder: Path to audio folder (default: ./audio relative to script)
        output_csv: Output CSV path (default: audio_folder/speech_analysis_results.csv)
        show_plots: Whether to show plots for each file
    """
    import glob
    
    if audio_folder is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        audio_folder = os.path.join(script_dir, "audio")
    
    if not os.path.isdir(audio_folder):
        print(f"❌ Audio folder not found: {audio_folder}")
        return
    
    # Get all .wav files
    audio_pattern = os.path.join(audio_folder, "*.wav")
    audio_files = glob.glob(audio_pattern)
    
    if not audio_files:
        print(f"❌ No .wav files found in {audio_folder}")
        return
    
    print(f"\n📁 Found {len(audio_files)} audio file(s) in {audio_folder}")
    print("Files to process:")
    for i, f in enumerate(audio_files, 1):
        print(f"  {i}. {os.path.basename(f)}")
    
    # Load Whisper model
    print("\n🧠 Loading Whisper model (base)...")
    model = whisper.load_model("base")
    print("✅ Model loaded successfully!")
    
    # Process each file
    results: List[Dict[str, Optional[float]]] = []
    for i, audio_path in enumerate(audio_files, 1):
        print(f"\n{'='*60}")
        print(f"Processing file {i}/{len(audio_files)}: {os.path.basename(audio_path)}")
        print(f"{'='*60}")
        try:
            metrics = analyze_audio(audio_path, model=model, show_plots=show_plots)
            results.append(metrics)
            print(f"✅ Successfully processed: {os.path.basename(audio_path)}")
        except Exception as exc:
            print(f"❌ Failed to analyze {os.path.basename(audio_path)}: {exc}")
            import traceback
            traceback.print_exc()
    
    # Save to CSV
    if results:
        if output_csv is None:
            # Always save to BETA CSV in the audio folder
            output_csv = os.path.join(audio_folder, "speech_analysis_results_BETA.csv")
        else:
            # If output_csv is provided but it's just a filename, save it in the audio folder
            if not os.path.dirname(output_csv):
                output_csv = os.path.join(audio_folder, output_csv)
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        
        print(f"\n💾 Saving results to: {output_csv}")
        print(f"   Full path: {os.path.abspath(output_csv)}")
        write_results_to_csv(results, output_csv)
        print(f"✅ Successfully saved analysis for {len(results)} file(s) to {output_csv}")
    else:
        print("\n⚠️ No results to save.")


def main() -> None:
    # Automatically process all files in audio folder (no prompts)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    audio_folder = os.path.join(script_dir, "audio")
    batch_process_audio_folder(audio_folder=audio_folder, show_plots=False)
    input("\nPress Enter to exit...")


if __name__ == "__main__":
    import sys
    
    # Automatically process all files in audio folder and save to BETA CSV (no prompts)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    audio_folder = os.path.join(script_dir, "audio")
    
    # Check if a specific folder is provided as argument (e.g., "nisaa_audio")
    if len(sys.argv) > 1:
        folder_name = sys.argv[1]
        if folder_name == "nisaa_audio":
            audio_folder = os.path.join(script_dir, "audio", "nisaa_audio")
        elif os.path.isdir(folder_name):
            audio_folder = folder_name
        elif os.path.isdir(os.path.join(script_dir, folder_name)):
            audio_folder = os.path.join(script_dir, folder_name)
    
    if not os.path.isdir(audio_folder):
        print(f"❌ Folder not found: {audio_folder}")
        input("\nPress Enter to exit...")
    else:
        batch_process_audio_folder(audio_folder=audio_folder, show_plots=False)
        input("\nPress Enter to exit...")
