import os
import sys
import csv
import random
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import whisper
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2Model

# Try to import noisereduce for advanced denoising
try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False
    print("⚠️  noisereduce not available. Using basic spectral gating for denoising.")

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
    "category_label_wpm", # label for wpm
    "filler_count",
    "detected_fillers",
    "filler_rate_percent", 
    "fillers_per_min", 
    "filler_label", 
    "category_label_filler", # label for fillers
    "pause_count",
    "avg_pause_sec", 
    "total_pause_time_sec",
    "pause_label", 
    "category_label_pause", # label for pause
    "pause_durations",
    "pause_threshold",
    "mfcc_std_mean_raw_before_cmvn",
    "mfcc_std_mean_after_cmvn",
    "mfcc_std_label", 
    "category_mfcc_label", # label for mfcc
    "studentnet_logit_0",
    "studentnet_logit_1",
    "studentnet_probability",
    "studentnet_predicted_label", # label for stress model
    "category_label_stress", # 
    "avg_heart_rate_bpm",
    "heart_rate_label", # label for heart rate
    "category_label_heart_rate",
    "eye_tracker_not_focus_count", 
    "eye_tracker_label",
    "category_label_eye_tracker", # label for eye tracker
    "speech_param", # calculated speech parameter score
    "speech_score", # final calculated speech score
]

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
# Speech Score Formula Weightage
# ===============================
# Speech param weightage
BETA_WPM = 0.25
BETA_FILLER = 0.25
BETA_PAUSE = 0.25
BETA_MFCC = 0.25

# Speech score weightage
BETA_SPEECH_PARAM = 0.25
BETA_STRESS_MODEL = 0.25
BETA_HEART_RATE = 0.25
BETA_ATTENTION = 0.25


# ===============================
# Global StudentNet Model (loaded lazily) ///////////
# ===============================
STUDENTNET_MODEL = None
STUDENTNET_W2V_MODEL = None
STUDENTNET_DEVICE = None
STUDENTNET_PATH = r"C:\Users\user\OneDrive\Documents\FYP ND\RenPy_Game\SpeechPerfect\voice-based-stress-recognition"


# ===============================
# STRESS MODEL functions
# ===============================
def load_studentnet_model():
    """Load the StudentNet stress detection model."""
    global STUDENTNET_MODEL, STUDENTNET_W2V_MODEL, STUDENTNET_DEVICE
    
    if STUDENTNET_MODEL is None:
        model_load_start = time.time()
        print("🔄 Loading StudentNet stress detection model...")
        try:
            # Add StudentNet repo to Python path
            sys.path.append(STUDENTNET_PATH)
            
            from models import StudentNet
            
            # Setup device
            STUDENTNET_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"   Using device: {STUDENTNET_DEVICE}")
            
            # Initialize StudentNet model
            print("   Loading StudentNet architecture...")
            STUDENTNET_MODEL = StudentNet().to(STUDENTNET_DEVICE)
            
            # Load pretrained weights
            print("   Loading pretrained weights...")
            weights_path = os.path.join(STUDENTNET_PATH, "pytorch_model.bin")
            state = torch.load(weights_path, map_location=STUDENTNET_DEVICE)
            STUDENTNET_MODEL.load_state_dict(state)
            STUDENTNET_MODEL.eval()
            
            # Load Wav2Vec2 feature extractor (this can be slow - downloads/loads large model)
            print("   Loading Wav2Vec2 model (this may take a while on first run)...")
            w2v_start = time.time()
            STUDENTNET_W2V_MODEL = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(STUDENTNET_DEVICE)
            STUDENTNET_W2V_MODEL.eval()
            w2v_time = time.time() - w2v_start
            print(f"   Wav2Vec2 loaded in {w2v_time:.2f} seconds")
            
            total_load_time = time.time() - model_load_start
            print(f"✅ StudentNet model loaded successfully! (total: {total_load_time:.2f} seconds)")
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


# ===============================
# Audio Cleaning and Denoising Functions
# ===============================
def clean_audio(y: np.ndarray, sr: int, apply_denoising: bool = True, 
                apply_trimming: bool = True, apply_normalization: bool = True) -> Tuple[np.ndarray, int]:
    """
    Clean and denoise raw audio signal.
    
    Args:
        y: Audio time series
        sr: Sample rate
        apply_denoising: Whether to apply noise reduction
        apply_trimming: Whether to trim silence from beginning/end
        apply_normalization: Whether to normalize volume
    
    Returns:
        Cleaned audio time series and sample rate
    """
    y_cleaned = y.copy()
    
    # 1. Noise Reduction
    if apply_denoising:
        try:
            if NOISEREDUCE_AVAILABLE:
                # Use noisereduce library for advanced denoising
                y_cleaned = nr.reduce_noise(
                    y=y_cleaned, 
                    sr=sr,
                    stationary=False,  # Non-stationary noise (better for speech)
                    prop_decrease=0.8  # Reduce 80% of noise
                )
            else:
                # Fallback: High-pass filter to remove low-frequency noise
                try:
                    from scipy import signal
                    # High-pass filter at 80Hz to remove low-frequency noise
                    sos = signal.butter(4, 80, 'hp', fs=sr, output='sos')
                    y_cleaned = signal.sosfilt(sos, y_cleaned)
                except ImportError:
                    # If scipy not available, use librosa's preemphasis as simple high-pass
                    y_cleaned = librosa.effects.preemphasis(y_cleaned, coef=0.97)
        except Exception as e:
            print(f"⚠️  Denoising failed: {e}. Continuing with original audio.")
            y_cleaned = y.copy()
    
    # 2. Trim silence from beginning and end
    if apply_trimming:
        try:
            y_cleaned, _ = librosa.effects.trim(
                y_cleaned, 
                top_db=20,  # Remove parts quieter than 20dB below peak
                frame_length=2048,
                hop_length=512
            )
        except Exception as e:
            print(f"⚠️  Silence trimming failed: {e}. Continuing with original audio.")
    
    # 3. Normalize volume (peak normalization)
    if apply_normalization:
        try:
            # Normalize to [-1, 1] range with peak at 0.95 to avoid clipping
            max_val = np.max(np.abs(y_cleaned))
            if max_val > 0:
                y_cleaned = y_cleaned / max_val * 0.95
        except Exception as e:
            print(f"⚠️  Normalization failed: {e}. Continuing with original audio.")
    
    return y_cleaned, sr


# DISPLAY SUMMARY FUNCTION
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

    print("\n🎶 MFCC-Based Voice Analysis:")
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

    print("\n❤️ Heart Rate Analysis (Simulated):")
    bpm = metrics.get("avg_heart_rate_bpm")
    hr_label = metrics.get("heart_rate_label", "Unknown")
    if bpm is not None:
        print(f"   • Average Heart Rate: {bpm:.0f} BPM → {hr_label}")
    else:
        print("   • Heart rate data unavailable")

    print("\n👁️ Eye Tracker Analysis (Simulated):")
    not_focus_count = metrics.get("eye_tracker_not_focus_count")
    et_label = metrics.get("eye_tracker_label", "Unknown")
    if not_focus_count is not None:
        print(f"   • Not Focus Detections: {not_focus_count} → {et_label}")
    else:
        print("   • Eye tracker data unavailable")

    print("\n📊 Speech Score Calculation:")
    speech_param = metrics.get("speech_param")
    speech_score = metrics.get("speech_score")
    if speech_param is not None and speech_score is not None:
        # Get category labels for calculation breakdown
        cat_wpm = metrics.get("category_label_wpm", 0.0) or 0.0
        cat_filler = metrics.get("category_label_filler", 0.0) or 0.0
        cat_pause = metrics.get("category_label_pause", 0.0) or 0.0
        cat_mfcc = metrics.get("category_mfcc_label", 0.0) or 0.0
        cat_stress = metrics.get("category_label_stress")
        cat_heart_rate = metrics.get("category_label_heart_rate")
        cat_eye_tracker = metrics.get("category_label_eye_tracker")
        
        # Calculate intermediate values for display
        wpm_contrib = BETA_WPM * cat_wpm
        filler_contrib = BETA_FILLER * cat_filler
        pause_contrib = BETA_PAUSE * cat_pause
        mfcc_contrib = BETA_MFCC * cat_mfcc
        
        stress_value = (1.0 - (cat_stress or 0.0)) if cat_stress is not None else 0.0
        heart_rate_value = (1.0 - (cat_heart_rate or 0.0)) if cat_heart_rate is not None else 0.0
        attention_value = (1.0 - (cat_eye_tracker or 0.0)) if cat_eye_tracker is not None else 0.0
        
        speech_param_contrib = BETA_SPEECH_PARAM * speech_param
        stress_contrib = BETA_STRESS_MODEL * stress_value
        heart_rate_contrib = BETA_HEART_RATE * heart_rate_value
        attention_contrib = BETA_ATTENTION * attention_value
        
        print("\n   📐 Formula 1: Speech Parameter Score")
        print(f"      speech_param = (β_wpm × category_label_wpm) + (β_filler × category_label_filler) +")
        print(f"                    (β_pause × category_label_pause) + (β_mfcc × category_mfcc_label)")
        print(f"      speech_param = ({BETA_WPM} × {cat_wpm:.2f}) + ({BETA_FILLER} × {cat_filler:.2f}) +")
        print(f"                    ({BETA_PAUSE} × {cat_pause:.2f}) + ({BETA_MFCC} × {cat_mfcc:.2f})")
        print(f"      speech_param = {wpm_contrib:.4f} + {filler_contrib:.4f} + {pause_contrib:.4f} + {mfcc_contrib:.4f}")
        print(f"      speech_param = {speech_param:.4f}")
        
        print("\n   📐 Formula 2: Final Speech Score")
        print(f"      speech_score = (β_speech_param × speech_param) + (β_stress × (1 - category_label_stress)) +")
        print(f"                    (β_heart_rate × (1 - category_label_heart_rate)) +")
        print(f"                    (β_attention × (1 - category_label_eye_tracker))")
        print(f"      speech_score = ({BETA_SPEECH_PARAM} × {speech_param:.4f}) + ({BETA_STRESS_MODEL} × {stress_value:.2f}) +")
        print(f"                    ({BETA_HEART_RATE} × {heart_rate_value:.2f}) + ({BETA_ATTENTION} × {attention_value:.2f})")
        print(f"      speech_score = {speech_param_contrib:.4f} + {stress_contrib:.4f} + {heart_rate_contrib:.4f} + {attention_contrib:.4f}")
        print(f"      speech_score = {speech_score:.4f}")
        
        print(f"\n   ✅ Final Results:")
        print(f"      • Speech Parameter Score: {speech_param:.4f}")
        print(f"      • Final Speech Score: {speech_score:.4f}")
    else:
        print("   • Speech score calculation unavailable")


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
    # ===============================
    # Initial Setup: Transcription and Audio Loading
    # ===============================
    print("\n⏱️  Starting audio analysis...")
    analysis_start_time = time.time()
    
    # Load audio once - reuse for both transcription and analysis
    print("📂 Loading audio file...")
    audio_load_start = time.time()
    y, sr = librosa.load(file_path, sr=None)
    audio_load_time = time.time() - audio_load_start
    print(f"   Audio loaded in {audio_load_time:.2f} seconds")
    
    # Clean and denoise the audio BEFORE transcription (cleaner transcription)
    print("🧹 Cleaning and denoising audio...")
    cleaning_start = time.time()
    y, sr = clean_audio(y, sr, apply_denoising=True, apply_trimming=True, apply_normalization=True)
    cleaning_time = time.time() - cleaning_start
    print(f"✅ Audio cleaning complete! (took {cleaning_time:.2f} seconds)")
    
    # Save cleaned audio to temporary file for Whisper (or use audio array directly if supported)
    # For now, transcribe from original file (Whisper handles it efficiently)
    print("🎤 Transcribing audio with Whisper...")
    transcription_start = time.time()
    result = model.transcribe(file_path, language="en", fp16=False)
    transcription_time = time.time() - transcription_start
    print(f"✅ Transcription complete! (took {transcription_time:.2f} seconds)")
    
    transcript = result.get("text", "").strip()

    words = transcript.lower().split()
    detected_fillers_list = [w for w in words if w in FILLER_WORDS]
    filler_count = len(detected_fillers_list)
    
    duration_sec = float(librosa.get_duration(y=y, sr=sr))
    duration_min = duration_sec / 60 if duration_sec > 0 else 1.0
    word_count = len(words)

    #  CATEGORIZATION FUNCTIONS
    # ===============================
    # 1️⃣ WPM (Words Per Minute) Analysis
    # ===============================
    wpm = word_count / duration_min if duration_min > 0 else 0.0
    if 80 <= wpm <= 120:
        wpm_label = "🟩 Good" #1
        category_label_wpm = 1.0
    elif (60 <= wpm < 80) or (120 < wpm <= 140):
        wpm_label = "🟨 Okay" #1
        category_label_wpm = 1.0
    elif (40 <= wpm < 60) or (140 < wpm <= 160):
        wpm_label = "🟧 Moderate" #0.5
        category_label_wpm = 0.5
    else:
        wpm_label = "🟥 Severe" #0
        category_label_wpm = 0.0

    # ===============================
    # 2️⃣ Filler Words Analysis
    # ===============================
    fillers_per_min = filler_count / duration_min if duration_min > 0 else 0.0
    if fillers_per_min <= 2:
        filler_label = "🟩 Good"
        category_label_filler = 1.0
    elif 3 <= fillers_per_min <= 5:
        filler_label = "🟨 Moderate"
        category_label_filler = 0.5
    else:
        filler_label = "🟥 Severe"
        category_label_filler = 0.0

    # ===============================
    # 3️⃣ MFCC Standard Deviation Analysis
    # ===============================
    print("\n🎵 Computing MFCC features...")
    mfcc_start = time.time()
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_time = time.time() - mfcc_start
    print(f"   MFCC computation took {mfcc_time:.2f} seconds")
    
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
    
    # Calculate standard deviation from CMVN-normalized MFCCs
    mfcc_std_after_cmvn = np.std(mfccs_cmvn, axis=1)  # Raw std values after CMVN
    mfcc_std_mean_after_cmvn = float(np.mean(mfcc_std_after_cmvn))
    
    # MFCC standard deviation categorization based on raw values before CMVN
    # Using thresholds: < 14.0 = Monotone, 14.0-21.0 = Moderate, > 21.0 = Expressive
    if mfcc_std_mean_raw_before_cmvn < RAW_STD_MONOTONE_THRESHOLD:
        mfcc_std_label = "Severe"  #Monotone Bad
        category_mfcc_label = 0.0
    elif mfcc_std_mean_raw_before_cmvn < RAW_STD_MODERATE_THRESHOLD:
        mfcc_std_label = "Moderate"  #Moderate Okay
        category_mfcc_label = 0.5
    else:  # >= RAW_STD_MODERATE_THRESHOLD (i.e., >= 21.0)
        mfcc_std_label = "Good"  #Expressive Good
        category_mfcc_label = 1.0

    # ===============================
    # 4️⃣ Pause Analysis
    # ===============================
    print("\n⏸️  Analyzing pauses...")
    pause_analysis_start = time.time()
    rms_energy = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    frames = np.arange(len(rms_energy))
    energy_times = librosa.frames_to_time(frames, sr=sr, hop_length=512)

    # Calculate adaptive threshold for silence detection
    mean_rms = float(np.mean(rms_energy))
    threshold = max(mean_rms * 0.5, 0.005)

    # Detect silent frames (pauses)
    silent_frames = rms_energy < threshold
    pause_durations: List[float] = []
    pause_start: Optional[float] = None

    # Extract pause durations from silent frames
    for i, is_silent in enumerate(silent_frames):
        if is_silent and pause_start is None:
            pause_start = float(energy_times[i])
        elif not is_silent and pause_start is not None:
            pause_end = float(energy_times[i])
            duration = pause_end - pause_start
            if duration > 0.2:  # Only count pauses longer than 0.2 seconds
                pause_durations.append(duration)
            pause_start = None

    # Handle pause at the end of audio
    if pause_start is not None:
        duration = float(energy_times[-1]) - pause_start
        if duration > 0.2:
            pause_durations.append(duration)

    # Calculate pause statistics
    avg_pause = float(np.mean(pause_durations)) if pause_durations else 0.0
    total_pause_time = float(np.sum(pause_durations)) if pause_durations else 0.0
    pause_count = len(pause_durations)
    pause_analysis_time = time.time() - pause_analysis_start
    print(f"   Pause analysis took {pause_analysis_time:.2f} seconds")

    # Categorize pause quality
    if avg_pause <= 1:
        pause_label = "🟩 Good"
        category_label_pause = 1.0
    elif 1 < avg_pause <= 2:
        pause_label = "🟨 Moderate"
        category_label_pause = 0.5
    else:
        pause_label = "🟥 Severe"
        category_label_pause = 0.0

    # StudentNet-based Stress Detection using pretrained model
    print("\n🧠 Running stress detection with StudentNet...")
    stress_start = time.time()
    studentnet_results = detect_stress_studentnet(file_path, y, sr)
    stress_time = time.time() - stress_start
    print(f"✅ Stress detection complete! (took {stress_time:.2f} seconds)")
    
    # Calculate stress category label (Stressed = 1, Not Stressed = 0)
    studentnet_label = studentnet_results.get("studentnet_predicted_label", "")
    if studentnet_label == "Stressed":
        category_label_stress = 1.0
    elif studentnet_label == "Not Stressed":
        category_label_stress = 0.0
    else:
        category_label_stress = None  # Model unavailable or error

    # ===============================
    # 5️⃣ Heart Rate (BPM) Analysis (Simulated)
    # ===============================
    # Generate random BPM between 50-120
    avg_heart_rate_bpm = float(random.randint(50, 120))
    
    # Categorize: > 100 = anxious (1), <= 100 = relax (0)
    if avg_heart_rate_bpm > 100:
        heart_rate_label = "Anxious"
        category_label_heart_rate = 1.0
    else:
        heart_rate_label = "Relax"
        category_label_heart_rate = 0.0

    # ===============================
    # 6️⃣ Eye Tracker Analysis (Simulated)
    # ===============================
    # Simulate 3 time periods of eye tracker readings
    # Each time period can have multiple "not focus" detections (0-5 per period)
    # Sum all "not focus" detections across the 3 time periods
    eye_tracker_not_focus_count = sum([random.randint(0, 5) for _ in range(3)])
    
    # Categorize: > 10 = not focus (0), <= 10 = focus (1)
    if eye_tracker_not_focus_count > 10:
        eye_tracker_label = "Not Focus"
        category_label_eye_tracker = 0.0
    else:
        eye_tracker_label = "Focus"
        category_label_eye_tracker = 1.0

    # ===============================
    # 7️⃣ Speech Score Calculation
    # ===============================
    # Calculate speech_param: weighted sum of speech parameters
    speech_param = (
        (BETA_WPM * (category_label_wpm or 0.0)) +
        (BETA_FILLER * (category_label_filler or 0.0)) +
        (BETA_PAUSE * (category_label_pause or 0.0)) +
        (BETA_MFCC * (category_mfcc_label or 0.0))
    )
    
    # Calculate speech_score: weighted sum of speech_param and other factors
    # Note: For stress, heart_rate, and eye_tracker, we use (1 - category_label) 
    # because higher category_label means worse (stressed/anxious/not focus)
    # So we invert them to get a positive score
    stress_value = (1.0 - (category_label_stress or 0.0)) if category_label_stress is not None else 0.0
    heart_rate_value = (1.0 - (category_label_heart_rate or 0.0)) if category_label_heart_rate is not None else 0.0
    attention_value = (1.0 - (category_label_eye_tracker or 0.0)) if category_label_eye_tracker is not None else 0.0
    
    speech_score = (
        (BETA_SPEECH_PARAM * speech_param) +
        (BETA_STRESS_MODEL * stress_value) +
        (BETA_HEART_RATE * heart_rate_value) +
        (BETA_ATTENTION * attention_value)
    )

    metrics: Dict[str, Optional[float]] = {
        # file info
        "file_name": os.path.basename(file_path),
        "file_path": file_path,
        # transcript info
        "transcript": transcript,
        # audio info
        "duration_sec": duration_sec,
        "duration_min": duration_min,
        "word_count": word_count,
        # 1. speech rate info
        "speech_rate_wpm": wpm,
        "wpm_label": wpm_label,
        "category_label_wpm": category_label_wpm,
        # 2.filler info
        "filler_count": filler_count,
        "detected_fillers": ";".join(detected_fillers_list),
        "filler_rate_percent": (filler_count / word_count * 100) if word_count else 0.0,
        "fillers_per_min": fillers_per_min,
        "filler_label": filler_label,
        "category_label_filler": category_label_filler,
        #3. pause info
        "pause_count": pause_count,
        "avg_pause_sec": avg_pause,
        "total_pause_time_sec": total_pause_time,
        "pause_label": pause_label,
        "category_label_pause": category_label_pause,
        "pause_durations": ";".join(f"{d:.3f}" for d in pause_durations),
        "pause_threshold": threshold,
        # 4. mfcc info
        "mfcc_std_mean_raw_before_cmvn": mfcc_std_mean_raw_before_cmvn,  # Raw MFCC std value before CMVN
        "mfcc_std_mean_after_cmvn": mfcc_std_mean_after_cmvn,  # Raw MFCC std value after CMVN normalization
        "mfcc_std_label": mfcc_std_label,  # Quality label based on raw std thresholds
        "category_mfcc_label": category_mfcc_label,  # Category label for MFCC (Good=1, Moderate=0.5, Severe=0)
        "studentnet_logit_0": studentnet_results.get("studentnet_logit_0"),
        "studentnet_logit_1": studentnet_results.get("studentnet_logit_1"),
        "studentnet_probability": studentnet_results.get("studentnet_probability"),
        "studentnet_predicted_label": studentnet_results.get("studentnet_predicted_label", ""),
        "category_label_stress": category_label_stress,
        # 5. heart rate info
        "avg_heart_rate_bpm": avg_heart_rate_bpm,
        "heart_rate_label": heart_rate_label,
        "category_label_heart_rate": category_label_heart_rate,
        # 6. eye tracker info
        "eye_tracker_not_focus_count": eye_tracker_not_focus_count,
        "eye_tracker_label": eye_tracker_label,
        "category_label_eye_tracker": category_label_eye_tracker,
        # 7. speech score calculations
        "speech_param": round(float(speech_param), 4),
        "speech_score": round(float(speech_score), 4),
    }

    # Calculate total analysis time
    total_analysis_time = time.time() - analysis_start_time
    
    print(f"\n{'='*60}")
    print(f"⏱️  ANALYSIS TIMING BREAKDOWN")
    print(f"{'='*60}")
    print(f"   Audio loading:        {audio_load_time:>8.2f} seconds ({audio_load_time/total_analysis_time*100:>5.1f}%)")
    print(f"   Audio cleaning:       {cleaning_time:>8.2f} seconds ({cleaning_time/total_analysis_time*100:>5.1f}%)")
    print(f"   Whisper transcription: {transcription_time:>8.2f} seconds ({transcription_time/total_analysis_time*100:>5.1f}%)")
    print(f"   MFCC computation:     {mfcc_time:>8.2f} seconds ({mfcc_time/total_analysis_time*100:>5.1f}%)")
    print(f"   Pause analysis:       {pause_analysis_time:>8.2f} seconds ({pause_analysis_time/total_analysis_time*100:>5.1f}%)")
    print(f"   Stress detection:     {stress_time:>8.2f} seconds ({stress_time/total_analysis_time*100:>5.1f}%)")
    print(f"   {'─'*58}")
    print(f"   Total analysis time:  {total_analysis_time:>8.2f} seconds ({total_analysis_time/60:>5.2f} minutes)")
    print(f"{'='*60}")

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


def process_single_audio_file(audio_file_path: str, output_csv: str = None, show_plots: bool = False) -> float:
    """
    Process a single audio file and save results to CSV.
    
    Args:
        audio_file_path: Path to the audio file to process
        output_csv: Output CSV path (default: same directory as audio file with name speech_analysis_results_BETA.csv)
        show_plots: Whether to show plots
    
    Returns:
        Total processing time in seconds
    """
    start_time = time.time()
    
    if not os.path.isfile(audio_file_path):
        print(f"❌ Audio file not found: {audio_file_path}")
        return 0.0
    
    # Validate file extension
    valid_ext = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
    if os.path.splitext(audio_file_path)[1].lower() not in valid_ext:
        print(f"❌ Unsupported audio format. Supported formats: {', '.join(valid_ext)}")
        return 0.0
    
    # Set default output CSV path
    if output_csv is None:
        audio_dir = os.path.dirname(os.path.abspath(audio_file_path))
        output_csv = os.path.join(audio_dir, "speech_analysis_results_BETA.csv")
    else:
        # If output_csv is provided but it's just a filename, save it in the same directory as audio file
        if not os.path.dirname(output_csv):
            audio_dir = os.path.dirname(os.path.abspath(audio_file_path))
            output_csv = os.path.join(audio_dir, output_csv)
    
    print(f"\n📁 Processing audio file: {os.path.basename(audio_file_path)}")
    print(f"   Full path: {audio_file_path}")
    
    # Load Whisper model
    print("\n🧠 Loading Whisper model (base)...")
    model_load_start = time.time()
    model = whisper.load_model("base")
    model_load_time = time.time() - model_load_start
    print(f"✅ Model loaded successfully! (took {model_load_time:.2f} seconds)")
    
    # Process the file
    try:
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(audio_file_path)}")
        print(f"{'='*60}")
        analysis_start = time.time()
        metrics = analyze_audio(audio_file_path, model=model, show_plots=show_plots)
        analysis_time = time.time() - analysis_start
        print(f"\n⏱️  Audio analysis completed in {analysis_time:.2f} seconds")
        
        # Save to CSV
        results = [metrics]
        print(f"\n💾 Saving results to: {output_csv}")
        print(f"   Full path: {os.path.abspath(output_csv)}")
        csv_save_start = time.time()
        write_results_to_csv(results, output_csv)
        csv_save_time = time.time() - csv_save_start
        print(f"✅ Successfully saved analysis to {output_csv} (took {csv_save_time:.2f} seconds)")
        
        total_time = time.time() - start_time
        return total_time
        
    except Exception as exc:
        print(f"❌ Failed to analyze {os.path.basename(audio_file_path)}: {exc}")
        import traceback
        traceback.print_exc()
        return time.time() - start_time


def main() -> None:
    # Process single audio file
    import sys
    
    script_start_time = time.time()
    
    if len(sys.argv) > 1:
        # Audio file path provided as command line argument
        audio_file_path = sys.argv[1]
        audio_file_path = os.path.expanduser(audio_file_path)
        audio_file_path = os.path.abspath(audio_file_path)
    else:
        # Prompt for audio file path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_audio_folder = os.path.join(script_dir, "audio")
        print(f"Enter the path to the audio file to process:")
        print(f"(Default folder: {default_audio_folder})")
        user_input = input("Audio file path: ").strip().strip('"')
        
        if not user_input:
            print("❌ No audio file path provided.")
            input("\nPress Enter to exit...")
            return
        
        audio_file_path = os.path.expanduser(user_input)
        audio_file_path = os.path.abspath(audio_file_path)
    
    processing_time = process_single_audio_file(audio_file_path=audio_file_path, show_plots=False)
    total_script_time = time.time() - script_start_time
    
    print(f"\n{'='*60}")
    print(f"⏱️  TIMING SUMMARY")
    print(f"{'='*60}")
    print(f"   Processing time: {processing_time:.2f} seconds ({processing_time/60:.2f} minutes)")
    print(f"   Total script time: {total_script_time:.2f} seconds ({total_script_time/60:.2f} minutes)")
    print(f"{'='*60}")
    
    input("\nPress Enter to exit...")


if __name__ == "__main__":
    import sys
    
    script_start_time = time.time()
    
    # Process single audio file
    if len(sys.argv) > 1:
        # Audio file path provided as command line argument
        audio_file_path = sys.argv[1]
        audio_file_path = os.path.expanduser(audio_file_path)
        audio_file_path = os.path.abspath(audio_file_path)
    else:
        # Prompt for audio file path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_audio_folder = os.path.join(script_dir, "audio")
        print(f"Enter the path to the audio file to process:")
        print(f"(Default folder: {default_audio_folder})")
        user_input = input("Audio file path: ").strip().strip('"')
        
        if not user_input:
            print("❌ No audio file path provided.")
            input("\nPress Enter to exit...")
            sys.exit(1)
        
        audio_file_path = os.path.expanduser(user_input)
        audio_file_path = os.path.abspath(audio_file_path)
    
    if not os.path.isfile(audio_file_path):
        print(f"❌ Audio file not found: {audio_file_path}")
        input("\nPress Enter to exit...")
        sys.exit(1)
    
    processing_time = process_single_audio_file(audio_file_path=audio_file_path, show_plots=False)
    total_script_time = time.time() - script_start_time
    
    print(f"\n{'='*60}")
    print(f"⏱️  TIMING SUMMARY")
    print(f"{'='*60}")
    print(f"   Processing time: {processing_time:.2f} seconds ({processing_time/60:.2f} minutes)")
    print(f"   Total script time: {total_script_time:.2f} seconds ({total_script_time/60:.2f} minutes)")
    print(f"{'='*60}")
    
    input("\nPress Enter to exit...")
