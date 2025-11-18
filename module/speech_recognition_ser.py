import torch
import librosa
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

# ==============================
# 1. Load Pretrained SER Model
# ==============================
model_name = "superb/wav2vec2-base-superb-er"

feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForAudioClassification.from_pretrained(model_name)


# ==============================
# 2. Emotion Analysis Function
# ==============================
def analyze_emotion(audio_path):
    # Load audio file at 16 kHz
    speech, sr = librosa.load(audio_path, sr=16000)

    # Extract features
    inputs = feature_extractor(
        speech,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )

    # Forward pass through model
    with torch.no_grad():
        outputs = model(**inputs)

    # Convert logits → probabilities
    probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    # Get emotion labels
    labels = model.config.id2label

    # ================
    # Print Results
    # ================
    print("\n======= EMOTION PROBABILITIES =======")
    for i, p in enumerate(probs):
        print(f"{labels[i]}: {p*100:.2f}%")

    # Dominant emotion
    dominant_idx = probs.argmax()
    dominant_emotion = labels[dominant_idx]

    print("\n💡 Dominant Emotion:", dominant_emotion)

    return probs, labels, dominant_emotion


# ==============================
# 3. Run on Your Audio File
# ==============================
audio_file = r"C:\Users\user\OneDrive\Documents\FYP ND\RenPy_Game\SpeechPerfect\module\audio\nisaa_bad.wav"

probs, labels, dominant_emotion = analyze_emotion(audio_file)
