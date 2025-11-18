import sys
import torch
import torch.nn.functional as F
import librosa

# Add StudentNet repo to Python path
sys.path.append(r"C:\Users\user\OneDrive\Documents\FYP ND\RenPy_Game\SpeechPerfect\voice-based-stress-recognition")

from models import StudentNet  # now Python can find it

# Setup device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = StudentNet().to(device)

# Load pretrained weights (.bin file)
# .bin files contains pretrined wight of StudentNet model
weights_path = r"C:\Users\user\OneDrive\Documents\FYP ND\RenPy_Game\SpeechPerfect\voice-based-stress-recognition\pytorch_model.bin"
state = torch.load(weights_path, map_location=device)

# Load the pretrained learned wieghts or the brain
model.load_state_dict(state)
# Put model in inference mode, so it can predict
model.eval()

# Wav2Vec2 feature extractor
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

# Use the Wav2Vec2 base model (768 hidden dimensions)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
w2v_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)
w2v_model.eval()

# Function to extract embedding from audio
def get_embedding(audio_path):
    waveform, sr = librosa.load(audio_path, sr=16000)
    waveform = torch.tensor(waveform).unsqueeze(0).to(device)  # shape [1, seq]
    with torch.no_grad():
        outputs = w2v_model(waveform, return_dict=True) # pass audio to teacher model
        hidden_states = outputs.last_hidden_state  # get the final hidden states (768-dim), [1, seq_len, hidden_dim]
        embedding = hidden_states.mean(dim=1)     # mean pooling over time to 1 vector
        embedding = embedding[:, :512]            # truncate/pad/trim to 512-dim
    return embedding # return the 512-dim embedding of your audio
 

# Predict stress probability + label + logits
def predict_stress(audio_path, threshold=0.5):
    embedding = get_embedding(audio_path)
    with torch.no_grad():
        logits, _ = model(embedding)

        # Extract raw logit values
        logit_not_stress = logits[0, 0].item()
        logit_stress = logits[0, 1].item()

        # Convert logits → probabilities (softmax)
        # softmax is a function that converts/turns logits to probabilities(0-1)
        probs = F.softmax(logits, dim=1)
        stress_prob = probs[0, 1].item()  # index 1 = stressed

        # Decide label
        stress_label = "Stressed" if stress_prob >= threshold else "Not Stressed"

    return stress_prob, stress_label, logit_not_stress, logit_stress

# Run example
if __name__ == "__main__":
    audio_file = r"C:\Users\user\OneDrive\Documents\FYP ND\RenPy_Game\SpeechPerfect\module\audio\glen_powel.wav"
    stress_prob, stress_label, logit0, logit1 = predict_stress(audio_file)

    print(f"Logit 0 (Not Stressed): {logit0:.4f}")
    print(f"Logit 1 (Stressed):     {logit1:.4f}")
    print(f"\nStress Probability: {stress_prob*100:.2f}%") # convert probability to percentage
    print(f"Predicted Label:    {stress_label}")











