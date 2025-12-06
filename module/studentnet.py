import sys
import os
import csv
import glob
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

# Run batch processing on all audio files
if __name__ == "__main__":
    # Path to audio folder
    audio_folder = r"C:\Users\user\OneDrive\Documents\FYP ND\RenPy_Game\SpeechPerfect\module\audio"
    
    # Find all audio files (common audio formats)
    audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.m4a', '*.ogg']
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(glob.glob(os.path.join(audio_folder, ext)))
        audio_files.extend(glob.glob(os.path.join(audio_folder, ext.upper())))
    
    # Filter out CSV files
    audio_files = [f for f in audio_files if not f.endswith('.csv')]
    
    if not audio_files:
        print(f"No audio files found in {audio_folder}")
        sys.exit(1)
    
    print(f"Found {len(audio_files)} audio file(s) to process...")
    print("-" * 80)
    
    # Prepare CSV output file path
    csv_output_path = os.path.join(audio_folder, "studentnet_results.csv")
    
    # Process all audio files and collect results
    results = []
    for idx, audio_file in enumerate(audio_files, 1):
        file_name = os.path.basename(audio_file)
        print(f"[{idx}/{len(audio_files)}] Processing: {file_name}")
        
        try:
            stress_prob, stress_label, logit0, logit1 = predict_stress(audio_file)
            
            # Store results
            results.append({
                'file_name': file_name,
                'file_path': audio_file,
                'logit_0': logit0,
                'logit_1': logit1,
                'probability': stress_prob,
                'predicted_label': stress_label
            })
            
            print(f"  ✓ Logit 0 (Not Stressed): {logit0:.4f}")
            print(f"  ✓ Logit 1 (Stressed):     {logit1:.4f}")
            print(f"  ✓ Stress Probability: {stress_prob*100:.2f}%")
            print(f"  ✓ Predicted Label:    {stress_label}")
            print()
            
        except Exception as e:
            print(f"  ✗ Error processing {file_name}: {str(e)}")
            print()
            # Still add entry with error info
            results.append({
                'file_name': file_name,
                'file_path': audio_file,
                'logit_0': None,
                'logit_1': None,
                'probability': None,
                'predicted_label': f'Error: {str(e)}'
            })
    
    # Write results to CSV file
    print(f"\nWriting results to: {csv_output_path}")
    csv_columns = ['file_name', 'file_path', 'logit_0', 'logit_1', 'probability', 'predicted_label']
    
    with open(csv_output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"✅ Successfully processed {len([r for r in results if r['logit_0'] is not None])} file(s)")
    print(f"✅ Results saved to: {csv_output_path}")











