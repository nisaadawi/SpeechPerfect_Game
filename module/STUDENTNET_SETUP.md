# StudentNet Setup Guide

## Overview

StudentNet is a voice-based stress recognition model that detects stress levels from audio recordings. It uses Wav2Vec2 embeddings to analyze voice patterns and predict stress.

## Prerequisites

- Python 3.7 or higher
- PyTorch (CPU or CUDA)
- Internet connection (for downloading models from Hugging Face)

## Installation

### Step 1: Install Required Dependencies

Install the required Python packages:

```bash
pip install torch torchaudio huggingface-hub
```

Or if you prefer using a requirements file, create one with:

```
torch>=1.7.0
torchaudio>=0.7.0
huggingface-hub>=0.16.0
```

**For Windows PowerShell:**
```powershell
pip install torch torchaudio huggingface-hub
```

**For CUDA support (if you have an NVIDIA GPU):**
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install huggingface-hub
```

### Step 2: Verify Installation

Test that everything is installed correctly:

```python
import torch
import torchaudio
from huggingface_hub import hf_hub_download

print(f"PyTorch version: {torch.__version__}")
print(f"Torchaudio version: {torchaudio.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

## Usage

### Basic Usage

```python
from module.studentnet import StressDetector

# Initialize the detector (this will download models on first run)
detector = StressDetector()

# Analyze an audio file
result = detector.predict_stress("path/to/your/audio.wav")

# Print results
print(f"Status: {result['status']}")
print(f"Stress Probability: {result['stress_probability']:.2%}")
print(f"Confidence: {result['confidence']}")
```

### Complete Example

```python
from module.studentnet import StressDetector
import os

# Initialize detector
print("Initializing stress detector...")
detector = StressDetector()

# Path to your audio file
audio_file = "module/audio/your_audio.wav"

if os.path.exists(audio_file):
    # Predict stress
    result = detector.predict_stress(audio_file)
    
    # Display results
    print("\n" + "="*50)
    print("STRESS DETECTION RESULTS")
    print("="*50)
    print(f"Not Stressed: {result['not_stressed_prob']*100:.2f}%")
    print(f"Stressed:     {result['stressed_prob']*100:.2f}%")
    print(f"\nStatus: {result['status']}")
    print(f"Confidence: {result['confidence']}")
    print("="*50)
else:
    print(f"Audio file not found: {audio_file}")
```

### Using Pre-computed Embeddings

If you already have Wav2Vec2 embeddings:

```python
import torch
from module.studentnet import StressDetector

detector = StressDetector()

# Your embedding should be shape (1, 512) or (512,)
embedding = torch.randn(1, 512)  # Replace with your actual embedding

result = detector.predict_stress_from_embedding(embedding)
print(f"Status: {result['status']}")
```

## Output Format

The `predict_stress()` method returns a dictionary with:

- `not_stressed_prob`: Probability of not being stressed (0.0 to 1.0)
- `stressed_prob`: Probability of being stressed (0.0 to 1.0)
- `stress_probability`: Same as `stressed_prob` (for convenience)
- `status`: "Stressed" or "Not Stressed"
- `confidence`: "High", "Medium", or "Low" (based on prediction confidence)

## Audio Requirements

- **Format**: WAV format is recommended (other formats supported by torchaudio also work)
- **Sample Rate**: Any sample rate (will be automatically resampled to 16kHz)
- **Channels**: Mono or stereo (will be converted to mono automatically)
- **Duration**: Any duration (model works on variable-length audio)

## First Run

On the first run, the script will:
1. Download the StudentNet model code from Hugging Face
2. Download the pretrained model weights (~few MB)
3. Download the Wav2Vec2 base model (~300 MB)

This may take a few minutes depending on your internet connection. Subsequent runs will be much faster as the models are cached.

## Troubleshooting

### Issue: "CUDA out of memory"
**Solution**: The model will automatically use CPU if CUDA is not available. To force CPU usage:
```python
detector = StressDetector(device="cpu")
```

### Issue: "Audio file not found"
**Solution**: Make sure the audio file path is correct. Use absolute paths or relative paths from your working directory.

### Issue: "ModuleNotFoundError: No module named 'torch'"
**Solution**: Install PyTorch:
```bash
pip install torch torchaudio
```

### Issue: Slow first run
**Solution**: This is normal. The models are being downloaded from Hugging Face. Subsequent runs will be faster.

### Issue: "Hugging Face authentication required"
**Solution**: The model is public, so authentication shouldn't be required. If you encounter this, try:
```python
from huggingface_hub import login
login()  # Follow prompts to authenticate
```

## Integration with Your Project

You can integrate StudentNet into your speech recognition pipeline:

```python
from module.studentnet import StressDetector

class SpeechAnalyzer:
    def __init__(self):
        self.stress_detector = StressDetector()
    
    def analyze_speech(self, audio_path):
        # Your existing speech analysis
        # ...
        
        # Add stress detection
        stress_result = self.stress_detector.predict_stress(audio_path)
        
        return {
            # Your existing results
            # ...
            "stress": stress_result
        }
```

## Model Information

- **Model**: StudentNet (distilled from TeacherNet)
- **Input**: 512-dimensional Wav2Vec2 embeddings
- **Output**: Binary classification (Stressed/Not Stressed)
- **Accuracy**: ~76% (as reported in model card)
- **Source**: [Hugging Face Model Hub](https://huggingface.co/forwarder1121/voice-based-stress-recognition)

## Performance Tips

1. **GPU Acceleration**: If you have a CUDA-capable GPU, the model will automatically use it for faster inference.

2. **Batch Processing**: For multiple audio files, you can process them in a loop:
   ```python
   audio_files = ["audio1.wav", "audio2.wav", "audio3.wav"]
   results = []
   for audio_file in audio_files:
       result = detector.predict_stress(audio_file)
       results.append(result)
   ```

3. **Reuse Detector**: Initialize the detector once and reuse it for multiple predictions to avoid reloading models.

## Support

For issues or questions:
- Check the model card: https://huggingface.co/forwarder1121/voice-based-stress-recognition
- Review the code in `module/studentnet.py`
- Check PyTorch and torchaudio documentation

