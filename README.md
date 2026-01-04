# SpeechPerfect 🎤

**AI-Powered Multimodal Speech Assessment System**

SpeechPerfect is a comprehensive speech analysis platform that combines multiple modalities (speech, eye tracking, heart rate monitoring, and emotion recognition) to provide real-time feedback and assessment of public speaking performance. The system includes a gamified training environment built with a modern React-based dashboard for detailed analytics.

## 🌟 Features

### Core Capabilities
- **🎙️ Speech Analysis**
  - Real-time speech transcription using OpenAI Whisper
  - Words Per Minute (WPM) calculation
  - Filler word detection ("um", "uh", "like", etc.)
  - Pause analysis and duration tracking
  - MFCC (Mel-frequency cepstral coefficients) analysis for vocal expression
  - Stress detection using StudentNet (voice-based stress recognition)
  - Comprehensive speech scoring system

- **👁️ Eye Tracking & Attention Monitoring**
  - Real-time gaze tracking using webcam
  - Focus/attention percentage calculation
  - Blink detection
  - Gaze direction analysis (left, center, right)

- **❤️ Physiological Monitoring**
  - Heart rate monitoring via Arduino pulse sensors
  - Heart Rate Variability (HRV) analysis
  - Real-time BPM tracking
  - Stress indicators based on heart rate patterns

- **😊 Emotion Recognition**
  - Facial expression analysis using deep learning
  - Real-time emotion detection
  - Integration with speech analysis for comprehensive assessment

- **📊 Dashboard & Analytics**
  - React-based modern dashboard
  - Session-based tracking and reporting
  - Persona detection system (identifies speaking style)
  - Historical performance tracking
  - Detailed metrics visualization

- **🎮 Gamified Training**
  - Ren'Py-based visual novel game environment
  - Multiple difficulty levels (Easy, Medium, Hard)
  - Interactive heckler system for realistic practice scenarios
  - Real-time feedback during speech practice

## 🏗️ Architecture

### Backend Components
- **Flask API** (`module/backend.py`): Main backend server providing RESTful APIs
  - Audio analysis endpoints
  - Real-time data collection
  - Eye tracking integration
  - Model caching for performance

- **PHP Backend** (`backend_api/`): Database operations and session management
  - User authentication (login/register)
  - Session creation and management
  - Data persistence
  - Dashboard data retrieval

- **Python Modules** (`module/`): Core analysis modules
  - Speech recognition (Whisper, Wav2Vec2)
  - Stress detection (StudentNet)
  - Emotion recognition
  - Real-time data collection
  - Gaze tracking integration

### Frontend Components
- **React Dashboard** (`speechperfect/`): Modern web interface
  - User authentication
  - Real-time monitoring
  - Game setup and calibration
  - Session reports and analytics
  - Persona visualization

### Hardware Integration
- **Arduino Sensors** (`arduino/`, `sensor_hr_bpm/`): Heart rate monitoring
  - Pulse sensor (finger/wrist)
  - Real-time BPM data transmission

## 📋 Prerequisites

### Software Requirements
- **Python 3.7+** (Python 3.9 recommended)
- **Node.js 16+** and npm
- **MySQL** database
- **FFmpeg** (included in `ffmpeg/` directory)
- **Webcam** for eye tracking
- **Microphone** for speech recording

### Hardware Requirements (Optional)
- Arduino-compatible board
- Pulse sensor (for heart rate monitoring)

## 📊 Analysis Metrics

### Speech Metrics
- **WPM (Words Per Minute)**: Speech rate (ideal: 120-180 WPM)
- **Filler Words**: Count of filler words per minute
- **Pause Duration**: Average pause length in seconds
- **MFCC**: Vocal expression and clarity indicators
- **Stress Level**: Probability of stress (0.0-1.0)
- **Speech Score**: Overall composite score (0-100)

### Physiological Metrics
- **Heart Rate (BPM)**: Average beats per minute
- **HRV**: Heart rate variability indicators
- **Attention Percentage**: Focus level based on eye tracking

### Persona System
The system automatically detects speaking personas based on performance:
- **The Confident Performer** ⭐
- **The Nervous Speaker** 😰
- **The Fast Talker** ⚡
- **The Filler-Heavy Speaker** 🗣️
- **The Overthinker** 🤔
- **The Calm Communicator** 😌
- **The Developing Speaker** 📈

## Interfaces

### Main Menu

<img width="600" height="300" alt="Mainmenu" src="https://github.com/user-attachments/assets/31731564-6ac8-482d-aaad-aaaea5e87d46" />

### Heckler Session

<img width="600" height="300" alt="HecklerSession" src="https://github.com/user-attachments/assets/e1d328a3-8c30-4e3b-816e-dc07198132a2" />

### Calibration Page

<img width="600" height="300" alt="calibrationPage" src="https://github.com/user-attachments/assets/8c8711b8-0a61-4d4e-b75b-4bd035efb83a" />


## 🙏 Acknowledgments

- GazeTracking library by Antoine Lamé
- StudentNet model from Hugging Face
- OpenAI Whisper for speech recognition
