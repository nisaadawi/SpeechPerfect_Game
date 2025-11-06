import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from collections import Counter
import requests
import time
import sys

# ==============================
# SETTINGS
# ==============================
API_URL = "http://humancc.site/ndhos/renpy_backend/http_add_emotions.php"  # Your PHP API endpoint

# Emotion model path (your .hdf5 file)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "fer2013_mini_XCEPTION.102-0.66.hdf5")
if not os.path.exists(MODEL_PATH):
    print(f"❌ Emotion model not found at {MODEL_PATH}")
    sys.exit(1)

# ==============================
# LOAD MODELS
# ==============================
print("🔄 Loading emotion recognition models...")
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
emotion_model = load_model(MODEL_PATH)
print("✅ Models loaded successfully!")

# Emotion labels (same order as model output)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# ==============================
# INITIALIZE CAMERA & LOG
# ==============================
emotion_log = []
start_time = time.time()

print("🎥 Initializing camera...")
cap = cv2.VideoCapture(0)

# Check if camera is available
if not cap.isOpened():
    print("❌ Camera not found or cannot be accessed.")
    sys.exit(1)

print("✅ Camera started successfully!")
print("🎯 Emotion recognition is now running...")

# Set camera properties for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

frame_count = 0
last_emotion_sent = None

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to read frame from camera.")
            time.sleep(1)
            continue

        frame_count += 1
        
        # Process every 3rd frame to reduce CPU usage
        if frame_count % 3 != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        emotion_detected = False
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (64, 64), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                # Predict emotion
                preds = emotion_model.predict(roi, verbose=0)[0]
                label = emotion_labels[preds.argmax()]
                confidence = preds[preds.argmax()]
                
                # Only log emotions with reasonable confidence
                if confidence > 0.3:
                    emotion_log.append(label)
                    emotion_detected = True
                    print(f"🎭 Detected emotion: {label} (confidence: {confidence:.2f})")

        # Every 10 seconds → summarize & send to PHP
        if time.time() - start_time >= 10:
            if emotion_log:
                most_common = Counter(emotion_log).most_common(1)[0][0]
                print(f"🕒 Emotion summary (last 10s): {most_common}")

                # Only send if emotion changed
                if most_common != last_emotion_sent:
                    try:
                        response = requests.post(API_URL, data={"emotion": most_common}, timeout=5)
                        if response.status_code == 200:
                            print(f"✅ Sent to server: {most_common} - {response.text}")
                            last_emotion_sent = most_common
                        else:
                            print(f"⚠️ Server response: {response.status_code}")
                    except Exception as e:
                        print(f"❌ Failed to send emotion: {e}")
            else:
                print("🕒 No emotion detected in last 10s.")

            emotion_log.clear()
            start_time = time.time()

        # Small delay to prevent excessive CPU usage
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n🛑 Emotion recognition stopped by user.")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
finally:
    # ==============================
    # CLEAN UP
    # ==============================
    cap.release()
    print("👋 Emotion recognition program ended.")
