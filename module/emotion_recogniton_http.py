import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from collections import Counter
import requests
import time

# ==============================
# SETTINGS
# ==============================
API_URL = "http://humancc.site/ndhos/renpy_backend/http_add_emotions.php"  # Your PHP API endpoint

# Emotion model path (your .hdf5 file)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "fer2013_mini_XCEPTION.102-0.66.hdf5")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Emotion model not found at {MODEL_PATH}")

# ==============================
# LOAD MODELS
# ==============================
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
emotion_model = load_model(MODEL_PATH)

# Emotion labels (same order as model output)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# ==============================
# INITIALIZE CAMERA & LOG
# ==============================
emotion_log = []
start_time = time.time()

cap = cv2.VideoCapture(0)
print("🎥 Camera started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Camera not found or cannot be accessed.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (64, 64), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # Predict emotion
            preds = emotion_model.predict(roi)[0]
            label = emotion_labels[preds.argmax()]
            emotion_log.append(label)

            # Display on frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        else:
            cv2.putText(frame, "No Face Found", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Facial Emotion Recognition', frame)

    # Every 10 seconds → summarize & send to PHP
    if time.time() - start_time >= 10:
        if emotion_log:
            most_common = Counter(emotion_log).most_common(1)[0][0]
            print(f"🕒 Emotion summary (last 10s): {most_common}")

            try:
                response = requests.post(API_URL, data={"emotion": most_common})
                print("✅ Sent to server:", response.text)
            except Exception as e:
                print("❌ Failed to send emotion:", e)
        else:
            print("🕒 No emotion detected in last 10s.")

        emotion_log.clear()
        start_time = time.time()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==============================
# CLEAN UP
# ==============================
cap.release()
cv2.destroyAllWindows()
print("👋 Program ended.")