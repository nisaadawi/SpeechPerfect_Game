# emotion_recognizer.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from collections import Counter
import time
import os

from firebase_config import db, user  # import Firebase connection

# Load face detector and emotion model
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

_model_path = os.path.join(os.path.dirname(__file__), "fer2013_mini_XCEPTION.102-0.66.hdf5")
if not os.path.exists(_model_path):
    raise FileNotFoundError("Emotion model file not found!")

emotion_model = load_model(_model_path)

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Start webcam
cap = cv2.VideoCapture(0)
start_time = time.time()
emotion_log = []

print("🎥 Starting facial emotion recognition...\n(Press 'q' to quit)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (64, 64), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = emotion_model.predict(roi)[0]
            label = emotion_labels[np.argmax(prediction)]
            emotion_log.append(label)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        else:
            cv2.putText(frame, "No Face Found", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Facial Emotion Recognition', frame)

    # Every 10 seconds, upload the most common emotion to Firebase
    if time.time() - start_time >= 10:
        if emotion_log:
            most_common = Counter(emotion_log).most_common(1)[0][0]
            print(f"🕒 Emotion summary (last 10 seconds): {most_common}")

            if user:
                db.child("emotion").set(most_common, user['idToken'])
                print("✅ Emotion uploaded to Firebase.")
        else:
            print("🕒 No emotion detected.")
        emotion_log.clear()
        start_time = time.time()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 Emotion recognition stopped.")
