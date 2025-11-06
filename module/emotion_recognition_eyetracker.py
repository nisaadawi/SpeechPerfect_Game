import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from collections import Counter
from gaze_tracking import GazeTracking
import time

# ==============================
# SETTINGS
# ==============================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "fer2013_mini_XCEPTION.102-0.66.hdf5")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Emotion model not found at {MODEL_PATH}")

# ==============================
# LOAD MODELS
# ==============================
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
emotion_model = load_model(MODEL_PATH)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize gaze tracker
gaze = GazeTracking()

# ==============================
# INITIALIZE CAMERA
# ==============================
emotion_log = []
start_time = time.time()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot access camera.")
    exit()

print("🎥 Camera started. Press 'q' to quit.")

# ==============================
# MAIN LOOP
# ==============================
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to read from camera.")
        break

    # Refresh gaze tracker
    gaze.refresh(frame)
    annotated_frame = gaze.annotated_frame()

    # Detect faces for emotion
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (64, 64), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = emotion_model.predict(roi, verbose=0)[0]
            label = emotion_labels[preds.argmax()]
            emotion_log.append(label)
        else:
            label = "Unknown"

        # Eye contact (if gaze detected)
        if gaze.is_blinking():
            eye_contact = "No"
        else:
            eye_contact = "Yes"

        # Gaze direction
        if gaze.is_right():
            gaze_text = "Right"
        elif gaze.is_left():
            gaze_text = "Left"
        elif gaze.is_center():
            gaze_text = "Center"
        else:
            gaze_text = "Unknown"

        # Display text
        cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(annotated_frame, f"Emotion: {label}", (x, y - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(annotated_frame, f"Eye: {eye_contact}", (x, y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Gaze: {gaze_text}", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Show camera
    cv2.imshow("Emotion + Gaze Tracking", annotated_frame)

    # Every 10 seconds → summarize emotion
    if time.time() - start_time >= 10:
        if emotion_log:
            most_common = Counter(emotion_log).most_common(1)[0][0]
            print(f"🕒 Emotion summary (last 10s): {most_common}")
        else:
            print("🕒 No emotion detected in last 10s.")

        emotion_log.clear()
        start_time = time.time()

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==============================
# CLEAN UP
# ==============================
cap.release()
cv2.destroyAllWindows()
print("👋 Program ended.")
