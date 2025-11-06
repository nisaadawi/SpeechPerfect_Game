import cv2
import time
from gaze_tracking import GazeTracking

# Initialize gaze tracker
gaze = GazeTracking()

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot access camera.")
    exit()

print("🎥 Simple emotion tracking started. Press 'q' to quit.")

emotion_log = []
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to read from camera.")
        break

    # Refresh gaze tracker
    gaze.refresh(frame)
    annotated_frame = gaze.annotated_frame()

    # Simple emotion detection (placeholder)
    emotion = "Neutral"  # Default emotion
    
    # Eye contact detection
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

    # Display information
    cv2.putText(annotated_frame, f"Emotion: {emotion}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(annotated_frame, f"Eye Contact: {eye_contact}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"Gaze: {gaze_text}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Show camera
    cv2.imshow("Simple Emotion + Gaze Tracking", annotated_frame)

    # Log emotion every 10 seconds
    if time.time() - start_time >= 10:
        emotion_log.append(emotion)
        print(f"🕒 Emotion logged: {emotion}")
        start_time = time.time()

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
print("👋 Simple emotion tracking ended.")

