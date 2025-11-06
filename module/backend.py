# backend.py
import time

import cv2
from flask import Flask, jsonify, Response
from flask_cors import CORS
from gaze_tracker import GazeTracker

app = Flask(__name__)
CORS(app)

focus_state = {"focused": False}
tracker = GazeTracker(show_window=False)


def _update_focus_state(focused: bool):
    focus_state["focused"] = focused


@app.before_request
def _ensure_tracker_running():
    if not tracker.is_running():
        tracker.start(callback=_update_focus_state)

@app.route("/focus", methods=["GET"])
def get_focus_state():
    return jsonify(focus_state)


def _generate_video_stream():
    while True:
        frame = tracker.get_latest_frame()
        if frame is None:
            time.sleep(0.03)
            continue

        success, encoded = cv2.imencode(".jpg", frame)
        if not success:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n"
        )


@app.route("/video_feed")
def video_feed():
    if not tracker.is_running():
        tracker.start(callback=_update_focus_state)
    return Response(
        _generate_video_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )

if __name__ == "__main__":
    tracker.start(callback=_update_focus_state)
    app.run(host="0.0.0.0", port=5000)
