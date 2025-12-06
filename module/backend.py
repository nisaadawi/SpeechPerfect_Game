# backend.py
import time
import os
import sys

import cv2
from flask import Flask, jsonify, Response, request
from flask_cors import CORS
from gaze_tracker import GazeTracker

# Add module directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from realtime_data_coll import RealTimeDataCollector

app = Flask(__name__)
CORS(app)

focus_state = {"focused": False}
tracker = GazeTracker(show_window=False)
realtime_collector = None  # Global collector instance


def _update_focus_state(focused: bool):
    focus_state["focused"] = focused


@app.before_request
def _ensure_tracker_running():
    # Don't start default tracker if realtime collector is active
    if realtime_collector and realtime_collector.gaze_tracker:
        return  # Use realtime collector's tracker instead
    
    # Only start default tracker if realtime collector is not active
    if not tracker.is_running():
        tracker.start(callback=_update_focus_state)

@app.route("/focus", methods=["GET"])
def get_focus_state():
    return jsonify(focus_state)


def _generate_video_stream():
    """Generate video stream from realtime collector or default tracker."""
    while True:
        # Try to get frame from realtime collector first
        frame = None
        if realtime_collector and realtime_collector.gaze_tracker:
            frame = realtime_collector.gaze_tracker.get_latest_frame()
        
        # Fallback to default tracker (only if realtime collector not active)
        if frame is None and not (realtime_collector and realtime_collector.gaze_tracker):
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
    # Don't start default tracker if realtime collector is active
    if realtime_collector and realtime_collector.gaze_tracker:
        # Use realtime collector's tracker
        pass
    elif not tracker.is_running():
        tracker.start(callback=_update_focus_state)
    
    return Response(
        _generate_video_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# Real-time monitor API endpoints
@app.route("/api/realtime/start", methods=["POST"])
def start_realtime_collection():
    """Start real-time data collection."""
    global realtime_collector, tracker
    
    try:
        data = request.get_json() or {}
        duration_minutes = data.get("duration_minutes")
        serial_port = data.get("serial_port", "COM7")
        camera_index = data.get("camera_index", 0)
        show_camera = data.get("show_camera", False)
        
        # Stop default tracker to free camera
        if tracker.is_running():
            print("🛑 Stopping default tracker to free camera...")
            tracker.stop()
        
        # Stop existing collector if running
        if realtime_collector:
            try:
                realtime_collector.stop_collection()
            except:
                pass
        
        # Create new collector
        realtime_collector = RealTimeDataCollector(
            camera_index=camera_index,
            show_camera=show_camera,
            serial_port=serial_port
        )
        
        # Start collection
        realtime_collector.start_collection()
        
        return jsonify({
            "status": "started",
            "message": "Data collection started successfully"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@app.route("/api/realtime/stop", methods=["POST"])
def stop_realtime_collection():
    """Stop real-time data collection."""
    global realtime_collector, tracker
    
    try:
        if realtime_collector:
            realtime_collector.stop_collection()
            # Restart default tracker after stopping realtime collector
            if not tracker.is_running():
                tracker.start(callback=_update_focus_state)
            return jsonify({
                "status": "stopped",
                "message": "Data collection stopped successfully"
            })
        else:
            return jsonify({
                "status": "error",
                "error": "No active collection"
            }), 400
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@app.route("/api/realtime/stats", methods=["GET"])
def get_realtime_stats():
    """Get current real-time statistics."""
    global realtime_collector
    
    try:
        if realtime_collector:
            stats = realtime_collector.get_all_stats()
            return jsonify(stats)
        else:
            return jsonify({
                "collection_duration_seconds": 0,
                "collection_duration_minutes": 0,
                "eye_tracker": {
                    "not_focus_count": 0,
                    "focus_count": 0,
                    "blink_count": 0,
                    "gaze_left_count": 0,
                    "gaze_right_count": 0,
                    "gaze_center_count": 0,
                    "gaze_unknown_count": 0,
                    "movement_count": 0,
                    "movement_per_minute": 0,
                    "total_samples": 0,
                    "focus_percentage": 0,
                    "not_focus_percentage": 0,
                    "not_focus_per_minute": 0,
                    "total_not_focus_time_seconds": 0,
                    "duration_seconds": 0,
                    "duration_minutes": 0
                },
                "heart_rate": {
                    "avg_bpm": None,
                    "min_bpm": None,
                    "max_bpm": None,
                    "total_readings": 0,
                    "latest_bpm": None
                }
            })
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


@app.route("/api/realtime/heart-rate", methods=["GET"])
def get_heart_rate_readings():
    """Get all heart rate readings."""
    global realtime_collector
    
    try:
        if realtime_collector and realtime_collector.hr_reader:
            # Get all readings from buffer
            with realtime_collector.hr_reader.lock:
                readings = list(realtime_collector.hr_reader.bpm_buffer)
            
            return jsonify({
                "readings": readings,
                "total": len(readings)
            })
        else:
            return jsonify({
                "readings": [],
                "total": 0
            })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "readings": [],
            "total": 0
        }), 500


if __name__ == "__main__":
    # Don't auto-start tracker - let it start on first request or when realtime collection starts
    # tracker.start(callback=_update_focus_state)  # Commented out to prevent camera conflict
    app.run(host="0.0.0.0", port=5000, debug=True)