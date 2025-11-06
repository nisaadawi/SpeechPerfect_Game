import threading
import cv2
from gaze_tracking import GazeTracking


def _compute_focus(gaze: GazeTracking) -> bool:
    blink = gaze.is_blinking()
    if blink is True:
        return False

    center = gaze.is_center()
    return bool(center)


class GazeTracker:
    """Continuously estimate gaze focus state in a background thread."""

    def __init__(self, camera_index=0, show_window=False, window_name="Gaze Tracker"):
        self.camera_index = camera_index
        self.show_window = show_window
        self.window_name = window_name
        self.focused = False
        self._stop_event = threading.Event()
        self._thread = None
        self._callback = None
        self._gaze = GazeTracking()
        self._frame_lock = threading.Lock()
        self._latest_frame = None

    def start(self, callback=None):
        if self.is_running():
            return
        self._callback = callback
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        if not self.is_running():
            return
        self._stop_event.set()
        self._thread.join()
        self._thread = None

    def is_running(self):
        return self._thread is not None and self._thread.is_alive()

    def _loop(self):
        cap = cv2.VideoCapture(self.camera_index)

        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break

                self._gaze.refresh(frame)
                focused = _compute_focus(self._gaze)
                self.focused = focused

                annotated = self._gaze.annotated_frame()
                if annotated is None:
                    annotated = frame.copy()

                cv2.putText(
                    annotated,
                    f"Focused: {focused}",
                    (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0) if focused else (0, 0, 255),
                    2,
                )

                with self._frame_lock:
                    self._latest_frame = annotated.copy()

                if self._callback:
                    self._callback(focused)

                if self.show_window:
                    cv2.imshow(self.window_name, annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        self._stop_event.set()

        finally:
            cap.release()
            if self.show_window:
                cv2.destroyWindow(self.window_name)


    def get_latest_frame(self):
        with self._frame_lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()


def check_focus():
    gaze = GazeTracking()
    cap = cv2.VideoCapture(0)
    focused = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gaze.refresh(frame)
        focused = _compute_focus(gaze)
        annotated = gaze.annotated_frame()
        if annotated is None:
            annotated = frame.copy()
        cv2.putText(
            annotated,
            f"Focused: {focused}",
            (30, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0) if focused else (0, 0, 255),
            2,
        )
        cv2.imshow("Gaze Tracker", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    return focused
