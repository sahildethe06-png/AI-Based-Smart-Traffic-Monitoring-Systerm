import os, time, cv2

class Camera:
    """Webcam / RTSP wrapper that never raises on disconnects and auto-reconnects."""
    def __init__(self, source="0", retry_s=3.0):
        self.source, self.retry_s = source, retry_s
        self.cap, self.status, self.last_try = None, "disconnected", 0.0
        self.writer, self.rec_path = None, None

    def open(self):
        try:
            src = int(self.source) if str(self.source).isdigit() else self.source
            self.cap = cv2.VideoCapture(src)
            ok = self.cap.isOpened()
            self.status = "connected" if ok else "unavailable"
            return ok
        except Exception:
            self.cap, self.status = None, "error"
            return False

    def read(self):
        try:
            if self.cap is None or not self.cap.isOpened():
                if time.time() - self.last_try > self.retry_s:
                    self.last_try = time.time()
                    self.status = "reconnecting"
                    self.open()
                return None
            ok, frame = self.cap.read()
            if not ok:
                self.cap.release(); self.cap = None; self.status = "reconnecting"
                return None
            return frame
        except Exception:
            self.cap, self.status = None, "error"
            return None

    def start_recording(self, frame, folder="recordings", fps=20):
        os.makedirs(folder, exist_ok=True)
        h, w = frame.shape[:2]
        self.rec_path = f"{folder}/rec_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        self.writer = cv2.VideoWriter(self.rec_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    def write(self, frame):
        if self.writer is not None:
            self.writer.write(frame)

    def stop_recording(self):
        if self.writer is not None:
            self.writer.release()
        self.writer = None

    def release(self):
        self.stop_recording()
        if self.cap is not None:
            self.cap.release()
        self.cap, self.status = None, "disconnected"
