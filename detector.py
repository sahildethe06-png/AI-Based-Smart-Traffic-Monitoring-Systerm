import os
from ultralytics import YOLO

COCO = {0: "pedestrian", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

class Detector:
    def __init__(self, model=None, tracker="bytetrack.yaml", conf=0.35, helmet_model=None):
        self.model = YOLO(model or os.getenv("YOLO_MODEL", "yolov8n.pt"))
        self.tracker, self.conf = tracker, conf
        hp = helmet_model or os.getenv("HELMET_MODEL")
        self.helmet = YOLO(hp) if hp and os.path.exists(hp) else None  # None => helmet detector disabled

    def track(self, frame):
        r = self.model.track(frame, persist=True, tracker=self.tracker, conf=self.conf,
                             classes=list(COCO), verbose=False)[0]
        out = []
        if r.boxes is None or r.boxes.id is None:
            return out
        for b, i, c, p in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.id.int().cpu().tolist(),
                              r.boxes.cls.int().cpu().tolist(), r.boxes.conf.cpu().tolist()):
            out.append(dict(id=i, cls=COCO[c], conf=float(p), box=tuple(float(v) for v in b)))
        return out

    def helmet_boxes(self, frame):
        """Returns [(box, label, conf)] from a user-trained helmet model (classes like 'helmet'/'no_helmet')."""
        if self.helmet is None:
            return []
        r = self.helmet(frame, conf=0.5, verbose=False)[0]
        names = r.names
        return [(tuple(b), names[int(c)].lower(), float(p)) for b, c, p in
                zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy())]
