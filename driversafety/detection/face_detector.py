from typing import List, Tuple
import numpy as np
from ultralytics import YOLO


class DriverFaceDetector:
    """YOLO-based detector to localize driver/person/face regions.

    Returns bounding boxes in (x1, y1, x2, y2, score, cls) format.
    """

    def __init__(self, model_path: str, conf_threshold: float = 0.25):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int, float, int]]:
        results = self.model(frame_bgr, verbose=False)
        boxes_out: List[Tuple[int, int, int, int, float, int]] = []
        if not results:
            return boxes_out
        res = results[0]
        if res.boxes is None or res.boxes.xyxy is None:
            return boxes_out
        for i, box in enumerate(res.boxes):
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, xyxy)
            conf = float(box.conf[0].item()) if hasattr(box, "conf") else 0.0
            cls = int(box.cls[0].item()) if hasattr(box, "cls") else -1
            if conf >= self.conf_threshold:
                boxes_out.append((x1, y1, x2, y2, conf, cls))
        return boxes_out

