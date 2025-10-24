from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage
import cv2
from typing import Optional
from PIL import Image
import numpy as np

from driversafety.detection.face_detector import DriverFaceDetector
from driversafety.classification.feature_extractor import ResNetFeatureExtractor
from driversafety.classification.behavior_classifier import BehaviorClassifier
from driversafety.visualization.gradcam import compute_gradcam
from driversafety.visualization.edges import get_edge_for_visualization
from driversafety.visualization.overlays import overlay_cam_on_image


class VideoWorker(QThread):
    frameReady = Signal(QImage)
    heatmapReady = Signal(QImage)
    statusText = Signal(str)
    classificationChanged = Signal(int, str, float)  # (idx, label, conf or -1)
    fpsUpdated = Signal(float)

    def __init__(self, source: str, detector: DriverFaceDetector, extractor: ResNetFeatureExtractor, classifier: BehaviorClassifier, resize_width: int = 960, parent=None):
        super().__init__(parent)
        self.source = source  # "webcam" or file path
        self.detector = detector
        self.extractor = extractor
        self.classifier = classifier
        self.resize_width = resize_width
        self._running = False

    def stop(self):
        self._running = False

    def run(self):
        import time
        self._running = True
        cap = cv2.VideoCapture(0) if self.source == "webcam" else cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.statusText.emit("Failed to open video source")
            return

        last_time = time.time()
        frames = 0
        while self._running:
            ok, frame = cap.read()
            if not ok:
                break

            # Optional resize for performance
            h, w = frame.shape[:2]
            if w > self.resize_width:
                scale = self.resize_width / w
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

            # Detection
            boxes = self.detector.detect(frame)

            label_text = "No detection"
            heat_qimg: Optional[QImage] = None
            idx = -1
            confp = -1.0

            if boxes:
                # Use top-1 detection for now
                x1, y1, x2, y2, conf, _ = boxes[0]
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                crop = frame[y1:y2, x1:x2]

                # Prepare PIL for extractor
                pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                feat, input_tensor = self.extractor.extract(pil)

                # Classification
                idx, label, confp_ = self.classifier.predict(feat)
                confp = confp_ if confp_ is not None else -1.0
                label_text = f"{label} ({confp:.2f})" if confp >= 0 else label

                # Draw box and label first (matching original order)
                cv2.putText(frame, label_text, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                # Grad-CAM over edges visualization (matching original exactly)
                try:
                    from pytorch_grad_cam import GradCAM
                    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
                    from driversafety.visualization.overlays import overlay_cam_on_image

                    classes = [ClassifierOutputTarget(idx)]
                    target_layer = [self.extractor.model.layer4[-1]]
                    cam = GradCAM(model=self.extractor.model, target_layers=target_layer)

                    heatmap = cam(input_tensor=input_tensor, targets=classes)
                    # Convert frame to RGB for edge processing (matching original)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    edge_image = get_edge_for_visualization(Image.fromarray(frame_rgb))
                    overlay_rgb = overlay_cam_on_image(edge_image, heatmap[0])

                    # Convert overlay to QImage
                    ov_h, ov_w, _ = overlay_rgb.shape
                    heat_qimg = QImage(overlay_rgb.data, ov_w, ov_h, ov_w * 3, QImage.Format_RGB888)
                except Exception:
                    # Skip heatmap if Grad-CAM fails; keep main frame going
                    pass

            # Emit main frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, w * ch, QImage.Format_RGB888)
            self.frameReady.emit(qimg.copy())

            if heat_qimg is not None:
                self.heatmapReady.emit(heat_qimg.copy())

            self.statusText.emit(label_text)
            self.classificationChanged.emit(int(idx), label_text.split(' (')[0], float(confp))

            frames += 1
            now = time.time()
            if now - last_time >= 1.0:
                self.fpsUpdated.emit(frames / (now - last_time))
                frames = 0
                last_time = now

        cap.release()

