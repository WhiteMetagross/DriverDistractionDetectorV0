from PySide6.QtCore import QThread, Signal, QMutex, QWaitCondition
from PySide6.QtGui import QImage
import cv2
import time
from typing import Optional
from PIL import Image
import numpy as np

from driversafety.detection.face_detector import DriverFaceDetector
from driversafety.classification.feature_extractor import ResNetFeatureExtractor
from driversafety.classification.behavior_classifier import BehaviorClassifier
from driversafety.visualization.gradcam import compute_gradcam
from driversafety.visualization.edges import get_edge_for_visualization
from driversafety.visualization.overlays import overlay_cam_on_image


class VideoFileWorker(QThread):
    frameReady = Signal(QImage)
    heatmapReady = Signal(QImage)
    statusText = Signal(str)
    classificationChanged = Signal(int, str, float)
    fpsUpdated = Signal(float)
    positionChanged = Signal(int, int)  # current_frame, total_frames
    timeChanged = Signal(float, float)  # current_sec, total_sec
    error = Signal(str)

    def __init__(self, path: str, detector: DriverFaceDetector, extractor: ResNetFeatureExtractor, classifier: BehaviorClassifier, resize_width: int = 960, parent=None):
        super().__init__(parent)
        self.path = path
        self.detector = detector
        self.extractor = extractor
        self.classifier = classifier
        self.resize_width = resize_width

        self._playing = True
        self._stop = False
        self._speed = 1.0
        self._mutex = QMutex()
        self._cond = QWaitCondition()

        self.cap = None
        self.total_frames = 0
        self.fps = 30.0
        self.cur_frame = 0

    def run(self):
        # Try multiple backends in case of codec quirks (e.g., H.264 on Windows)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            for api in (cv2.CAP_FFMPEG, cv2.CAP_MSMF):
                try:
                    c = cv2.VideoCapture(self.path, api)
                    if c.isOpened():
                        self.cap = c
                        break
                except Exception:
                    pass
        if not self.cap or not self.cap.isOpened():
            self.error.emit("Could not open video file")
            return
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        total_sec = self.total_frames / self.fps if self.fps > 0 else 0
        last_fps_t = time.time(); frames = 0

        while not self._stop:
            self._mutex.lock()
            if not self._playing:
                self._cond.wait(self._mutex, 100)
                self._mutex.unlock()
                continue
            self._mutex.unlock()

            ok, frame = self.cap.read()
            if not ok:
                self._playing = False
                self.positionChanged.emit(self.total_frames, self.total_frames)
                break

            self.cur_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

            # Resize
            h, w = frame.shape[:2]
            if w > self.resize_width:
                scale = self.resize_width / w
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

            # Detection
            boxes = self.detector.detect(frame)
            label_text = "No detection"
            heat_qimg: Optional[QImage] = None
            idx = -1; confp = -1.0

            if boxes:
                x1, y1, x2, y2, conf, _ = boxes[0]
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                crop = frame[y1:y2, x1:x2]
                pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                feat, input_tensor = self.extractor.extract(pil)
                idx, label, confp_ = self.classifier.predict(feat)
                confp = confp_ if confp_ is not None else -1.0
                label_text = f"{label} ({confp:.2f})" if confp >= 0 else label

                # Draw box and label (matching original order)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

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

                    ov_h, ov_w, _ = overlay_rgb.shape
                    heat_qimg = QImage(overlay_rgb.data, ov_w, ov_h, ov_w * 3, QImage.Format_RGB888)
                except Exception:
                    pass

            # Emit frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, w * ch, QImage.Format_RGB888)
            self.frameReady.emit(qimg.copy())
            if heat_qimg is not None:
                self.heatmapReady.emit(heat_qimg.copy())
            self.statusText.emit(label_text)
            self.classificationChanged.emit(int(idx), label_text.split(' (')[0], float(confp))

            # Position/time updates
            self.positionChanged.emit(self.cur_frame, self.total_frames)
            cur_sec = self.cur_frame / self.fps if self.fps > 0 else 0
            self.timeChanged.emit(cur_sec, total_sec)

            # FPS
            frames += 1
            now = time.time()
            if now - last_fps_t >= 1.0:
                self.fpsUpdated.emit(frames / (now - last_fps_t))
                frames = 0
                last_fps_t = now

            # Frame pacing for playback
            delay = max(0.0, (1.0 / self.fps) / self._speed)
            self.msleep(int(delay * 1000))

        if self.cap:
            self.cap.release()

    # Controls
    def play(self):
        self._mutex.lock(); self._playing = True; self._cond.wakeAll(); self._mutex.unlock()

    def pause(self):
        self._mutex.lock(); self._playing = False; self._mutex.unlock()

    def stop(self):
        self._mutex.lock(); self._stop = True; self._playing = False; self._cond.wakeAll(); self._mutex.unlock()

    def set_speed(self, speed: float):
        self._mutex.lock(); self._speed = max(0.1, min(speed, 4.0)); self._mutex.unlock()

    def seek_to_frame(self, frame_index: int):
        if self.cap is None: return
        frame_index = max(0, min(frame_index, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        self.cur_frame = frame_index

    def step_next(self):
        self.pause()
        self.seek_to_frame(self.cur_frame + 1)

    def step_prev(self):
        self.pause()
        self.seek_to_frame(self.cur_frame - 1)

