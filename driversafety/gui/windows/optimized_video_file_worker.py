from PySide6.QtCore import QThread, Signal, QMutex, QWaitCondition
from PySide6.QtGui import QImage
import cv2
import time
from typing import Optional
from PIL import Image
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty
import torch

from driversafety.detection.optimized_face_detector import OptimizedDriverFaceDetector
from driversafety.classification.optimized_feature_extractor import OptimizedResNetFeatureExtractor
from driversafety.classification.behavior_classifier import BehaviorClassifier
from driversafety.visualization.optimized_gradcam import OptimizedGradCAM
from driversafety.visualization.edges import get_edge_for_visualization
from driversafety.visualization.overlays import overlay_cam_on_image


class OptimizedVideoFileWorker(QThread):
    """High-performance video file processing worker with parallel pipeline."""
    
    frameReady = Signal(QImage)
    heatmapReady = Signal(QImage)
    statusText = Signal(str)
    classificationChanged = Signal(int, str, float)
    fpsUpdated = Signal(float)
    positionChanged = Signal(int, int)  # current_frame, total_frames
    timeChanged = Signal(float, float)  # current_sec, total_sec
    error = Signal(str)
    performanceStats = Signal(dict)  # Performance monitoring

    def __init__(self, path: str, detector: OptimizedDriverFaceDetector,
                 extractor: OptimizedResNetFeatureExtractor, classifier: BehaviorClassifier,
                 resize_width: int = 960, max_workers: int = 19, target_fps: float = 25.0,
                 enable_frame_skipping: bool = True, parent=None):
        super().__init__(parent)
        print(f"[VideoFileWorker] __init__ path={path}, resize_width={resize_width}, max_workers={max_workers}, target_fps={target_fps}, skip={enable_frame_skipping}")
        self.path = path
        self.detector = detector
        self.extractor = extractor
        self.classifier = classifier
        self.resize_width = resize_width
        self.max_workers = max_workers
        self.target_fps = target_fps
        self.enable_frame_skipping = enable_frame_skipping

        self._playing = True
        self._stop = False
        self._speed = 1.0
        self._mutex = QMutex()
        self._cond = QWaitCondition()
        # Mutex to guard VideoCapture operations (seek/read)
        self._cap_mutex = QMutex()

        self.cap = None
        self.total_frames = 0
        self.fps = 30.0
        self.cur_frame = 0
        # Stepping/seeking state
        self.pending_seek_frame = None
        self._single_step = False

        # Performance optimization
        self.frame_queue = Queue(maxsize=10)
        self.result_queue = Queue(maxsize=10)
        
        # Thread pools for parallel processing
        self.detection_pool = ThreadPoolExecutor(max_workers=max(1, max_workers // 4))
        self.classification_pool = ThreadPoolExecutor(max_workers=max(1, max_workers // 2))
        self.gradcam_pool = ThreadPoolExecutor(max_workers=max(1, max_workers // 4))
        
        # Performance tracking
        self.processing_times = []
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.performance_lock = threading.Lock()
        
        # Frame skipping logic
        self.last_processed_time = 0
        self.frame_interval = 1.0 / target_fps if target_fps > 0 else 0
        
        print(f"OptimizedVideoFileWorker initialized:")
        print(f"  - Video: {path}")
        print(f"  - Max workers: {max_workers}")
        print(f"  - Target FPS: {target_fps}")
        print(f"  - Frame skipping: {enable_frame_skipping}")

        # Initialize single Grad-CAM instance for this worker (avoid per-frame init)
        try:
            device_str = str(self.extractor.device) if hasattr(self.extractor, 'device') else ('cuda' if torch.cuda.is_available() else 'cpu')
            enable_fp16 = bool(getattr(self.extractor, 'enable_fp16', False))
            self.gradcam = OptimizedGradCAM(self.extractor.model, [self.extractor.model.layer4[-1]], device=device_str, enable_fp16=enable_fp16)
        except Exception as e:
            print(f"[VideoFileWorker] Failed to initialize Grad-CAM instance: {e}")
            self.gradcam = None

        # Temporal smoothing buffers
        from collections import deque
        self._bbox_history = deque(maxlen=5)  # (x1,y1,x2,y2)
        self._class_history = deque(maxlen=7)  # (idx, confidence)

    def _smooth_bbox(self, bbox):
        """Simple moving average smoothing over recent bounding boxes."""
        x1, y1, x2, y2 = bbox
        self._bbox_history.append((x1, y1, x2, y2))
        arr = np.array(self._bbox_history, dtype=np.float32)
        avg = arr.mean(axis=0)
        return tuple(int(v) for v in avg)

    def _smooth_classification(self, idx: int, confidence: float):
        """Confidence-weighted majority smoothing of class index.
        If confidence is None, treat as 1.0.
        Returns (smoothed_idx, smoothed_confidence_estimate).
        """
        conf = float(confidence) if confidence is not None else 1.0
        self._class_history.append((idx, conf))
        # Aggregate scores per class
        scores = {}
        counts = {}
        for c, w in self._class_history:
            scores[c] = scores.get(c, 0.0) + (w if w is not None else 1.0)
            counts[c] = counts.get(c, 0) + 1
        best = max(scores.items(), key=lambda kv: kv[1])[0]
        avg_conf = scores[best] / max(counts[best], 1)
        return int(best), float(avg_conf)


    def stop(self):
        """Stop the worker and clean up resources."""
        self._stop = True
        self._cond.wakeAll()
        
        # Clean up thread pools
        self.detection_pool.shutdown(wait=False)
        self.classification_pool.shutdown(wait=False)
        self.gradcam_pool.shutdown(wait=False)

    def pause(self):
        """Pause video playback."""
        self._mutex.lock()
        self._playing = False
        self._mutex.unlock()

    def play(self):
        """Start/resume video playback (alias to resume)."""
        self.resume()

    def resume(self):
        """Resume video playback."""
        self._mutex.lock()
        self._playing = True
        self._cond.wakeAll()
        self._mutex.unlock()

    def step_next(self):
        """Advance to next frame (single-step)."""
        self._mutex.lock()
        target = self.cur_frame + 1
        if self.total_frames > 0:
            target = min(target, self.total_frames - 1)
        self.pending_seek_frame = target
        self._single_step = True
        self._playing = True
        self._cond.wakeAll()
        self._mutex.unlock()

    def step_prev(self):
        """Go back to previous frame (single-step)."""
        self._mutex.lock()
        target = max(self.cur_frame - 1, 0)
        self.pending_seek_frame = target
        self._single_step = True
        self._playing = True
        self._cond.wakeAll()
        self._mutex.unlock()

    def set_speed(self, speed: float):
        """Set playback speed."""
        self._mutex.lock()
        self._speed = speed
        self._mutex.unlock()

    def seek_to_frame(self, frame_num: int):
        """Request seek to specific frame. Actual seek is performed in worker thread."""
        self._mutex.lock()
        if 0 <= frame_num < (self.total_frames or 0):
            self.pending_seek_frame = int(frame_num)
            self._playing = True  # ensure processing continues to apply seek
            self._cond.wakeAll()
        self._mutex.unlock()

    def restart(self):
        """Restart video from the beginning."""
        self.seek_to_frame(0)

    def _should_process_frame(self) -> bool:
        """Determine if current frame should be processed based on target FPS."""
        if not self.enable_frame_skipping:
            return True
        
        current_time = time.time()
        if current_time - self.last_processed_time >= self.frame_interval:
            self.last_processed_time = current_time
            return True
        return False

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Optimized frame resizing."""
        h, w = frame.shape[:2]
        if w > self.resize_width:
            scale = self.resize_width / w
            new_size = (int(w * scale), int(h * scale))
            return cv2.resize(frame, new_size, interpolation=cv2.INTER_LINEAR)
        return frame

    def _process_frame_parallel(self, frame: np.ndarray, frame_id: int) -> dict:
        """Process a single frame with parallel pipeline."""
        try:
            # Detection
            boxes = self.detector.detect(frame)
            
            if not boxes:
                return {
                    'frame': frame,
                    'frame_id': frame_id,
                    'detection': None,
                    'classification': None,
                    'gradcam': None
                }
            
            # Use first detection
            x1, y1, x2, y2, conf, _ = boxes[0]
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])
            
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return {
                    'frame': frame,
                    'frame_id': frame_id,
                    'detection': boxes[0],
                    'classification': None,
                    'gradcam': None
                }
            
            # Classification
            pil_image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            features, input_tensor = self.extractor.extract(pil_image)
            idx, label, confidence = self.classifier.predict(features)
            
            # Grad-CAM with improved accuracy (no fallbacks)
            try:
                target_layers = [self.extractor.model.layer4[-1]]
                heat = None
                if self.gradcam is not None:
                    heat_np = self.gradcam.compute_heatmap(
                        input_tensor,
                        target_class=idx,
                        bbox=(x1, y1, x2, y2)
                    )
                    # Convert to torch tensor with batch dim to match original usage
                    heat = torch.from_numpy(heat_np).unsqueeze(0)
            except Exception as gradcam_error:
                print(f"Grad-CAM computation failed: {gradcam_error}")
                import traceback
                traceback.print_exc()
                heat = None

            if heat is not None:
                heat0 = heat[0]  # HxW in [0,1]
                # Edge visualization
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                edge_img = get_edge_for_visualization(Image.fromarray(frame_rgb))
                overlay_rgb = overlay_cam_on_image(edge_img, heat0.numpy())

                # Convert to QImage
                ov_h, ov_w, _ = overlay_rgb.shape
                heat_qimg = QImage(overlay_rgb.data, ov_w, ov_h, ov_w * 3, QImage.Format_RGB888)
            else:
                heat_qimg = None

            return {
                'frame': frame,
                'frame_id': frame_id,
                'detection': boxes[0],
                'classification': {
                    'bbox': (x1, y1, x2, y2),
                    'idx': idx,
                    'label': label,
                    'confidence': confidence
                },
                'gradcam': heat_qimg
            }
            
        except Exception as e:
            print(f"Frame processing error: {e}")
            return {
                'frame': frame,
                'frame_id': frame_id,
                'detection': None,
                'classification': None,
                'gradcam': None
            }

    def run(self):
        """Main processing loop with optimized parallel pipeline."""
        print(f"Opening video file: {self.path}")

        # Open video with multiple backend fallbacks
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            print("Failed to open with default backend, trying alternatives...")
            # Prefer MSMF on Windows to avoid FFmpeg threading assertion during seeks
            for api in (cv2.CAP_MSMF, cv2.CAP_FFMPEG):
                c = cv2.VideoCapture(self.path, api)
                if c.isOpened():
                    self.cap = c
                    print(f"Successfully opened with backend: {api}")
                    break
            else:
                error_msg = f"Failed to open video: {self.path}"
                print(error_msg)
                self.error.emit(error_msg)
                return

        if not self.cap.isOpened():
            error_msg = f"Video file could not be opened: {self.path}"
            print(error_msg)
            self.error.emit(error_msg)
            return

        print(f"Video opened successfully: {self.path}")

        # Get video properties
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_sec = self.total_frames / self.fps if self.fps else 0
        print(f"[VideoFileWorker] Opened: frames={self.total_frames}, fps={self.fps}, duration={total_sec:.2f}s")
        self.statusText.emit(f"Opened video: {self.total_frames} frames @ {self.fps:.1f} FPS")

        processing_futures = []

        while not self._stop:
            start_time = time.time()
            
            # Handle pause/resume
            self._mutex.lock()
            if not self._playing:
                self._cond.wait(self._mutex)
            self._mutex.unlock()
            
            if self._stop:
                break

            # Handle pending seek requests
            self._mutex.lock()
            pending = self.pending_seek_frame
            self.pending_seek_frame = None
            self._mutex.unlock()
            if pending is not None and self.cap:
                try:
                    self._cap_mutex.lock()
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(pending))
                finally:
                    self._cap_mutex.unlock()

            # Read frame
            self._cap_mutex.lock()
            ok, frame = self.cap.read()
            self._cap_mutex.unlock()
            if not ok:
                # End of video reached - pause and emit end signal
                self._mutex.lock()
                self._playing = False
                self._mutex.unlock()

                # Emit end-of-video position
                self.positionChanged.emit(self.total_frames, self.total_frames)
                self.timeChanged.emit(total_sec, total_sec)

                # Wait for user action (play/seek) instead of breaking
                self._mutex.lock()
                if not self._stop:
                    self._cond.wait(self._mutex)
                self._mutex.unlock()

                if self._stop:
                    break

                # If we reach here, user has requested play/seek - continue
                continue

            # Resize frame for performance
            frame = self._resize_frame(frame)
            
            # Update position
            self.cur_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            current_sec = self.cur_frame / self.fps
            self.positionChanged.emit(self.cur_frame, self.total_frames)
            self.timeChanged.emit(current_sec, total_sec)

            # Frame skipping logic
            if not self._should_process_frame():
                continue

            # Submit parallel processing
            if self._stop:
                break
            if len(processing_futures) < self.max_workers:
                try:
                    future = self.classification_pool.submit(
                        self._process_frame_parallel, frame.copy(), self.cur_frame
                    )
                    processing_futures.append(future)
                except RuntimeError as e:
                    # ThreadPool might be shutting down during window close; exit loop cleanly
                    print(f"Submission error (shutting down?): {e}")
                    break

            # Process completed futures
            completed_futures = []
            for future in processing_futures:
                if future.done():
                    try:
                        result = future.result()
                        self._emit_results(result)
                        completed_futures.append(future)
                    except Exception as e:
                        print(f"Processing error: {e}")
                        completed_futures.append(future)

            # Remove completed futures
            for future in completed_futures:
                processing_futures.remove(future)

            # Update performance stats
            self._update_performance_stats(start_time)

            # If single-step, pause after processing one frame
            if self._single_step:
                self._mutex.lock()
                self._playing = False
                self._single_step = False
                self._mutex.unlock()

            # Control playback speed
            if self._speed > 0 and self._playing:
                time.sleep((1.0 / self.fps) / self._speed)

        # Wait for remaining futures
        for future in processing_futures:
            try:
                future.result(timeout=1.0)
            except:
                pass

        if self.cap:
            self._cap_mutex.lock()
            try:
                self.cap.release()
            finally:
                self._cap_mutex.unlock()

    def _emit_results(self, result: dict):
        """Emit processing results to GUI."""
        frame = result['frame']
        label_text = "No detection"
        idx = -1
        confidence = -1.0

        # Draw detection and classification
        if result['detection'] and result['classification']:
            x1, y1, x2, y2, conf, _ = result['detection']
            classification = result['classification']
            idx = classification['idx']
            label = classification['label']
            confidence = classification['confidence'] if classification['confidence'] is not None else -1.0

            # Temporal smoothing
            x1, y1, x2, y2 = self._smooth_bbox((x1, y1, x2, y2))
            idx, smoothed_conf = self._smooth_classification(idx, confidence)
            label = self.classifier.label_map.get(idx, label)
            confidence = smoothed_conf if smoothed_conf is not None else confidence

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_text = f"{label} ({confidence:.2f})" if confidence is not None and confidence >= 0 else label
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Emit main frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, w * ch, QImage.Format_RGB888)
        self.frameReady.emit(qimg.copy())

        # Emit heatmap only if available (no placeholders)
        if result['gradcam'] is not None:
            self.heatmapReady.emit(result['gradcam'].copy())

        # Emit status and classification
        self.statusText.emit(label_text)
        self.classificationChanged.emit(int(idx), label_text.split(' (')[0], float(confidence))

    def _update_performance_stats(self, start_time: float):
        """Update and emit performance statistics."""
        processing_time = time.time() - start_time
        
        with self.performance_lock:
            self.processing_times.append(processing_time)
            if len(self.processing_times) > 100:
                self.processing_times.pop(0)
            
            self.fps_counter += 1
            current_time = time.time()
            
            if current_time - self.last_fps_time >= 1.0:
                fps = self.fps_counter / (current_time - self.last_fps_time)
                self.fpsUpdated.emit(fps)
                
                # Emit detailed performance stats
                stats = {
                    'fps': fps,
                    'avg_processing_time': np.mean(self.processing_times),
                    'detector_stats': self.detector.get_performance_stats(),
                    'extractor_stats': self.extractor.get_performance_stats(),
                }
                self.performanceStats.emit(stats)
                
                self.fps_counter = 0
                self.last_fps_time = current_time
