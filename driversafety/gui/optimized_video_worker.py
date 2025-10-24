from PySide6.QtCore import QThread, Signal, QMutex, QWaitCondition
from PySide6.QtGui import QImage
import cv2
import numpy as np
from typing import Optional, List, Tuple
from PIL import Image
import time
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


class FrameData:
    """Container for frame processing data."""
    def __init__(self, frame: np.ndarray, frame_id: int, timestamp: float):
        self.frame = frame
        self.frame_id = frame_id
        self.timestamp = timestamp
        self.detection_result = None
        self.classification_result = None
        self.gradcam_result = None


class OptimizedVideoWorker(QThread):
    """High-performance video processing worker with parallel pipeline and GPU acceleration."""
    
    frameReady = Signal(QImage)
    heatmapReady = Signal(QImage)
    statusText = Signal(str)
    classificationChanged = Signal(int, str, float)
    fpsUpdated = Signal(float)
    performanceStats = Signal(dict)  # New signal for performance monitoring

    def __init__(self, source: str, detector: OptimizedDriverFaceDetector, 
                 extractor: OptimizedResNetFeatureExtractor, classifier: BehaviorClassifier,
                 resize_width: int = 960, max_workers: int = 19, target_fps: float = 25.0,
                 enable_frame_skipping: bool = True, parent=None):
        super().__init__(parent)
        
        self.source = source
        self.detector = detector
        self.extractor = extractor
        self.classifier = classifier
        self.resize_width = resize_width
        self.max_workers = max_workers
        self.target_fps = target_fps
        self.enable_frame_skipping = enable_frame_skipping
        
        # Threading and synchronization
        self._running = False
        self.frame_queue = Queue(maxsize=10)
        self.result_queue = Queue(maxsize=10)
        
        # Thread pools for parallel processing
        self.detection_pool = ThreadPoolExecutor(max_workers=max(1, max_workers // 4))
        self.classification_pool = ThreadPoolExecutor(max_workers=max(1, max_workers // 2))
        self.gradcam_pool = ThreadPoolExecutor(max_workers=max(1, max_workers // 4))
        
        # Performance tracking
        self.frame_times = []
        self.processing_times = []
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.performance_lock = threading.Lock()
        
        # Frame skipping logic
        self.last_processed_time = 0
        self.frame_interval = 1.0 / target_fps if target_fps > 0 else 0
        
        print(f"OptimizedVideoWorker initialized:")
        print(f"  - Max workers: {max_workers}")
        print(f"  - Target FPS: {target_fps}")
        print(f"  - Frame skipping: {enable_frame_skipping}")
        print(f"  - GPU acceleration: {detector.device}")

        # Initialize single Grad-CAM instance for this worker
        try:
            device_str = str(self.extractor.device) if hasattr(self.extractor, 'device') else ('cuda' if torch.cuda.is_available() else 'cpu')
            enable_fp16 = bool(getattr(self.extractor, 'enable_fp16', False))
            self.gradcam = OptimizedGradCAM(self.extractor.model, [self.extractor.model.layer4[-1]], device=device_str, enable_fp16=enable_fp16)
        except Exception as e:
            print(f"[OptimizedVideoWorker] Failed to initialize Grad-CAM: {e}")
            self.gradcam = None

        # Temporal smoothing buffers
        from collections import deque
        self._bbox_history = deque(maxlen=5)  # (x1,y1,x2,y2)
        self._class_history = deque(maxlen=7)  # (idx, confidence)

    def _smooth_bbox(self, bbox):
        x1, y1, x2, y2 = bbox
        self._bbox_history.append((x1, y1, x2, y2))
        arr = np.array(self._bbox_history, dtype=np.float32)
        avg = arr.mean(axis=0)
        return tuple(int(v) for v in avg)

    def _smooth_classification(self, idx: int, confidence: float):
        conf = float(confidence) if confidence is not None else 1.0
        self._class_history.append((idx, conf))
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
        self._running = False
        
        # Clean up thread pools
        self.detection_pool.shutdown(wait=False)
        self.classification_pool.shutdown(wait=False)
        self.gradcam_pool.shutdown(wait=False)
        
        # Clean up models
        self.detector.cleanup()
        self.extractor.cleanup()

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

    def _process_detection(self, frame_data: FrameData) -> FrameData:
        """Process detection in parallel."""
        try:
            boxes = self.detector.detect(frame_data.frame)
            frame_data.detection_result = boxes
        except Exception as e:
            print(f"Detection error: {e}")
            frame_data.detection_result = []
        return frame_data

    def _process_classification(self, frame_data: FrameData) -> FrameData:
        """Process classification in parallel."""
        if not frame_data.detection_result:
            return frame_data
        
        try:
            # Use first detection
            x1, y1, x2, y2, conf, _ = frame_data.detection_result[0]
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, frame_data.frame.shape[1]), min(y2, frame_data.frame.shape[0])
            
            crop = frame_data.frame[y1:y2, x1:x2]
            if crop.size > 0:
                pil_image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                features, input_tensor = self.extractor.extract(pil_image)
                idx, label, confidence = self.classifier.predict(features)
                
                frame_data.classification_result = {
                    'bbox': (x1, y1, x2, y2),
                    'idx': idx,
                    'label': label,
                    'confidence': confidence,
                    'input_tensor': input_tensor
                }
        except Exception as e:
            print(f"Classification error: {e}")
            frame_data.classification_result = None
        
        return frame_data

    def _process_gradcam(self, frame_data: FrameData) -> FrameData:
        """Process Grad-CAM in parallel with improved accuracy."""
        if not frame_data.classification_result:
            return frame_data
        
        try:
            result = frame_data.classification_result

            # Reuse single Grad-CAM instance with coordinate mapping
            if self.gradcam is None:
                return frame_data
            heat_np = self.gradcam.compute_heatmap(
                result['input_tensor'],
                target_class=result['idx'],
                bbox=result['bbox']
            )
            heat0 = torch.from_numpy(heat_np)  # HxW in [0,1]

            # Edge visualization with proper frame
            frame_rgb = cv2.cvtColor(frame_data.frame, cv2.COLOR_BGR2RGB)
            edge_img = get_edge_for_visualization(Image.fromarray(frame_rgb))
            overlay_rgb = overlay_cam_on_image(edge_img, heat0.numpy())
            
            # Convert to QImage
            ov_h, ov_w, _ = overlay_rgb.shape
            heat_qimg = QImage(overlay_rgb.data, ov_w, ov_h, ov_w * 3, QImage.Format_RGB888)
            
            frame_data.gradcam_result = heat_qimg
            
        except Exception as e:
            print(f"Grad-CAM error: {e}")
            import traceback
            traceback.print_exc()
            frame_data.gradcam_result = None
        
        return frame_data

    def run(self):
        """Main processing loop with parallel pipeline."""
        self._running = True
        
        # Open video source
        cap = cv2.VideoCapture(0) if self.source == "webcam" else cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.statusText.emit("Failed to open video source")
            return

        frame_id = 0
        processing_futures = []
        
        while self._running:
            start_time = time.time()
            
            # Read frame
            ok, frame = cap.read()
            if not ok:
                break

            # Resize frame for performance
            frame = self._resize_frame(frame)
            
            # Frame skipping logic
            if not self._should_process_frame():
                continue

            # Create frame data
            frame_data = FrameData(frame.copy(), frame_id, start_time)
            
            # Submit parallel processing tasks
            if len(processing_futures) < self.max_workers:
                # Detection
                det_future = self.detection_pool.submit(self._process_detection, frame_data)
                
                # Chain classification and Grad-CAM
                def process_chain(det_fut):
                    frame_data = det_fut.result()
                    frame_data = self._process_classification(frame_data)
                    frame_data = self._process_gradcam(frame_data)
                    return frame_data
                
                chain_future = self.classification_pool.submit(process_chain, det_future)
                processing_futures.append(chain_future)
            
            # Process completed futures
            completed_futures = []
            for future in processing_futures:
                if future.done():
                    try:
                        processed_frame = future.result()
                        self._emit_results(processed_frame)
                        completed_futures.append(future)
                    except Exception as e:
                        print(f"Processing error: {e}")
                        completed_futures.append(future)
            
            # Remove completed futures
            for future in completed_futures:
                processing_futures.remove(future)
            
            # Update performance stats
            self._update_performance_stats(start_time)
            
            frame_id += 1

        # Wait for remaining futures to complete
        for future in processing_futures:
            try:
                future.result(timeout=1.0)
            except:
                pass
        
        cap.release()

    def _emit_results(self, frame_data: FrameData):
        """Emit processing results to GUI."""
        # Draw detection and classification on frame
        frame = frame_data.frame.copy()
        label_text = "No detection"
        idx = -1
        confidence = -1.0
        
        if frame_data.detection_result and frame_data.classification_result:
            result = frame_data.classification_result
            x1, y1, x2, y2 = result['bbox']
            idx = result['idx']
            label = result['label']
            confidence = result['confidence'] if result['confidence'] is not None else -1.0
            
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
        if frame_data.gradcam_result is not None:
            self.heatmapReady.emit(frame_data.gradcam_result.copy())

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
