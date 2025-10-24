from typing import List, Tuple, Optional
import numpy as np
from ultralytics import YOLO
import torch
import threading
import time
from concurrent.futures import ThreadPoolExecutor


class OptimizedDriverFaceDetector:
    """High-performance YOLO-based detector with GPU acceleration and optimizations."""

    def __init__(self, model_path: str, conf_threshold: float = 0.25, device: str = "auto", 
                 enable_fp16: bool = True, max_workers: int = 2):
        self.conf_threshold = conf_threshold
        self.max_workers = max_workers
        
        # Device selection
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.enable_fp16 = enable_fp16 and self.device == "cuda"
        
        print(f"Initializing OptimizedDriverFaceDetector on {self.device}")
        if self.enable_fp16:
            print("FP16 half-precision enabled for YOLO inference")
        
        # Load YOLO model with optimizations
        self.model = YOLO(model_path)
        
        # Configure model for optimal performance
        if hasattr(self.model.model, 'to'):
            self.model.model.to(self.device)
        
        # Enable half-precision if supported
        if self.enable_fp16:
            try:
                if hasattr(self.model.model, 'half'):
                    self.model.model.half()
                print("✓ YOLO model converted to FP16")
            except Exception as e:
                print(f"FP16 conversion failed: {e}, using FP32")
                self.enable_fp16 = False
        
        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        
        # Performance tracking
        self.detection_times = []
        self.lock = threading.Lock()
        
        # Warmup
        self._warmup()
    
    def _warmup(self):
        """Warmup the model with dummy data."""
        print("Warming up YOLO detector...")
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Run a few warmup detections
        for _ in range(3):
            _ = self.model(dummy_frame, verbose=False, device=self.device, half=self.enable_fp16)
        
        if self.device == "cuda":
            torch.cuda.synchronize()
        print("✓ YOLO detector warmup complete")
    
    def detect_single(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int, float, int]]:
        """Detect faces/drivers in a single frame with optimized performance."""
        start_time = time.time()
        
        # Run YOLO detection with optimizations
        results = self.model(
            frame_bgr, 
            verbose=False, 
            device=self.device,
            half=self.enable_fp16,
            conf=self.conf_threshold
        )
        
        boxes_out: List[Tuple[int, int, int, int, float, int]] = []
        
        if not results:
            return boxes_out
        
        res = results[0]
        if res.boxes is None or res.boxes.xyxy is None:
            return boxes_out
        
        # Process detections
        for i, box in enumerate(res.boxes):
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, xyxy)
            conf = float(box.conf[0].item()) if hasattr(box, "conf") else 0.0
            cls = int(box.cls[0].item()) if hasattr(box, "cls") else -1
            
            if conf >= self.conf_threshold:
                boxes_out.append((x1, y1, x2, y2, conf, cls))
        
        # Synchronize for accurate timing
        if self.device == "cuda":
            torch.cuda.synchronize()
        
        # Track performance
        detection_time = time.time() - start_time
        with self.lock:
            self.detection_times.append(detection_time)
            if len(self.detection_times) > 100:  # Keep last 100 measurements
                self.detection_times.pop(0)
        
        return boxes_out
    
    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Tuple[int, int, int, int, float, int]]]:
        """Detect faces/drivers in multiple frames in parallel."""
        if len(frames) == 1:
            return [self.detect_single(frames[0])]
        
        # Process frames in parallel using thread pool
        futures = [self.thread_pool.submit(self.detect_single, frame) for frame in frames]
        results = [future.result() for future in futures]
        
        return results
    
    def detect(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int, float, int]]:
        """Main detection method - maintains compatibility with original interface."""
        return self.detect_single(frame_bgr)
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics."""
        with self.lock:
            if not self.detection_times:
                return {"avg_time": 0, "fps": 0, "samples": 0}
            
            avg_time = np.mean(self.detection_times)
            fps = 1.0 / avg_time if avg_time > 0 else 0
            
            return {
                "avg_time": avg_time,
                "fps": fps,
                "samples": len(self.detection_times),
                "min_time": np.min(self.detection_times),
                "max_time": np.max(self.detection_times)
            }
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)
        
        # Clear GPU cache
        if self.device == "cuda":
            torch.cuda.empty_cache()


# Backward compatibility alias
DriverFaceDetector = OptimizedDriverFaceDetector
