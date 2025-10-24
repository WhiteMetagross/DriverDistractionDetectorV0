from PySide6.QtCore import QObject, Signal
from typing import Optional, Union
import time

from driversafety.gui.optimized_video_worker import OptimizedVideoWorker
from driversafety.gui.windows.optimized_video_file_worker import OptimizedVideoFileWorker


class UnifiedVideoWorker(QObject):
    """
    Unified video worker that can seamlessly switch between webcam and video file inputs.
    Acts as a wrapper around the existing optimized workers.
    """
    
    # Common signals for both worker types
    frameReady = Signal(object)  # QImage
    heatmapReady = Signal(object)  # QImage
    statusText = Signal(str)
    classificationChanged = Signal(int, str, float)  # idx, label, confidence
    fpsUpdated = Signal(float)
    performanceStats = Signal(dict)
    
    # Video file specific signals
    positionChanged = Signal(int, int)  # current_frame, total_frames
    timeChanged = Signal(float, float)  # current_sec, total_sec
    error = Signal(str)
    
    def __init__(self, detector, extractor, classifier, resize_width: int = 960,
                 max_workers: int = 19, target_fps: float = 25.0,
                 enable_frame_skipping: bool = True, parent=None):
        super().__init__(parent)
        
        # Store initialization parameters
        self.detector = detector
        self.extractor = extractor
        self.classifier = classifier
        self.resize_width = resize_width
        self.max_workers = max_workers
        self.target_fps = target_fps
        self.enable_frame_skipping = enable_frame_skipping
        
        # Current worker and mode
        self.current_worker: Optional[Union[OptimizedVideoWorker, OptimizedVideoFileWorker]] = None
        self.current_mode: str = "none"  # "webcam", "video_file", or "none"
        self.current_video_path: Optional[str] = None
        
        # State tracking
        self._is_playing = False
        self._current_speed = 1.0
        self.fps = 25.0  # Default FPS for calculations
        
    def switch_to_webcam(self):
        """Switch to webcam input mode."""
        if self.current_mode == "webcam":
            return  # Already in webcam mode
            
        self._stop_current_worker()
        
        # Create webcam worker
        self.current_worker = OptimizedVideoWorker(
            "webcam", self.detector, self.extractor, self.classifier,
            resize_width=self.resize_width,
            max_workers=self.max_workers,
            target_fps=self.target_fps,
            enable_frame_skipping=self.enable_frame_skipping
        )
        
        self.current_mode = "webcam"
        self.current_video_path = None
        self._connect_worker_signals()
        self.current_worker.start()
        self.statusText.emit("Switched to webcam mode")
        
    def switch_to_video_file(self, path: str):
        """Switch to video file input mode."""
        if self.current_mode == "video_file" and self.current_video_path == path:
            return  # Already playing this video file
            
        self._stop_current_worker()
        
        # Create video file worker
        self.current_worker = OptimizedVideoFileWorker(
            path, self.detector, self.extractor, self.classifier,
            resize_width=self.resize_width,
            max_workers=self.max_workers,
            target_fps=self.target_fps,
            enable_frame_skipping=self.enable_frame_skipping
        )
        
        self.current_mode = "video_file"
        self.current_video_path = path
        self._connect_worker_signals()
        self.current_worker.start()
        self.statusText.emit(f"Switched to video file: {path.split('/')[-1]}")
        
    def _stop_current_worker(self):
        """Stop the current worker if running."""
        if self.current_worker:
            self.current_worker.stop()
            self.current_worker.wait(2000)
            self.current_worker = None
            
        self.current_mode = "none"
        self._is_playing = False
        
    def _connect_worker_signals(self):
        """Connect current worker signals to unified signals."""
        if not self.current_worker:
            return
            
        # Common signals
        self.current_worker.frameReady.connect(self.frameReady.emit)
        self.current_worker.heatmapReady.connect(self.heatmapReady.emit)
        self.current_worker.statusText.connect(self.statusText.emit)
        self.current_worker.classificationChanged.connect(self.classificationChanged.emit)
        self.current_worker.fpsUpdated.connect(self._on_fps_updated)
        
        # Performance monitoring
        if hasattr(self.current_worker, 'performanceStats'):
            self.current_worker.performanceStats.connect(self.performanceStats.emit)
            
        # Video file specific signals
        if hasattr(self.current_worker, 'positionChanged'):
            self.current_worker.positionChanged.connect(self.positionChanged.emit)
        if hasattr(self.current_worker, 'timeChanged'):
            self.current_worker.timeChanged.connect(self.timeChanged.emit)
        if hasattr(self.current_worker, 'error'):
            self.current_worker.error.connect(self.error.emit)
            
    def _on_fps_updated(self, fps: float):
        """Handle FPS update and store for calculations."""
        self.fps = fps
        self.fpsUpdated.emit(fps)
        
    # Control methods that work for both modes
    def play(self):
        """Start/resume playback."""
        if self.current_worker and hasattr(self.current_worker, 'play'):
            self.current_worker.play()
            self._is_playing = True
        elif self.current_mode == "webcam":
            # Webcam is always "playing"
            self._is_playing = True
            
    def pause(self):
        """Pause playback."""
        if self.current_worker and hasattr(self.current_worker, 'pause'):
            self.current_worker.pause()
            self._is_playing = False
            
    def stop(self):
        """Stop playback and reset to beginning."""
        if self.current_worker:
            if hasattr(self.current_worker, 'pause'):
                self.current_worker.pause()
            if hasattr(self.current_worker, 'seek_to_frame'):
                self.current_worker.seek_to_frame(0)
        self._is_playing = False
        
    def step_prev(self):
        """Step to previous frame (video file only)."""
        if self.current_worker and hasattr(self.current_worker, 'step_prev'):
            self.current_worker.step_prev()
            
    def step_next(self):
        """Step to next frame (video file only)."""
        if self.current_worker and hasattr(self.current_worker, 'step_next'):
            self.current_worker.step_next()
            
    def set_speed(self, speed: float):
        """Set playback speed (video file only)."""
        self._current_speed = speed
        if self.current_worker and hasattr(self.current_worker, 'set_speed'):
            self.current_worker.set_speed(speed)
            
    def seek_to_frame(self, frame: int):
        """Seek to specific frame (video file only)."""
        if self.current_worker and hasattr(self.current_worker, 'seek_to_frame'):
            self.current_worker.seek_to_frame(frame)
            
    def seek_to_time(self, seconds: float):
        """Seek to specific time (video file only)."""
        if self.current_worker and hasattr(self.current_worker, 'seek_to_frame') and self.fps > 0:
            target_frame = int(seconds * self.fps)
            self.current_worker.seek_to_frame(target_frame)
            
    # Properties
    @property
    def is_playing(self) -> bool:
        """Check if currently playing."""
        if self.current_worker and hasattr(self.current_worker, '_playing'):
            return self.current_worker._playing
        return self._is_playing
        
    @property
    def mode(self) -> str:
        """Get current mode."""
        return self.current_mode
        
    @property
    def video_path(self) -> Optional[str]:
        """Get current video file path."""
        return self.current_video_path
        
    @property
    def has_video_controls(self) -> bool:
        """Check if current mode supports video controls."""
        return self.current_mode == "video_file"
        
    def get_current_frame_data(self):
        """Get current frame data for export."""
        if self.current_worker and hasattr(self.current_worker, '_last_frame_data'):
            return self.current_worker._last_frame_data
        return None
        
    def cleanup(self):
        """Clean up resources."""
        self._stop_current_worker()
        
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
