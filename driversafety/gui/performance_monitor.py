from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QProgressBar, QGroupBox, QGridLayout)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
import time
from typing import Dict, Any


class PerformanceMonitor(QWidget):
    """Real-time performance monitoring widget for video processing pipeline."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Performance Monitor")
        self.setMinimumSize(400, 300)
        
        # Performance data storage
        self.performance_history = []
        self.max_history = 100
        
        # Setup UI
        self._setup_ui()
        
        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_display)
        self.update_timer.start(500)  # Update every 500ms
        
        # Initialize with default values
        self._reset_stats()
    
    def _setup_ui(self):
        """Setup the performance monitoring UI."""
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Real-Time Performance Monitor")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(title)
        
        # Overall Performance Group
        overall_group = QGroupBox("Overall Performance")
        overall_layout = QGridLayout(overall_group)
        
        # FPS Display
        self.fps_label = QLabel("FPS:")
        self.fps_value = QLabel("0.0")
        self.fps_value.setStyleSheet("color: #2E8B57; font-weight: bold; font-size: 16px;")
        self.fps_bar = QProgressBar()
        self.fps_bar.setRange(0, 60)
        self.fps_bar.setValue(0)
        
        overall_layout.addWidget(self.fps_label, 0, 0)
        overall_layout.addWidget(self.fps_value, 0, 1)
        overall_layout.addWidget(self.fps_bar, 0, 2)
        
        # Processing Time
        self.proc_time_label = QLabel("Avg Processing Time:")
        self.proc_time_value = QLabel("0.0 ms")
        self.proc_time_bar = QProgressBar()
        self.proc_time_bar.setRange(0, 100)  # 0-100ms
        
        overall_layout.addWidget(self.proc_time_label, 1, 0)
        overall_layout.addWidget(self.proc_time_value, 1, 1)
        overall_layout.addWidget(self.proc_time_bar, 1, 2)
        
        layout.addWidget(overall_group)
        
        # Component Performance Group
        component_group = QGroupBox("Component Performance")
        component_layout = QGridLayout(component_group)
        
        # Detection Performance
        self.det_label = QLabel("YOLO Detection:")
        self.det_fps = QLabel("0.0 FPS")
        self.det_time = QLabel("0.0 ms")
        self.det_bar = QProgressBar()
        self.det_bar.setRange(0, 100)
        
        component_layout.addWidget(self.det_label, 0, 0)
        component_layout.addWidget(self.det_fps, 0, 1)
        component_layout.addWidget(self.det_time, 0, 2)
        component_layout.addWidget(self.det_bar, 0, 3)
        
        # Feature Extraction Performance
        self.feat_label = QLabel("ResNet Features:")
        self.feat_fps = QLabel("0.0 FPS")
        self.feat_time = QLabel("0.0 ms")
        self.feat_bar = QProgressBar()
        self.feat_bar.setRange(0, 100)
        
        component_layout.addWidget(self.feat_label, 1, 0)
        component_layout.addWidget(self.feat_fps, 1, 1)
        component_layout.addWidget(self.feat_time, 1, 2)
        component_layout.addWidget(self.feat_bar, 1, 3)
        
        layout.addWidget(component_group)
        
        # GPU/System Info Group
        system_group = QGroupBox("System Information")
        system_layout = QGridLayout(system_group)
        
        self.gpu_label = QLabel("GPU Acceleration:")
        self.gpu_status = QLabel("Unknown")
        self.precision_label = QLabel("Precision:")
        self.precision_status = QLabel("Unknown")
        self.threads_label = QLabel("Worker Threads:")
        self.threads_status = QLabel("Unknown")
        
        system_layout.addWidget(self.gpu_label, 0, 0)
        system_layout.addWidget(self.gpu_status, 0, 1)
        system_layout.addWidget(self.precision_label, 1, 0)
        system_layout.addWidget(self.precision_status, 1, 1)
        system_layout.addWidget(self.threads_label, 2, 0)
        system_layout.addWidget(self.threads_status, 2, 1)
        
        layout.addWidget(system_group)
        
        # Performance Comparison Group
        comparison_group = QGroupBox("Performance Comparison")
        comparison_layout = QGridLayout(comparison_group)
        
        self.baseline_label = QLabel("Baseline (Original):")
        self.baseline_fps = QLabel("~15 FPS")
        self.baseline_fps.setStyleSheet("color: #CD5C5C;")
        
        self.optimized_label = QLabel("Optimized (Current):")
        self.optimized_fps = QLabel("0.0 FPS")
        self.optimized_fps.setStyleSheet("color: #2E8B57; font-weight: bold;")
        
        self.improvement_label = QLabel("Improvement:")
        self.improvement_value = QLabel("0.0x")
        self.improvement_value.setStyleSheet("color: #4169E1; font-weight: bold;")
        
        comparison_layout.addWidget(self.baseline_label, 0, 0)
        comparison_layout.addWidget(self.baseline_fps, 0, 1)
        comparison_layout.addWidget(self.optimized_label, 1, 0)
        comparison_layout.addWidget(self.optimized_fps, 1, 1)
        comparison_layout.addWidget(self.improvement_label, 2, 0)
        comparison_layout.addWidget(self.improvement_value, 2, 1)
        
        layout.addWidget(comparison_group)
    
    def _reset_stats(self):
        """Reset all statistics to default values."""
        self.current_fps = 0.0
        self.current_proc_time = 0.0
        self.detector_stats = {"fps": 0.0, "avg_time": 0.0}
        self.extractor_stats = {"fps": 0.0, "avg_time": 0.0}
        self.baseline_fps = 15.0  # Estimated baseline performance
    
    def update_performance_stats(self, stats: Dict[str, Any]):
        """Update performance statistics from the video worker."""
        self.current_fps = stats.get('fps', 0.0)
        self.current_proc_time = stats.get('avg_processing_time', 0.0) * 1000  # Convert to ms
        
        self.detector_stats = stats.get('detector_stats', {"fps": 0.0, "avg_time": 0.0})
        self.extractor_stats = stats.get('extractor_stats', {"fps": 0.0, "avg_time": 0.0})
        
        # Store in history
        self.performance_history.append({
            'timestamp': time.time(),
            'fps': self.current_fps,
            'proc_time': self.current_proc_time
        })
        
        # Limit history size
        if len(self.performance_history) > self.max_history:
            self.performance_history.pop(0)
    
    def update_system_info(self, gpu_enabled: bool, fp16_enabled: bool, num_threads: int):
        """Update system information display."""
        gpu_text = "CUDA Enabled" if gpu_enabled else "CPU Only"
        gpu_color = "#2E8B57" if gpu_enabled else "#CD5C5C"
        self.gpu_status.setText(gpu_text)
        self.gpu_status.setStyleSheet(f"color: {gpu_color}; font-weight: bold;")
        
        precision_text = "FP16 (Half)" if fp16_enabled else "FP32 (Full)"
        precision_color = "#2E8B57" if fp16_enabled else "#FF8C00"
        self.precision_status.setText(precision_text)
        self.precision_status.setStyleSheet(f"color: {precision_color}; font-weight: bold;")
        
        self.threads_status.setText(str(num_threads))
        self.threads_status.setStyleSheet("color: #4169E1; font-weight: bold;")
    
    def _update_display(self):
        """Update the display with current performance data."""
        # Overall Performance
        self.fps_value.setText(f"{self.current_fps:.1f}")
        self.fps_bar.setValue(min(int(self.current_fps), 60))
        
        # Color coding for FPS
        if self.current_fps >= 25:
            fps_color = "#2E8B57"  # Green
        elif self.current_fps >= 15:
            fps_color = "#FF8C00"  # Orange
        else:
            fps_color = "#CD5C5C"  # Red
        
        self.fps_value.setStyleSheet(f"color: {fps_color}; font-weight: bold; font-size: 16px;")
        
        # Processing Time
        self.proc_time_value.setText(f"{self.current_proc_time:.1f} ms")
        self.proc_time_bar.setValue(min(int(self.current_proc_time), 100))
        
        # Component Performance
        det_fps = self.detector_stats.get('fps', 0.0)
        det_time = self.detector_stats.get('avg_time', 0.0) * 1000
        
        self.det_fps.setText(f"{det_fps:.1f} FPS")
        self.det_time.setText(f"{det_time:.1f} ms")
        self.det_bar.setValue(min(int(det_time), 100))
        
        feat_fps = self.extractor_stats.get('fps', 0.0)
        feat_time = self.extractor_stats.get('avg_time', 0.0) * 1000
        
        self.feat_fps.setText(f"{feat_fps:.1f} FPS")
        self.feat_time.setText(f"{feat_time:.1f} ms")
        self.feat_bar.setValue(min(int(feat_time), 100))
        
        # Performance Comparison
        self.optimized_fps.setText(f"{self.current_fps:.1f} FPS")
        
        if self.baseline_fps > 0:
            improvement = self.current_fps / self.baseline_fps
            self.improvement_value.setText(f"{improvement:.1f}x")
            
            # Color coding for improvement
            if improvement >= 2.0:
                improvement_color = "#2E8B57"  # Green
            elif improvement >= 1.5:
                improvement_color = "#FF8C00"  # Orange
            else:
                improvement_color = "#CD5C5C"  # Red
            
            self.improvement_value.setStyleSheet(f"color: {improvement_color}; font-weight: bold;")
    
    def get_average_fps(self, window_seconds: int = 10) -> float:
        """Get average FPS over the specified time window."""
        if not self.performance_history:
            return 0.0
        
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        recent_data = [entry for entry in self.performance_history 
                      if entry['timestamp'] >= cutoff_time]
        
        if not recent_data:
            return 0.0
        
        return sum(entry['fps'] for entry in recent_data) / len(recent_data)
    
    def export_performance_data(self) -> Dict[str, Any]:
        """Export current performance data for analysis."""
        return {
            'current_fps': self.current_fps,
            'current_proc_time': self.current_proc_time,
            'detector_stats': self.detector_stats,
            'extractor_stats': self.extractor_stats,
            'average_fps_10s': self.get_average_fps(10),
            'performance_history': self.performance_history[-50:]  # Last 50 entries
        }
