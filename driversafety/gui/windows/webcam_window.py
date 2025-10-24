from PySide6.QtWidgets import QMainWindow, QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QStyle
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt

from driversafety.gui.styles import APP_STYLESHEET, CLASS_COLORS
from driversafety.gui.optimized_video_worker import OptimizedVideoWorker
from driversafety.gui.performance_monitor import PerformanceMonitor


class WebcamWindow(QMainWindow):
    def __init__(self, detector, extractor, classifier, resize_width: int = 960,
                 performance_settings: dict = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Webcam - About/Settings (Optimized)")
        self.setStyleSheet(APP_STYLESHEET)
        self.setWindowIcon(self.style().standardIcon(QStyle.SP_ComputerIcon))

        # Performance settings
        perf = performance_settings or {}
        max_workers = perf.get('max_workers', 19)
        target_fps = perf.get('target_fps', 25.0)
        enable_frame_skipping = perf.get('enable_frame_skipping', True)

        # Create optimized worker
        self.worker = OptimizedVideoWorker(
            "webcam", detector, extractor, classifier,
            resize_width=resize_width,
            max_workers=max_workers,
            target_fps=target_fps,
            enable_frame_skipping=enable_frame_skipping
        )

        # UI
        self.video_label = QLabel("Video")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 360)
        self.heat_label = QLabel("Grad-CAM")
        self.heat_label.setAlignment(Qt.AlignCenter)
        self.status_label = QLabel("Initializing...")
        self.class_label = QLabel("Class: -")
        self.fps_label = QLabel("FPS: -")
        self.btn_close = QPushButton(self.style().standardIcon(QStyle.SP_DialogCloseButton), " Stop / Close")

        left = QVBoxLayout()
        left.addWidget(self.video_label, 5)
        left.addWidget(self.status_label)

        right = QVBoxLayout()
        right.addWidget(self.heat_label, 3)
        right.addWidget(self.class_label)
        right.addWidget(self.fps_label)
        right.addWidget(self.btn_close)

        root = QHBoxLayout()
        root.addLayout(left, 2)
        root.addLayout(right, 1)
        container = QWidget(); container.setLayout(root)
        self.setCentralWidget(container)

        # Performance monitor
        self.performance_monitor = PerformanceMonitor()
        self.performance_monitor.update_system_info(
            perf.get('enable_cuda', True),
            perf.get('enable_fp16', True),
            max_workers
        )

        # Add performance monitor button
        self.btn_perf = QPushButton("Performance Monitor")
        self.btn_perf.clicked.connect(self.show_performance_monitor)
        right.addWidget(self.btn_perf)

        # Signals
        self.worker.frameReady.connect(self._on_frame)
        self.worker.heatmapReady.connect(self._on_heat)
        self.worker.statusText.connect(self._on_status)
        self.worker.classificationChanged.connect(self._on_class)
        self.worker.fpsUpdated.connect(self._on_fps)
        if hasattr(self.worker, 'performanceStats'):
            self.worker.performanceStats.connect(self.performance_monitor.update_performance_stats)
        self.btn_close.clicked.connect(self.close)

        # Shortcuts
        from PySide6.QtGui import QShortcut, QKeySequence
        QShortcut(QKeySequence(Qt.Key_Escape), self, activated=self.close)

        self.worker.start()

    def closeEvent(self, event):
        if self.worker:
            self.worker.stop(); self.worker.wait(1000)
        event.accept()

    def _on_frame(self, img):
        self.video_label.setPixmap(QPixmap.fromImage(img).scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _on_heat(self, img):
        self.heat_label.setPixmap(QPixmap.fromImage(img).scaled(self.heat_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _on_status(self, text: str):
        self.status_label.setText(text)

    def _on_class(self, idx: int, label: str, conf: float):
        self.class_label.setText(f"Class: {label} ({conf:.2f})" if conf >= 0 else f"Class: {label}")
        color = CLASS_COLORS.get(idx)
        if color:
            self.class_label.setStyleSheet(f"background-color: {color.name()}; color: white; padding: 6px; border-radius: 4px;")

    def _on_fps(self, fps: float):
        self.fps_label.setText(f"FPS: {fps:.1f}")

    def show_performance_monitor(self):
        """Show the performance monitoring window."""
        self.performance_monitor.show()
        self.performance_monitor.raise_()
        self.performance_monitor.activateWindow()

