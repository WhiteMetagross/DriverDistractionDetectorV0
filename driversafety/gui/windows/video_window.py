from PySide6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QFileDialog,
    QSlider, QComboBox, QStyle, QToolBar, QStatusBar, QMessageBox, QProgressDialog
)
from PySide6.QtGui import QPixmap, QShortcut, QKeySequence
from PySide6.QtCore import Qt

from driversafety.gui.styles import APP_STYLESHEET, CLASS_COLORS
from driversafety.gui.windows.optimized_video_file_worker import OptimizedVideoFileWorker
from driversafety.gui.performance_monitor import PerformanceMonitor


class VideoWindow(QMainWindow):
    def __init__(self, path: str, detector, extractor, classifier, resize_width: int = 960,
                 performance_settings: dict = None, parent=None):
        super().__init__(parent)
        print(f"[VideoWindow] Initializing with path: {path}")
        self.setWindowTitle("Video Player - About/Settings")
        self.setStyleSheet(APP_STYLESHEET)
        self.setWindowIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.path = path

        # UI widgets
        self.video_label = QLabel("Video")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(800, 450)
        self.heat_label = QLabel("Grad-CAM")
        self.heat_label.setAlignment(Qt.AlignCenter)
        self.status_label = QLabel("Initializing...")
        self.class_label = QLabel("Class: -")
        self.fps_label = QLabel("FPS: -")
        self.time_label = QLabel("00:00 / 00:00")

        # Controls
        self.btn_play = QPushButton(self.style().standardIcon(QStyle.SP_MediaPlay), "")
        self.btn_pause = QPushButton(self.style().standardIcon(QStyle.SP_MediaPause), "")
        self.btn_stop = QPushButton(self.style().standardIcon(QStyle.SP_MediaStop), "")
        self.btn_prev = QPushButton(self.style().standardIcon(QStyle.SP_MediaSkipBackward), "")
        self.btn_next = QPushButton(self.style().standardIcon(QStyle.SP_MediaSkipForward), "")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.speed = QComboBox(); self.speed.addItems(["0.5x", "1x", "2x"]); self.speed.setCurrentIndex(1)
        self.btn_export_frame = QPushButton(self.style().standardIcon(QStyle.SP_DialogSaveButton), " Export Frame")
        self.btn_export_video = QPushButton(self.style().standardIcon(QStyle.SP_DialogSaveButton), " Export Processed Video")
        self.btn_performance = QPushButton("Performance Monitor")

        # Layouts
        left = QVBoxLayout()
        left.addWidget(self.video_label, 5)
        left.addWidget(self.slider)
        controls = QHBoxLayout()
        controls.addWidget(self.btn_play)
        controls.addWidget(self.btn_pause)
        controls.addWidget(self.btn_stop)
        controls.addWidget(self.btn_prev)
        controls.addWidget(self.btn_next)
        controls.addWidget(self.speed)
        controls.addWidget(self.time_label)
        left.addLayout(controls)

        right = QVBoxLayout()
        right.addWidget(self.heat_label, 3)
        right.addWidget(self.status_label)
        right.addWidget(self.class_label)
        right.addWidget(self.fps_label)
        right.addWidget(self.btn_export_frame)
        right.addWidget(self.btn_export_video)
        right.addWidget(self.btn_performance)

        root = QHBoxLayout()
        root.addLayout(left, 3)
        root.addLayout(right, 1)
        container = QWidget(); container.setLayout(root)
        self.setCentralWidget(container)
        self.setStatusBar(QStatusBar())

        # Performance settings
        perf = performance_settings or {}
        max_workers = perf.get('max_workers', 19)
        target_fps = perf.get('target_fps', 25.0)
        enable_frame_skipping = perf.get('enable_frame_skipping', True)

        # Performance monitor
        self.performance_monitor = PerformanceMonitor()
        self.performance_monitor.update_system_info(
            perf.get('enable_cuda', True),
            perf.get('enable_fp16', True),
            max_workers
        )

        # Worker
        self.worker = OptimizedVideoFileWorker(
            path, detector, extractor, classifier,
            resize_width=resize_width,
            max_workers=max_workers,
            target_fps=target_fps,
            enable_frame_skipping=enable_frame_skipping
        )
        self.worker.frameReady.connect(self._on_frame)
        self.worker.heatmapReady.connect(self._on_heat)
        self.worker.statusText.connect(self._on_status)
        self.worker.classificationChanged.connect(self._on_class)
        self.worker.fpsUpdated.connect(self._on_fps)
        self.worker.positionChanged.connect(self._on_pos)
        self.worker.timeChanged.connect(self._on_time)
        self.worker.error.connect(self._on_error)
        if hasattr(self.worker, 'performanceStats'):
            self.worker.performanceStats.connect(self.performance_monitor.update_performance_stats)
        self._total_frames = 0

        # Connect controls
        for b, tip in [
            (self.btn_play, "Play (Space)"),
            (self.btn_pause, "Pause (Space)"),
            (self.btn_stop, "Stop / Restart"),
            (self.btn_prev, "Previous frame (←)"),
            (self.btn_next, "Next frame (→)"),
        ]:
            b.setToolTip(tip)
        self.btn_export_frame.setToolTip("Export current frame (PNG)")
        self.btn_export_video.setToolTip("Export processed video")

        self.btn_play.clicked.connect(self.worker.play)
        self.btn_pause.clicked.connect(self.worker.pause)
        self.btn_stop.clicked.connect(self._on_stop_clicked)
        self.btn_prev.clicked.connect(self.worker.step_prev)
        self.btn_next.clicked.connect(self.worker.step_next)
        self.speed.currentIndexChanged.connect(self._on_speed_changed)
        self.slider.sliderPressed.connect(self._on_slider_pressed)
        self.slider.sliderReleased.connect(self._on_slider_released)
        self.slider.valueChanged.connect(self._on_slider_value)
        self.btn_export_frame.clicked.connect(self._export_frame)
        self.btn_export_video.clicked.connect(self._export_video)
        self.btn_performance.clicked.connect(self.show_performance_monitor)

        # Shortcuts
        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self._toggle_play_pause)
        QShortcut(QKeySequence(Qt.Key_Left), self, activated=self.worker.step_prev)
        QShortcut(QKeySequence(Qt.Key_Right), self, activated=self.worker.step_next)
        QShortcut(QKeySequence(Qt.Key_Plus), self, activated=self._speed_up)
        QShortcut(QKeySequence(Qt.Key_Minus), self, activated=self._speed_down)
        QShortcut(QKeySequence(Qt.Key_Home), self, activated=lambda: self.worker.seek_to_frame(0))
        QShortcut(QKeySequence(Qt.Key_End), self, activated=self._seek_end)
        QShortcut(QKeySequence(Qt.Key_Escape), self, activated=self.close)

        self._seeking = False
        self._last_frame_img = None
        print("[VideoWindow] Starting video file worker thread...")
        self.worker.start()
        print("[VideoWindow] Worker thread started.")

    def closeEvent(self, event):
        if self.worker:
            self.worker.stop(); self.worker.wait(2000)
        event.accept()

    # Slots
    def _on_frame(self, img):
        # Debug first-frame log
        if self._last_frame_img is None:
            print("[VideoWindow] Received first frame.")
        self._last_frame_img = img
        self.video_label.setPixmap(QPixmap.fromImage(img).scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _on_heat(self, img):
        self.heat_label.setPixmap(QPixmap.fromImage(img).scaled(self.heat_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _on_status(self, text: str):
        self.statusBar().showMessage(text)

    def _on_class(self, idx: int, label: str, conf: float):
        self.class_label.setText(f"Class: {label} ({conf:.2f})" if conf >= 0 else f"Class: {label}")
        # Colorize background
        from driversafety.gui.styles import CLASS_COLORS
        color = CLASS_COLORS.get(idx)
        if color:
            self.class_label.setStyleSheet(f"background-color: {color.name()}; color: white; padding: 6px; border-radius: 4px;")

    def _on_fps(self, fps: float):
        self.fps_label.setText(f"FPS: {fps:.1f}")

    def _on_pos(self, cur: int, total: int):
        self._total_frames = total
        if not self._seeking and total > 0:
            self.slider.blockSignals(True)
            self.slider.setValue(int(cur / total * 100))
            self.slider.blockSignals(False)

    def _on_time(self, cur_sec: float, total_sec: float):
        def fmt(s):
            m = int(s // 60); s2 = int(s % 60); return f"{m:02}:{s2:02}"
        self.time_label.setText(f"{fmt(cur_sec)} / {fmt(total_sec)}")

    def _on_error(self, msg: str):
        print(f"[VideoWindow] Error: {msg}")
        QMessageBox.critical(self, "Error", msg)

    def _toggle_play_pause(self):
        # crude: if paused, play; else pause by toggling based on icon
        if self.btn_play.isEnabled():
            self.worker.play()
        else:
            self.worker.pause()

    def _on_stop_clicked(self):
        self.worker.pause()
        self.worker.seek_to_frame(0)

    def _on_speed_changed(self, idx: int):
        sp = {0:0.5, 1:1.0, 2:2.0}.get(idx, 1.0)
        self.worker.set_speed(sp)

    def _speed_up(self):
        i = min(self.speed.currentIndex() + 1, self.speed.count()-1)
        self.speed.setCurrentIndex(i)

    def _speed_down(self):
        i = max(self.speed.currentIndex() - 1, 0)
        self.speed.setCurrentIndex(i)

    def _seek_end(self):
        if hasattr(self, "_total_frames") and self._total_frames > 0:
            self.worker.seek_to_frame(self._total_frames - 1)

    def _on_slider_pressed(self):
        self._seeking = True

    def _on_slider_released(self):
        self._seeking = False

    def _on_slider_value(self, value: int):
        if self._seeking and self._total_frames > 0:
            frame_index = int(value / 100.0 * (self._total_frames - 1))
            self.worker.seek_to_frame(frame_index)

    # Export operations
    def _export_frame(self):
        if self._last_frame_img is None:
            QMessageBox.information(self, "Export", "No frame to export yet.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Current Frame", "frame.png", "PNG Image (*.png)")
        if not path:
            return
        self._last_frame_img.save(path)

    def _export_video(self):
        out_path, _ = QFileDialog.getSaveFileName(self, "Export Processed Video", "processed.mp4", "MP4 Video (*.mp4);;AVI (*.avi)")
        if not out_path:
            return

        # Create progress dialog
        dlg = QProgressDialog("Exporting...", "Cancel", 0, 100, self)
        dlg.setWindowModality(Qt.ApplicationModal)
        dlg.show()

        try:
            # Use the VideoExporter class with fallback codec support
            from driversafety.utils.video_export import VideoExporter

            # Create exporter using detector/extractor/classifier from worker
            exporter = VideoExporter(
                self.worker.detector,
                self.worker.extractor,
                self.worker.classifier
            )

            # Define progress callback
            def progress_callback(percent):
                dlg.setValue(percent)
                # Process events to allow cancel button to work
                from PySide6.QtWidgets import QApplication
                QApplication.processEvents()

            # Export video with fallback codec support
            success = exporter.export_video(
                self.path,
                out_path,
                progress_callback=progress_callback,
                include_gradcam=True,
                include_classification=True
            )

            dlg.close()

            if success:
                QMessageBox.information(self, "Export", f"Export finished successfully.\nOutput: {out_path}")
            else:
                QMessageBox.critical(self, "Export Error", "Export failed. Check console for details.")

        except Exception as e:
            dlg.close()
            QMessageBox.critical(self, "Export Error", f"Export failed:\n{str(e)}")

    def show_performance_monitor(self):
        """Show the performance monitoring window."""
        self.performance_monitor.show()
        self.performance_monitor.raise_()
        self.performance_monitor.activateWindow()

