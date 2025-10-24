from PySide6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QFileDialog,
    QSlider, QComboBox, QStyle, QStatusBar, QMessageBox, QProgressDialog, QRadioButton,
    QButtonGroup, QGroupBox, QFrame, QSizePolicy
)
from PySide6.QtGui import QPixmap, QShortcut, QKeySequence, QPainter, QPen
from PySide6.QtCore import Qt, QTimer
import time

from driversafety.gui.styles import APP_STYLESHEET, CLASS_COLORS
from driversafety.gui.performance_monitor import PerformanceMonitor


class UnifiedVideoWindow(QMainWindow):
    """Unified video processing interface that combines webcam and video file functionality."""
    
    def __init__(self, detector, extractor, classifier, resize_width: int = 960,
                 performance_settings: dict = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Driver Behavior Analysis - Unified Video Interface")
        self.setStyleSheet(APP_STYLESHEET)
        self.setWindowIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        
        # Store models and settings
        self.detector = detector
        self.extractor = extractor
        self.classifier = classifier
        self.resize_width = resize_width
        self.performance_settings = performance_settings or {}
        
        # Current mode and state
        self.current_mode = "webcam"  # "webcam" or "video_file"
        self.current_video_path = None
        self.worker = None
        self._last_frame_img = None
        self._seeking = False
        self._total_frames = 0
        
        # Initialize UI
        self._setup_ui()
        self._setup_performance_monitor()
        self._setup_connections()
        self._setup_shortcuts()
        
        # Start with webcam mode
        self._switch_to_webcam()
        
    def _setup_ui(self):
        """Set up the user interface."""
        # Main video display area with overlay for Grad-CAM
        self.video_display_frame = QFrame()
        self.video_display_frame.setFrameStyle(QFrame.Box)
        self.video_display_frame.setMinimumSize(800, 450)
        
        # Main video label
        self.video_label = QLabel("Video Display")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(800, 450)
        self.video_label.setStyleSheet("border: 1px solid #ccc; background-color: #000;")
        
        # Grad-CAM overlay label (positioned in top-right corner)
        self.heat_label = QLabel("Grad-CAM")
        self.heat_label.setAlignment(Qt.AlignCenter)
        self.heat_label.setFixedSize(200, 150)
        self.heat_label.setStyleSheet("border: 2px solid #ff6b6b; background-color: rgba(0,0,0,0.8); color: white;")
        
        # Position Grad-CAM overlay in top-right corner
        video_layout = QVBoxLayout()
        video_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create overlay container
        overlay_container = QWidget()
        overlay_container.setLayout(QVBoxLayout())
        overlay_container.layout().setContentsMargins(0, 0, 0, 0)
        overlay_container.layout().addWidget(self.video_label)
        
        # Position heat map in top-right
        self.heat_label.setParent(overlay_container)
        
        video_layout.addWidget(overlay_container)
        self.video_display_frame.setLayout(video_layout)
        
        # Mode selection controls
        mode_group = QGroupBox("Input Source")
        mode_layout = QHBoxLayout()
        
        self.radio_webcam = QRadioButton("Webcam")
        self.radio_video_file = QRadioButton("Video File")
        self.radio_webcam.setChecked(True)
        
        self.btn_select_file = QPushButton("Select Video File")
        self.btn_select_file.setEnabled(False)
        
        mode_layout.addWidget(self.radio_webcam)
        mode_layout.addWidget(self.radio_video_file)
        mode_layout.addWidget(self.btn_select_file)
        mode_layout.addStretch()
        mode_group.setLayout(mode_layout)
        
        # Video controls
        controls_group = QGroupBox("Video Controls")
        controls_layout = QVBoxLayout()
        
        # Playback controls
        playback_layout = QHBoxLayout()
        self.btn_play = QPushButton(self.style().standardIcon(QStyle.SP_MediaPlay), "")
        self.btn_pause = QPushButton(self.style().standardIcon(QStyle.SP_MediaPause), "")
        self.btn_stop = QPushButton(self.style().standardIcon(QStyle.SP_MediaStop), "")
        self.btn_prev = QPushButton(self.style().standardIcon(QStyle.SP_MediaSkipBackward), "")
        self.btn_next = QPushButton(self.style().standardIcon(QStyle.SP_MediaSkipForward), "")
        
        playback_layout.addWidget(self.btn_play)
        playback_layout.addWidget(self.btn_pause)
        playback_layout.addWidget(self.btn_stop)
        playback_layout.addWidget(self.btn_prev)
        playback_layout.addWidget(self.btn_next)
        playback_layout.addStretch()
        
        # Seek slider and time display
        seek_layout = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.time_label = QLabel("00:00 / 00:00")
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.5x", "1x", "2x"])
        self.speed_combo.setCurrentIndex(1)
        
        seek_layout.addWidget(self.slider, 3)
        seek_layout.addWidget(self.time_label)
        seek_layout.addWidget(QLabel("Speed:"))
        seek_layout.addWidget(self.speed_combo)
        
        controls_layout.addLayout(playback_layout)
        controls_layout.addLayout(seek_layout)
        controls_group.setLayout(controls_layout)
        
        # Status and information panel
        info_group = QGroupBox("Analysis Information")
        info_layout = QVBoxLayout()
        
        self.status_label = QLabel("Initializing...")
        self.class_label = QLabel("Class: -")
        self.fps_label = QLabel("FPS: -")
        
        info_layout.addWidget(self.status_label)
        info_layout.addWidget(self.class_label)
        info_layout.addWidget(self.fps_label)
        info_group.setLayout(info_layout)
        
        # Export controls
        export_group = QGroupBox("Export Options")
        export_layout = QHBoxLayout()
        
        self.btn_export_frame = QPushButton(self.style().standardIcon(QStyle.SP_DialogSaveButton), " Export Frame")
        self.btn_export_video = QPushButton(self.style().standardIcon(QStyle.SP_DialogSaveButton), " Export Video")
        self.btn_performance = QPushButton("Performance Monitor")
        
        export_layout.addWidget(self.btn_export_frame)
        export_layout.addWidget(self.btn_export_video)
        export_layout.addWidget(self.btn_performance)
        export_group.setLayout(export_layout)
        
        # Main layout
        left_panel = QVBoxLayout()
        left_panel.addWidget(self.video_display_frame, 4)
        left_panel.addWidget(controls_group)
        
        right_panel = QVBoxLayout()
        right_panel.addWidget(mode_group)
        right_panel.addWidget(info_group)
        right_panel.addWidget(export_group)
        right_panel.addStretch()
        
        main_layout = QHBoxLayout()
        main_layout.addLayout(left_panel, 3)
        main_layout.addLayout(right_panel, 1)
        
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        self.setStatusBar(QStatusBar())
        
        # Initially disable video file controls
        self._update_controls_for_mode()
        
    def _setup_performance_monitor(self):
        """Set up performance monitoring."""
        self.performance_monitor = PerformanceMonitor()
        perf = self.performance_settings
        self.performance_monitor.update_system_info(
            perf.get('enable_cuda', True),
            perf.get('enable_fp16', True),
            perf.get('max_workers', 19)
        )
        
    def _setup_connections(self):
        """Set up signal connections."""
        # Mode selection
        self.radio_webcam.toggled.connect(self._on_mode_changed)
        self.radio_video_file.toggled.connect(self._on_mode_changed)
        self.btn_select_file.clicked.connect(self._select_video_file)
        
        # Video controls (will be connected to worker when created)
        self.btn_play.clicked.connect(self._on_play)
        self.btn_pause.clicked.connect(self._on_pause)
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_prev.clicked.connect(self._on_prev_frame)
        self.btn_next.clicked.connect(self._on_next_frame)
        self.speed_combo.currentIndexChanged.connect(self._on_speed_changed)
        self.slider.sliderPressed.connect(self._on_slider_pressed)
        self.slider.sliderReleased.connect(self._on_slider_released)
        self.slider.valueChanged.connect(self._on_slider_value)
        
        # Export controls
        self.btn_export_frame.clicked.connect(self._export_frame)
        self.btn_export_video.clicked.connect(self._export_video)
        self.btn_performance.clicked.connect(self.show_performance_monitor)
        
    def _setup_shortcuts(self):
        """Set up keyboard shortcuts."""
        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self._toggle_play_pause)
        QShortcut(QKeySequence(Qt.Key_Escape), self, activated=self.close)
        QShortcut(QKeySequence(Qt.Key_Left), self, activated=self._on_prev_frame)
        QShortcut(QKeySequence(Qt.Key_Right), self, activated=self._on_next_frame)
        
    def resizeEvent(self, event):
        """Handle window resize to reposition Grad-CAM overlay."""
        super().resizeEvent(event)
        if hasattr(self, 'heat_label') and self.heat_label.parent():
            # Position heat label in top-right corner of video display
            parent_rect = self.video_label.geometry()
            x = parent_rect.width() - self.heat_label.width() - 10
            y = 10
            self.heat_label.move(x, y)
            self.heat_label.raise_()  # Ensure it's on top

    def _update_controls_for_mode(self):
        """Update control availability based on current mode."""
        is_video_file = self.current_mode == "video_file"

        # Video file specific controls
        self.slider.setEnabled(is_video_file)
        self.btn_prev.setEnabled(is_video_file)
        self.btn_next.setEnabled(is_video_file)
        self.speed_combo.setEnabled(is_video_file)

        # File selection
        self.btn_select_file.setEnabled(self.radio_video_file.isChecked())

        # Export video only available for video files
        self.btn_export_video.setEnabled(is_video_file and self.current_video_path is not None)

    def _on_mode_changed(self):
        """Handle mode change between webcam and video file."""
        if self.radio_webcam.isChecked() and self.current_mode != "webcam":
            self.current_mode = "webcam"
            self._switch_to_webcam()
        elif self.radio_video_file.isChecked() and self.current_mode != "video_file":
            self.current_mode = "video_file"
            if self.current_video_path:
                self._switch_to_video_file(self.current_video_path)
            else:
                self._select_video_file()

        self._update_controls_for_mode()

    def _select_video_file(self):
        """Open file dialog to select video file."""
        initial_dir = r"C:\\Users\\Xeron\\Videos\\DriverBehaviorVideo"
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select video file",
            initial_dir,
            "Videos (*.mp4 *.mov *.avi *.mkv)"
        )

        if path:
            self.current_video_path = path
            self._switch_to_video_file(path)
            self._update_controls_for_mode()
        else:
            # If no file selected and we're in video mode, switch back to webcam
            if self.current_mode == "video_file" and not self.current_video_path:
                self.radio_webcam.setChecked(True)

    def _switch_to_webcam(self):
        """Switch to webcam mode."""
        self._stop_current_worker()

        # Import and create webcam worker
        from driversafety.gui.optimized_video_worker import OptimizedVideoWorker

        perf = self.performance_settings
        self.worker = OptimizedVideoWorker(
            "webcam", self.detector, self.extractor, self.classifier,
            resize_width=self.resize_width,
            max_workers=perf.get('max_workers', 19),
            target_fps=perf.get('target_fps', 25.0),
            enable_frame_skipping=perf.get('enable_frame_skipping', True)
        )

        self._connect_worker_signals()
        self.worker.start()
        self.status_label.setText("Webcam mode active")

    def _switch_to_video_file(self, path: str):
        """Switch to video file mode."""
        self._stop_current_worker()

        # Import and create video file worker
        from driversafety.gui.windows.optimized_video_file_worker import OptimizedVideoFileWorker

        perf = self.performance_settings
        self.worker = OptimizedVideoFileWorker(
            path, self.detector, self.extractor, self.classifier,
            resize_width=self.resize_width,
            max_workers=perf.get('max_workers', 19),
            target_fps=perf.get('target_fps', 25.0),
            enable_frame_skipping=perf.get('enable_frame_skipping', True)
        )

        self._connect_worker_signals()
        self.worker.start()
        self.status_label.setText(f"Video file mode: {path.split('/')[-1]}")

    def _stop_current_worker(self):
        """Stop the current worker if running."""
        if self.worker:
            self.worker.stop()
            self.worker.wait(2000)
            self.worker = None

    def _connect_worker_signals(self):
        """Connect worker signals to UI slots."""
        if not self.worker:
            return

        # Common signals for both worker types
        self.worker.frameReady.connect(self._on_frame)
        self.worker.heatmapReady.connect(self._on_heat)
        self.worker.statusText.connect(self._on_status)
        self.worker.classificationChanged.connect(self._on_class)
        self.worker.fpsUpdated.connect(self._on_fps)

        # Performance monitoring
        if hasattr(self.worker, 'performanceStats'):
            self.worker.performanceStats.connect(self.performance_monitor.update_performance_stats)

        # Video file specific signals
        if hasattr(self.worker, 'positionChanged'):
            self.worker.positionChanged.connect(self._on_position)
        if hasattr(self.worker, 'timeChanged'):
            self.worker.timeChanged.connect(self._on_time)
        if hasattr(self.worker, 'error'):
            self.worker.error.connect(self._on_error)

    # Video control event handlers
    def _on_play(self):
        """Handle play button click."""
        if self.worker and hasattr(self.worker, 'play'):
            self.worker.play()

    def _on_pause(self):
        """Handle pause button click."""
        if self.worker and hasattr(self.worker, 'pause'):
            self.worker.pause()

    def _on_stop(self):
        """Handle stop button click."""
        if self.worker:
            if hasattr(self.worker, 'pause'):
                self.worker.pause()
            if hasattr(self.worker, 'seek_to_frame'):
                self.worker.seek_to_frame(0)

    def _on_prev_frame(self):
        """Handle previous frame button click."""
        if self.worker and hasattr(self.worker, 'step_prev'):
            self.worker.step_prev()

    def _on_next_frame(self):
        """Handle next frame button click."""
        if self.worker and hasattr(self.worker, 'step_next'):
            self.worker.step_next()

    def _on_speed_changed(self, index: int):
        """Handle speed change."""
        if self.worker and hasattr(self.worker, 'set_speed'):
            speed_map = {0: 0.5, 1: 1.0, 2: 2.0}
            speed = speed_map.get(index, 1.0)
            self.worker.set_speed(speed)

    def _on_slider_pressed(self):
        """Handle slider press (start seeking)."""
        self._seeking = True

    def _on_slider_released(self):
        """Handle slider release (end seeking)."""
        self._seeking = False
        if self.worker and hasattr(self.worker, 'seek_to_frame') and self._total_frames > 0:
            target_frame = int(self.slider.value() / 100.0 * self._total_frames)
            self.worker.seek_to_frame(target_frame)

    def _on_slider_value(self, value):
        """Handle slider value change during seeking."""
        if self._seeking and self._total_frames > 0:
            target_frame = int(value / 100.0 * self._total_frames)
            # Update time display during seeking
            if hasattr(self.worker, 'fps') and self.worker.fps > 0:
                target_sec = target_frame / self.worker.fps
                total_sec = self._total_frames / self.worker.fps
                self._format_time_display(target_sec, total_sec)

    def _toggle_play_pause(self):
        """Toggle between play and pause."""
        if self.worker:
            if hasattr(self.worker, '_playing') and self.worker._playing:
                self._on_pause()
            else:
                self._on_play()

    # Worker signal handlers
    def _on_frame(self, img):
        """Handle new frame from worker."""
        self._last_frame_img = img
        # Scale and display the frame
        scaled_pixmap = QPixmap.fromImage(img).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)

        # Reposition Grad-CAM overlay after frame update
        QTimer.singleShot(10, self._reposition_gradcam_overlay)

    def _on_heat(self, img):
        """Handle new heatmap from worker."""
        # Scale and display the heatmap in the overlay
        scaled_pixmap = QPixmap.fromImage(img).scaled(
            self.heat_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.heat_label.setPixmap(scaled_pixmap)

    def _on_status(self, text: str):
        """Handle status update from worker."""
        self.statusBar().showMessage(text)

    def _on_class(self, idx: int, label: str, conf: float):
        """Handle classification result from worker."""
        if conf >= 0:
            self.class_label.setText(f"Class: {label} ({conf:.2f})")
        else:
            self.class_label.setText(f"Class: {label}")

        # Apply color coding
        color = CLASS_COLORS.get(idx)
        if color:
            self.class_label.setStyleSheet(
                f"background-color: {color.name()}; color: white; padding: 6px; border-radius: 4px;"
            )

    def _on_fps(self, fps: float):
        """Handle FPS update from worker."""
        self.fps_label.setText(f"FPS: {fps:.1f}")

    def _on_position(self, current: int, total: int):
        """Handle position update from video file worker."""
        self._total_frames = total
        if not self._seeking and total > 0:
            self.slider.blockSignals(True)
            self.slider.setValue(int(current / total * 100))
            self.slider.blockSignals(False)

    def _on_time(self, current_sec: float, total_sec: float):
        """Handle time update from video file worker."""
        self._format_time_display(current_sec, total_sec)

    def _on_error(self, msg: str):
        """Handle error from worker."""
        QMessageBox.critical(self, "Error", msg)

    def _format_time_display(self, current_sec: float, total_sec: float):
        """Format and display time information."""
        def format_time(seconds):
            minutes = int(seconds // 60)
            seconds = int(seconds % 60)
            return f"{minutes:02d}:{seconds:02d}"

        self.time_label.setText(f"{format_time(current_sec)} / {format_time(total_sec)}")

    def _reposition_gradcam_overlay(self):
        """Reposition the Grad-CAM overlay in the top-right corner."""
        if hasattr(self, 'heat_label') and self.heat_label.parent():
            parent_rect = self.video_label.geometry()
            x = parent_rect.width() - self.heat_label.width() - 10
            y = 10
            self.heat_label.move(x, y)
            self.heat_label.raise_()

    # Export functionality
    def _export_frame(self):
        """Export current frame as image with analysis overlays."""
        if not self._last_frame_img:
            QMessageBox.warning(self, "Warning", "No frame available to export.")
            return

        # Get save location
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Frame",
            f"frame_{int(time.time())}.png",
            "PNG Images (*.png);;JPEG Images (*.jpg);;All Files (*)"
        )

        if filename:
            try:
                # Import video export functionality
                from driversafety.utils.video_export import VideoExporter

                # Create exporter
                exporter = VideoExporter(
                    self.detector, self.extractor, self.classifier,
                    resize_width=self.resize_width
                )

                # Export frame with analysis overlays
                success = exporter.export_frame(
                    self._last_frame_img,
                    filename,
                    include_gradcam=True,
                    include_classification=True
                )

                if success:
                    QMessageBox.information(self, "Success", f"Frame exported to:\n{filename}")
                else:
                    QMessageBox.critical(self, "Error", "Failed to save frame.")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export frame:\n{str(e)}")

    def _export_video(self):
        """Export processed video with effects and overlays."""
        if self.current_mode != "video_file" or not self.current_video_path:
            QMessageBox.warning(self, "Warning", "Video export is only available for video files.")
            return

        # Get save location
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Processed Video",
            f"processed_{int(time.time())}.mp4",
            "MP4 Videos (*.mp4);;AVI Videos (*.avi);;All Files (*)"
        )

        if filename:
            self._start_video_export(filename)

    def _start_video_export(self, output_path: str):
        """Start video export process."""
        try:
            # Create progress dialog
            progress = QProgressDialog("Exporting video...", "Cancel", 0, 100, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setAutoClose(False)
            progress.show()

            # Import video export functionality
            from driversafety.utils.video_export import VideoExporter
            from PySide6.QtCore import QThread, QObject, Signal

            # Create export worker thread
            class ExportWorker(QThread):
                progress_updated = Signal(int)
                export_finished = Signal(bool, str)

                def __init__(self, input_path, output_path, exporter):
                    super().__init__()
                    self.input_path = input_path
                    self.output_path = output_path
                    self.exporter = exporter

                def run(self):
                    try:
                        def progress_callback(percent):
                            self.progress_updated.emit(percent)

                        success = self.exporter.export_video(
                            self.input_path,
                            self.output_path,
                            progress_callback=progress_callback,
                            include_gradcam=True,
                            include_classification=True
                        )

                        self.export_finished.emit(success, self.output_path)
                    except Exception as e:
                        self.export_finished.emit(False, str(e))

            # Create exporter and worker
            exporter = VideoExporter(
                self.detector, self.extractor, self.classifier,
                resize_width=self.resize_width
            )

            self.export_worker = ExportWorker(
                self.current_video_path, output_path, exporter
            )

            # Connect signals
            self.export_worker.progress_updated.connect(progress.setValue)
            self.export_worker.export_finished.connect(
                lambda success, path: self._on_export_finished(success, path, progress)
            )

            # Handle cancel
            progress.canceled.connect(lambda: self.export_worker.terminate())

            # Start export
            self.export_worker.start()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start video export:\n{str(e)}")

    def _on_export_finished(self, success: bool, result: str, progress_dialog):
        """Handle export completion."""
        progress_dialog.close()

        if success:
            QMessageBox.information(
                self,
                "Export Complete",
                f"Video exported successfully to:\n{result}"
            )
        else:
            QMessageBox.critical(
                self,
                "Export Failed",
                f"Video export failed:\n{result}"
            )

    def show_performance_monitor(self):
        """Show performance monitor window."""
        self.performance_monitor.show()
        self.performance_monitor.raise_()
        self.performance_monitor.activateWindow()

    def closeEvent(self, event):
        """Handle window close event."""
        self._stop_current_worker()
        if hasattr(self, 'performance_monitor'):
            self.performance_monitor.close()
        event.accept()
