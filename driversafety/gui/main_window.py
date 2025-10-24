from PySide6.QtWidgets import QMainWindow, QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QStatusBar, QMenuBar, QMenu, QStyle
from PySide6.QtCore import Qt

from driversafety.config.loader import load_config
from driversafety.detection.optimized_face_detector import OptimizedDriverFaceDetector
from driversafety.classification.optimized_feature_extractor import OptimizedResNetFeatureExtractor
from driversafety.classification.behavior_classifier import BehaviorClassifier
from driversafety.gui.styles import get_stylesheet
from driversafety.gui.windows.unified_video_window import UnifiedVideoWindow
from driversafety.gui.windows.settings_dialog import SettingsDialog


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.cfg = load_config()
        self.setWindowTitle(self.cfg.app.window_title)
        self._theme = "light"
        self.setStyleSheet(get_stylesheet(self._theme))
        self.setWindowIcon(self.style().standardIcon(QStyle.SP_DesktopIcon))

        # Initialize optimized models with performance settings
        perf_cfg = getattr(self.cfg, 'performance', None)
        enable_cuda = perf_cfg.enable_cuda if perf_cfg else True
        enable_fp16 = perf_cfg.enable_fp16 if perf_cfg else True
        max_workers = perf_cfg.max_worker_threads if perf_cfg else 19
        optimization_method = perf_cfg.optimization_method if perf_cfg else "jit_trace"

        device = "cuda" if enable_cuda else "cpu"

        print("Initializing optimized models...")
        self.detector = OptimizedDriverFaceDetector(
            self.cfg.models.yolo_model_path,
            conf_threshold=self.cfg.processing.confidence_threshold,
            device=device,
            enable_fp16=enable_fp16,
            max_workers=max(1, max_workers // 4)
        )
        self.extractor = OptimizedResNetFeatureExtractor(
            self.cfg.models.resnet_variant,
            device=device,
            enable_fp16=enable_fp16,
            max_workers=max(1, max_workers // 2),
            optimization_method=optimization_method
        )
        self.classifier = BehaviorClassifier(self.cfg.models.xgb_model_path)

        print(f"✓ Models initialized with device: {device}, FP16: {enable_fp16}, Optimization: {optimization_method}")

        # Store performance settings for windows
        self.performance_settings = {
            'max_workers': max_workers,
            'enable_cuda': enable_cuda,
            'enable_fp16': enable_fp16,
            'optimization_method': optimization_method,
            'target_fps': perf_cfg.target_fps if perf_cfg else 25.0,
            'enable_frame_skipping': perf_cfg.enable_frame_skipping if perf_cfg else True
        }

        # Launcher UI
        title = QLabel("Driver Behavior Analysis")
        title.setObjectName("Title")
        title.setAlignment(Qt.AlignCenter)
        subtitle = QLabel("Unified video processing interface for webcam and video files")
        subtitle.setAlignment(Qt.AlignCenter)

        self.btn_unified = QPushButton("Start Video Analysis")
        self.btn_unified.clicked.connect(self.on_unified_interface)

        # Style the main button
        self.btn_unified.setStyleSheet("QPushButton { font-weight: bold; background-color: #4CAF50; color: white; font-size: 16px; }")
        self.btn_unified.setMinimumHeight(50)
        self.btn_unified.setMinimumWidth(300)

        col = QVBoxLayout()
        col.addStretch(1)
        col.addWidget(title)
        col.addWidget(subtitle)
        col.addSpacing(30)

        # Main interface button
        button_row = QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(self.btn_unified)
        button_row.addStretch(1)
        col.addLayout(button_row)
        col.addStretch(2)

        container = QWidget(); container.setLayout(col)
        self.setCentralWidget(container)
        self.setStatusBar(QStatusBar())

        # Menu
        menubar = QMenuBar(self)
        menu_app = QMenu("App", menubar)
        act_settings = menu_app.addAction("About / Settings")
        act_settings.triggered.connect(self._open_settings)
        menubar.addMenu(menu_app)
        self.setMenuBar(menubar)

        self.resize(self.cfg.app.max_width, self.cfg.app.max_height)

        self._unified_windows = []  # keep refs to unified windows

    def on_unified_interface(self):
        """Open the unified video processing interface."""
        try:
            print("[MainWindow] Creating unified video interface...")
            win = UnifiedVideoWindow(
                self.detector, self.extractor, self.classifier,
                resize_width=self.cfg.processing.resize_width,
                performance_settings=self.performance_settings,
                parent=self
            )
            # Apply current theme to child window
            win.setStyleSheet(get_stylesheet(self._theme))
            win.show()
            win.raise_()
            win.activateWindow()
            self._unified_windows.append(win)
            print("[MainWindow] Unified video interface shown successfully.")
        except Exception as e:
            import traceback
            print(f"[MainWindow] Error opening unified interface: {e}")
            traceback.print_exc()
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error", f"Failed to open unified interface:\n{e}")



    def _open_settings(self):
        dlg = SettingsDialog(self._theme, self)
        dlg.themeChanged.connect(self._apply_theme)
        dlg.exec()

    def _apply_theme(self, mode: str):
        self._theme = mode
        self.setStyleSheet(get_stylesheet(self._theme))
        for w in self._unified_windows:
            try:
                w.setStyleSheet(get_stylesheet(self._theme))
            except Exception:
                pass

