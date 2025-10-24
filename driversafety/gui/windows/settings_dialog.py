from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QCheckBox, QFormLayout
from PySide6.QtCore import Qt, Signal

from driversafety.config.loader import load_config
from driversafety.gui.styles import get_stylesheet


class SettingsDialog(QDialog):
    themeChanged = Signal(str)  # "light" or "dark"

    def __init__(self, current_theme: str = "light", parent=None):
        super().__init__(parent)
        self.setWindowTitle("About / Settings")
        self.resize(520, 360)
        self.cfg = load_config()
        self._current_theme = current_theme

        # Info section
        form = QFormLayout()
        form.addRow("YOLO model:", QLabel(self.cfg.models.yolo_model_path))
        form.addRow("XGBoost model:", QLabel(self.cfg.models.xgb_model_path))
        form.addRow("ResNet variant:", QLabel(self.cfg.models.resnet_variant))
        form.addRow("Conf. threshold:", QLabel(f"{self.cfg.processing.confidence_threshold}"))
        form.addRow("Resize width:", QLabel(f"{self.cfg.processing.resize_width}"))
        form.addRow("Version:", QLabel("1.0.0"))
        desc = QLabel("Desktop GUI for Driver Safety Detection with YOLOv8 + ResNet50 + XGBoost + Grad-CAM")
        desc.setWordWrap(True)

        # Theme toggle
        self.chk_dark = QCheckBox("Enable dark mode")
        self.chk_dark.setChecked(current_theme.lower() == "dark")

        # Buttons
        self.btn_ok = QPushButton("OK")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_ok.clicked.connect(self._on_ok)
        self.btn_cancel.clicked.connect(self.reject)

        # Layout
        btns = QHBoxLayout(); btns.addStretch(1); btns.addWidget(self.btn_ok); btns.addWidget(self.btn_cancel)
        root = QVBoxLayout()
        root.addLayout(form)
        root.addWidget(desc)
        root.addSpacing(8)
        root.addWidget(self.chk_dark)
        root.addStretch(1)
        root.addLayout(btns)
        self.setLayout(root)

    def _on_ok(self):
        mode = "dark" if self.chk_dark.isChecked() else "light"
        if mode != self._current_theme:
            self.themeChanged.emit(mode)
        self.accept()

