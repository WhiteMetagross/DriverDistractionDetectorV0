"""Driver Safety Detection package

Provides modular components for driver behavior detection:
- detection: YOLO-based driver/face detector
- classification: ResNet feature extractor + XGBoost behavior classifier
- visualization: Grad-CAM and edge visualizations
- gui: PySide6 desktop application
- config: configuration management
"""

__all__ = [
    "config",
    "detection",
    "classification",
    "visualization",
    "gui",
]

__version__ = "0.1.0"

