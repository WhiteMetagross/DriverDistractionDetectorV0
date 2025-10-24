from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class AppConfig:
    window_title: str = "Driver Safety Detection"
    max_width: int = 1280
    max_height: int = 720


@dataclass
class ModelConfig:
    yolo_model_path: str = "modelll.pt"
    xgb_model_path: str = "xg_drive.json"
    resnet_variant: str = "resnet50"


@dataclass
class ProcessingConfig:
    confidence_threshold: float = 0.25
    max_detections: int = 5
    resize_width: int = 960


@dataclass
class Config:
    app: AppConfig
    models: ModelConfig
    processing: ProcessingConfig


def _default_config_path() -> Path:
    return Path(__file__).with_name("default.yaml")


def load_config(path: Optional[str] = None) -> Config:
    """Load YAML config into dataclasses.

    If path is None, loads the package default.yaml. If the provided path
    does not exist, falls back to default.yaml.
    """
    cfg_path = Path(path) if path else _default_config_path()
    if not cfg_path.exists():
        cfg_path = _default_config_path()

    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    app = AppConfig(**(raw.get("app", {}) or {}))
    models = ModelConfig(**(raw.get("models", {}) or {}))
    processing = ProcessingConfig(**(raw.get("processing", {}) or {}))
    return Config(app=app, models=models, processing=processing)

