from typing import Dict, Tuple, Optional
import numpy as np
from xgboost import XGBClassifier


class BehaviorClassifier:
    """XGBoost classifier over ResNet features for driver behavior."""

    def __init__(self, model_path: str, label_map: Optional[Dict[int, str]] = None):
        self.model = XGBClassifier()
        self.model.load_model(model_path)
        self.label_map = label_map or {
            0: "safe drive",
            1: "Using phone",
            2: "Talking on phone ",  # Note: trailing space to match original exactly
            3: "Trying to reach behind",
            4: "Talking to a passenger",
        }

    def predict(self, features: np.ndarray) -> Tuple[int, str, Optional[float]]:
        """Return (class_idx, label, confidence).
        Confidence is best-effort using predict_proba if available.
        """
        idx = int(self.model.predict([features])[0])
        label = self.label_map.get(idx, "unknown")
        conf: Optional[float] = None
        try:
            proba = self.model.predict_proba([features])[0]
            conf = float(proba[idx])
        except Exception:
            pass
        return idx, label, conf

