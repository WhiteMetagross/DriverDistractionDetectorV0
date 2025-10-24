from typing import Tuple
import torch
from torchvision import models, transforms
import numpy as np
from PIL import Image


def default_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class ResNetFeatureExtractor:
    """ResNet-based feature extractor that outputs a 2048-d feature for ResNet50."""

    def __init__(self, variant: str = "resnet50", device: str = "cpu"):
        self.device = torch.device(device)
        if variant != "resnet50":
            raise ValueError("Only resnet50 is supported in this reference implementation")
        # Use pretrained=True to match original Streamlit exactly
        backbone = models.resnet50(pretrained=True)
        # Save full model for Grad-CAM
        self.model = backbone.to(self.device).eval()
        # Feature extractor w/o final FC
        self.extractor = torch.nn.Sequential(*list(backbone.children())[:-1]).to(self.device).eval()
        self.transform = default_transform()

    @torch.no_grad()
    def extract(self, pil_image: Image.Image) -> Tuple[np.ndarray, torch.Tensor]:
        """Return (features_numpy, input_tensor) for downstream classifier and Grad-CAM."""
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        feat = self.extractor(tensor).squeeze().detach().cpu().numpy()
        return feat, tensor

