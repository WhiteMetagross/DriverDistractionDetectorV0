from typing import List, Optional
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def compute_gradcam(model: torch.nn.Module, input_tensor: torch.Tensor, target_layers: List[torch.nn.Module], target_class: Optional[int]) -> torch.Tensor:
    """Compute Grad-CAM heatmap for a single image tensor.

    Returns a 1xHxW tensor on CPU with values in [0, 1] (float32).
    """
    # pytorch-grad-cam infers device; 'use_cuda' arg is deprecated/removed in newer versions
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = None if target_class is None else [ClassifierOutputTarget(target_class)]
    heatmap = cam(input_tensor=input_tensor, targets=targets)
    # heatmap is numpy array (N,H,W) in [0,1]; convert to torch for consistency
    return torch.from_numpy(heatmap)

