import numpy as np
from pytorch_grad_cam.utils.image import show_cam_on_image


def overlay_cam_on_image(rgb_float_image_0_1: np.ndarray, heatmap_0_1: np.ndarray) -> np.ndarray:
    """Overlay Grad-CAM heatmap on an RGB image in [0,1]; returns uint8 BGR for display."""
    over = show_cam_on_image(rgb_float_image_0_1, heatmap_0_1, use_rgb=True)
    return over  # already uint8 RGB by library; caller may convert if needed

