from PIL import Image
import numpy as np
import cv2


def get_edge_for_visualization(pil_img: Image.Image) -> np.ndarray:
    """Return an edge-based RGB image in [0,1] float32 suitable for overlays."""
    w, h = pil_img.size
    scale = 256 / min(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = pil_img.resize((new_w, new_h), Image.BILINEAR)

    left = (resized.width - 224) // 2
    top = (resized.height - 224) // 2
    cropped = resized.crop((left, top, left + 224, top + 224))

    gray = cv2.cvtColor(np.array(cropped), cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = 255 - cv2.Canny(blurred, 30, 80)

    edge_rgb = np.stack([edges] * 3, axis=-1).astype(np.float32) / 255.0
    return edge_rgb

