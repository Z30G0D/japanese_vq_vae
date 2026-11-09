"""Geometry transforms on stacks of float images in [0, 1].

ETL4 records are 72x76, so they are padded to square rather than cropped; the
2018 project cropped the top 72 rows, which can clip strokes.
"""

import numpy as np
from PIL import Image


def pad_to_square(images: np.ndarray) -> np.ndarray:
    """Zero-pad each image to a square, centering the original content."""
    _, height, width = images.shape
    side = max(height, width)
    if height == width:
        return images

    top = (side - height) // 2
    left = (side - width) // 2
    out = np.zeros((len(images), side, side), dtype=images.dtype)
    out[:, top : top + height, left : left + width] = images
    return out


def resize(images: np.ndarray, size: int) -> np.ndarray:
    """Bilinearly resize each image to ``size`` x ``size``.

    Uses PIL's 32-bit float mode so stroke-pressure gradients survive the
    resize without being quantized to 8 bits.
    """
    out = np.empty((len(images), size, size), dtype=np.float32)
    for i, image in enumerate(images):
        pil = Image.fromarray(image.astype(np.float32), mode="F")
        out[i] = np.asarray(pil.resize((size, size), Image.BILINEAR), dtype=np.float32)
    return np.clip(out, 0.0, 1.0)
