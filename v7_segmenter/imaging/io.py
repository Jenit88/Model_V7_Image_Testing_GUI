"""Reading and writing images on Windows paths of any kind."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
FILE_DIALOG_TYPES = (
    ("Images", " ".join(f"*{ext}" for ext in IMAGE_EXTENSIONS)),
    ("All files", "*.*"),
)


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def read_image(path: Path) -> np.ndarray:
    """Read a photo the way model_v7.read_rgb_image does.

    np.fromfile + imdecode works for any Windows path (spaces, non-ASCII), and
    IMREAD_UNCHANGED means no EXIF rotation -- the model applies none either,
    so the display and the prediction always line up.
    """
    encoded = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Not a readable image: {path}")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def to_display_uint8(image: np.ndarray) -> np.ndarray:
    """uint8 RGB for display. 16-bit and float photos are scaled down; the
    model's own conversion replaces this view once a prediction exists."""
    if image.dtype == np.uint8:
        return np.ascontiguousarray(image)
    if image.dtype == np.uint16:
        return (image >> 8).astype(np.uint8)
    values = np.nan_to_num(image.astype(np.float32))
    if values.size and float(values.max()) <= 1.5:
        values = values * 255.0
    return np.clip(values, 0, 255).astype(np.uint8)


def write_image(path: Path, rgb: np.ndarray) -> None:
    """Save an RGB image; the format follows the file extension."""
    extension = path.suffix.lower() or ".png"
    params = [cv2.IMWRITE_JPEG_QUALITY, 95] if extension in (".jpg", ".jpeg") else []
    ok, encoded = cv2.imencode(extension, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), params)
    if not ok:
        raise ValueError(f"Could not encode an image as {extension}")
    encoded.tofile(str(path))
