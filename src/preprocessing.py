"""Image validation and preprocessing.

The SAME `apply_preprocessing` function is used by training, evaluation and the
web app, so inference preprocessing always matches training.
"""
from __future__ import annotations

import io
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from src.utils import load_config


class ImageValidationError(ValueError):
    """Raised with a user-friendly message when an uploaded image is unusable."""


def validate_image_bytes(data: bytes, filename: str = "image") -> Image.Image:
    """Validate extension, size, decodability and dimensions; return an RGB image."""
    cfg = load_config()
    allowed = {e.lower() for e in cfg["dataset"]["allowed_extensions"]}
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in allowed:
        raise ImageValidationError(
            f"Unsupported file type '{ext or 'unknown'}'. Please upload a JPG, JPEG or PNG image."
        )
    if not data:
        raise ImageValidationError("The uploaded file is empty.")
    max_bytes = cfg["image"]["max_upload_mb"] * 1024 * 1024
    if len(data) > max_bytes:
        raise ImageValidationError(f"File is larger than {cfg['image']['max_upload_mb']} MB.")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()                       # detects truncated / corrupted files
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        raise ImageValidationError("The file is not a valid or readable image.") from None
    w, h = img.size
    lo, hi = cfg["image"]["min_side"], cfg["image"]["max_side"]
    if min(w, h) < lo:
        raise ImageValidationError(f"Image is too small ({w}x{h}). Minimum side is {lo}px.")
    if max(w, h) > hi:
        raise ImageValidationError(f"Image is too large ({w}x{h}). Maximum side is {hi}px.")
    return to_rgb(img)


def to_rgb(img: Image.Image) -> Image.Image:
    """Convert any mode (RGBA, palette, grayscale) to RGB; flatten alpha onto white."""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[-1])
        return bg
    return img.convert("RGB")


def resize_uint8(img: Image.Image, size: tuple[int, int]) -> np.ndarray:
    """THE resize used everywhere (training, evaluation, app): PIL bilinear -> uint8 (H, W, 3)."""
    h, w = size
    return np.asarray(to_rgb(img).resize((w, h), Image.BILINEAR), dtype=np.uint8)


def resize_image(img: Image.Image, size: tuple[int, int]) -> np.ndarray:
    """Resize to (height, width) -> float32 array (H, W, 3) in [0, 255]."""
    return resize_uint8(img, size).astype(np.float32)


def load_resized(path, size: tuple[int, int]) -> np.ndarray:
    """Read an image file and return the resized uint8 array (same code path as inference)."""
    with Image.open(path) as im:
        im.load()
        return resize_uint8(im, size)


def apply_preprocessing(x: Any, scheme: str) -> Any:
    """Model-specific scaling. Works on numpy arrays and TF tensors (batched or not)."""
    if scheme == "rescale_01":
        return x / 255.0
    if scheme == "mobilenet_v2":
        from keras.applications.mobilenet_v2 import preprocess_input
        return preprocess_input(x)
    if scheme == "resnet50":
        from keras.applications.resnet50 import preprocess_input
        return preprocess_input(x)
    raise ValueError(f"Unknown preprocessing scheme: {scheme}")


def prepare_input(img: Image.Image, size: tuple[int, int], scheme: str) -> np.ndarray:
    """PIL image -> model-ready batch of shape (1, H, W, 3)."""
    arr = resize_image(to_rgb(img), size)[None, ...]
    return np.asarray(apply_preprocessing(arr.copy(), scheme), dtype=np.float32)
