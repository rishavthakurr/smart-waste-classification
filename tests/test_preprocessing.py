import io

import numpy as np
import pytest
from PIL import Image

from src.preprocessing import (ImageValidationError, apply_preprocessing, prepare_input,
                               to_rgb, validate_image_bytes)


def make_bytes(size=(64, 64), mode="RGB", fmt="PNG", color=(10, 120, 30)):
    buf = io.BytesIO(); Image.new(mode, size, color if mode == "RGB" else None).save(buf, fmt)
    return buf.getvalue()


def test_valid_png_and_jpeg_accepted():
    assert validate_image_bytes(make_bytes(), "a.png").mode == "RGB"
    assert validate_image_bytes(make_bytes(fmt="JPEG"), "a.JPG").size == (64, 64)


@pytest.mark.parametrize("name", ["a.gif", "a.txt", "noextension"])
def test_unsupported_extension_rejected(name):
    with pytest.raises(ImageValidationError, match="Unsupported"):
        validate_image_bytes(make_bytes(), name)


def test_empty_corrupt_and_truncated_rejected():
    with pytest.raises(ImageValidationError):
        validate_image_bytes(b"", "a.png")
    with pytest.raises(ImageValidationError, match="not a valid"):
        validate_image_bytes(b"not an image at all", "a.png")
    with pytest.raises(ImageValidationError):
        validate_image_bytes(make_bytes(size=(200, 200), fmt="JPEG")[:300], "a.jpg")


def test_dimension_limits():
    with pytest.raises(ImageValidationError, match="too small"):
        validate_image_bytes(make_bytes(size=(10, 10)), "a.png")


def test_rgba_flattened_onto_white():
    rgba = Image.new("RGBA", (40, 40), (0, 0, 0, 0))
    assert to_rgb(rgba).getpixel((0, 0)) == (255, 255, 255)
    assert to_rgb(Image.new("L", (8, 8), 5)).mode == "RGB"


def test_prepare_input_shape_dtype_and_ranges():
    img = Image.new("RGB", (300, 200), (255, 255, 255))
    x = prepare_input(img, (96, 96), "rescale_01")
    assert x.shape == (1, 96, 96, 3) and x.dtype == np.float32 and np.isclose(x.max(), 1.0)
    m = prepare_input(img, (64, 64), "mobilenet_v2")
    assert m.shape == (1, 64, 64, 3) and np.isclose(m.max(), 1.0)
    black = prepare_input(Image.new("RGB", (50, 50)), (32, 32), "mobilenet_v2")
    assert np.isclose(black.min(), -1.0)


def test_resnet_preprocessing_is_bgr_mean_subtraction():
    x = np.full((1, 4, 4, 3), [255.0, 0.0, 0.0], dtype=np.float32)   # pure red (RGB)
    y = apply_preprocessing(x.copy(), "resnet50")
    # caffe style: BGR order, mean-subtracted -> blue channel first = 0-103.939
    assert np.isclose(y[0, 0, 0, 0], -103.939, atol=1e-3) and np.isclose(y[0, 0, 0, 2], 255 - 123.68, atol=1e-3)


def test_unknown_scheme_raises():
    with pytest.raises(ValueError):
        apply_preprocessing(np.zeros((1, 2, 2, 3)), "nope")
