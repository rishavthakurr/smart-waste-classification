"""Grad-CAM for the project's linear Keras models (custom CNN, MobileNetV2, ResNet50).

Walks the model layer-by-layer so the gradient can be taken w.r.t. the last
convolutional feature map even when the backbone is a nested model.
Grad-CAM shows regions that influenced the score of a class; it does not prove correctness.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


class GradCAMNotSupported(RuntimeError):
    """The model has no convolutional feature map Grad-CAM can use."""


def find_target_layer(model) -> str:
    names = [l.name for l in model.layers]
    for preferred in ("last_conv_activation", "backbone"):
        if preferred in names:
            return preferred
    import keras
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            return layer.name
    raise GradCAMNotSupported("This model has no convolutional layer, so Grad-CAM is not applicable.")


def compute_gradcam(model, batch: np.ndarray, class_index: int | None = None) -> tuple[np.ndarray, int]:
    """Return (heatmap in [0,1] shaped (h, w), class_index used) for a preprocessed (1,H,W,3) batch."""
    import keras
    import tensorflow as tf
    target = find_target_layer(model)
    x = tf.convert_to_tensor(batch, dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_out, preds = None, None
        for layer in model.layers:
            if isinstance(layer, keras.layers.InputLayer):
                continue
            x = layer(x, training=False)
            if layer.name == target:
                conv_out = x
                tape.watch(conv_out)
        preds = x
        if conv_out is None:
            raise GradCAMNotSupported("Target convolutional layer was not reached.")
        if class_index is None:
            class_index = int(tf.argmax(preds[0]))
        score = preds[:, class_index]
    grads = tape.gradient(score, conv_out)
    if grads is None:
        raise GradCAMNotSupported("Gradients could not be computed for this model.")
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(weights * conv_out, axis=-1))[0].numpy()
    cam = cam / cam.max() if cam.max() > 0 else cam
    return cam, class_index


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> tuple[np.ndarray, np.ndarray]:
    """Return (colored heatmap RGB, overlay RGB), both at the original image size."""
    rgb = np.asarray(image.convert("RGB"))
    h, w = rgb.shape[:2]
    resized = cv2.resize(heatmap.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    colored = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * np.clip(resized, 0, 1)), cv2.COLORMAP_JET),
                           cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(rgb, 1 - alpha, colored, alpha, 0)
    return colored, overlay
