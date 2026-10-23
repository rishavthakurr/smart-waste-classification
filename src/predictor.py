"""Inference: PIL image -> structured prediction result."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

from src.model_loader import input_size, load_model
from src.preprocessing import prepare_input
from src.utils import class_names, load_config

import time


class PredictionError(RuntimeError):
    """Inference failed for a reason that can be shown to the user."""


@dataclass
class PredictionResult:
    model_key: str
    predicted_class: str
    confidence: float                       # softmax probability of predicted class (not a guarantee)
    probabilities: dict[str, float]         # all classes
    top3: list[tuple[str, float]] = field(default_factory=list)
    inference_ms: float = 0.0
    low_confidence: bool = False

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def result_from_probabilities(probs: np.ndarray, model_key: str, inference_ms: float = 0.0) -> PredictionResult:
    """Pure function: convert a probability vector to a PredictionResult."""
    classes = class_names()
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    if probs.shape[0] != len(classes):
        raise PredictionError(f"Model returned {probs.shape[0]} outputs, expected {len(classes)}.")
    if not np.all(np.isfinite(probs)):
        raise PredictionError("Model returned invalid (non-finite) probabilities.")
    order = np.argsort(probs)[::-1]
    conf = float(probs[order[0]])
    return PredictionResult(
        model_key=model_key, predicted_class=classes[int(order[0])], confidence=conf,
        probabilities={c: float(p) for c, p in zip(classes, probs)},
        top3=[(classes[int(i)], float(probs[i])) for i in order[:3]],
        inference_ms=inference_ms,
        low_confidence=conf < load_config()["ui"]["low_confidence_threshold"])


def predict(image: Image.Image, model_key: str, model=None) -> PredictionResult:
    """Classify one image with the requested model (loads it if `model` is not supplied)."""
    from src.utils import load_config as _lc
    scheme = _lc()["models"][model_key]["preprocessing"]
    model = model if model is not None else load_model(model_key)
    try:
        batch = prepare_input(image, input_size(model), scheme)
        t0 = time.perf_counter()
        probs = np.asarray(model(batch, training=False))[0] if callable(model) else model.predict(batch, verbose=0)[0]
        ms = (time.perf_counter() - t0) * 1000
    except PredictionError:
        raise
    except Exception as exc:
        raise PredictionError(f"Inference failed ({type(exc).__name__}). Please try a different image.") from exc
    return result_from_probabilities(probs, model_key, ms)
