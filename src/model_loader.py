"""Discover and load trained models. Never trains; never silently substitutes models."""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from src.utils import (class_names, load_config, model_display_name, model_path, read_json,
                       results_dir, run_results_dir)


class ModelNotAvailableError(RuntimeError):
    """The requested model file does not exist (it must be trained first)."""


class ModelLoadError(RuntimeError):
    """The model file exists but could not be loaded or is inconsistent with config."""


def model_meta(model_key: str, tag: str | None = None) -> dict[str, Any]:
    meta = read_json((results_dir() / tag if tag else results_dir()) / model_key / "meta.json")   # read-only: never creates folders
    return meta if isinstance(meta, dict) else {}


def is_available(model_key: str) -> bool:
    return model_path(model_key).is_file()


def list_models() -> dict[str, dict[str, Any]]:
    """Status of every configured model: availability + training metadata."""
    out = {}
    for key, m in load_config()["models"].items():
        meta = model_meta(key)
        out[key] = {"key": key, "display_name": m["display_name"], "available": is_available(key),
                    "smoke_test": bool(meta.get("smoke_test", False)),
                    "pretrained": meta.get("pretrained_imagenet"), "meta": meta}
    return out


def available_models() -> list[str]:
    return [k for k, v in list_models().items() if v["available"]]


def get_deployment_model() -> tuple[str | None, str]:
    """Return (model_key or None, human-readable status)."""
    cfg_choice = load_config().get("deployment_model", "auto")
    avail = available_models()
    if cfg_choice != "auto":
        if cfg_choice in avail:
            return cfg_choice, "configured in config.yaml"
        return None, f"{model_display_name(cfg_choice)} is configured but not trained yet"
    rec = read_json(results_dir() / "recommended_model.json")
    if isinstance(rec, dict) and rec.get("model_key") in avail:
        return rec["model_key"], "recommended by evaluation"
    if avail:
        return avail[0], "evaluation pending (first available model)"
    return None, "no trained model available"


@lru_cache(maxsize=8)
def _load_cached(model_key: str, mtime: float, tag: str | None = None):
    import keras
    try:
        return keras.saving.load_model(str(model_path(model_key, tag)), compile=False)
    except Exception as exc:
        raise ModelLoadError(f"Could not load {model_display_name(model_key)}: {type(exc).__name__}") from exc


def load_model(model_key: str, tag: str | None = None):
    """Load (and cache per file-modification time) a trained Keras model (optionally from models/<tag>/)."""
    if model_key not in load_config()["models"]:
        raise ModelNotAvailableError(f"Unknown model '{model_key}'.")
    path = model_path(model_key, tag)
    if not path.is_file():
        raise ModelNotAvailableError(
            f"{model_display_name(model_key)} has not been trained yet. "
            f"Run: python -m training.train_{ {'cnn':'cnn','mobilenetv2':'mobilenet','resnet50':'resnet'}[model_key] }")
    model = _load_cached(model_key, os.path.getmtime(path), tag)
    meta = model_meta(model_key, tag)
    if meta.get("classes") and meta["classes"] != class_names():
        raise ModelLoadError("This model was trained with a different class order than config.yaml. "
                             "Retrain it or restore the original config.")
    n_out = model.output_shape[-1]
    if n_out != len(class_names()):
        raise ModelLoadError(f"Model outputs {n_out} classes but config defines {len(class_names())}.")
    return model


def input_size(model) -> tuple[int, int]:
    shape = model.input_shape
    return int(shape[1]), int(shape[2])
