"""Shared helpers: project paths and configuration loading."""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> dict[str, Any]:
    """Load config.yaml once and validate the essentials."""
    cfg_path = Path(path or os.environ.get("SWC_CONFIG") or CONFIG_PATH)
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    classes = cfg["classes"]
    if len(set(classes)) != len(classes):
        raise ValueError("config.yaml: duplicate class names")
    unknown = set(cfg["dataset"]["label_map"].values()) - set(classes)
    if unknown:
        raise ValueError(f"config.yaml: label_map targets unknown classes: {unknown}")
    return cfg


def class_names() -> list[str]:
    return list(load_config()["classes"])


def pretty(name: str) -> str:
    """'general_waste' -> 'General Waste'."""
    return name.replace("_", " ").title()


def resolve(rel: str | Path) -> Path:
    """Resolve a config-relative path against the project root."""
    p = Path(rel)
    return p if p.is_absolute() else PROJECT_ROOT / p


def results_dir() -> Path:
    d = resolve(load_config()["paths"]["results_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d


def models_dir() -> Path:
    d = resolve(load_config()["paths"]["models_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d


def model_path(model_key: str, tag: str | None = None) -> Path:
    """models/<file>; with a tag: models/<tag>/<file> (e.g. 'original', 'improved') so runs never overwrite each other."""
    p = resolve(load_config()["models"][model_key]["file"])
    return p.parent / tag / p.name if tag else p


def run_results_dir(tag: str | None = None) -> Path:
    """results/ or results/<tag>/ (tagged runs keep their own metrics, histories and comparison tables)."""
    d = results_dir() / tag if tag else results_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


def model_display_name(model_key: str) -> str:
    return load_config()["models"][model_key]["display_name"]


def read_json(path: Path) -> dict | list | None:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)
