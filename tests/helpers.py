"""Shared builder for the synthetic-data sandbox used by end-to-end tests (mechanics only)."""
import copy
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TINTS = {"plastic": (200, 60, 60), "paper": (60, 200, 60), "cardboard": (60, 60, 200),
         "glass": (200, 200, 60), "metal": (60, 200, 200), "trash": (200, 60, 200)}


def run_module(args, env, expect_ok=True, timeout=600):
    r = subprocess.run([sys.executable, "-m", *args], cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)
    if expect_ok:
        assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    return r


def make_sandbox(tmp: Path):
    """Create synthetic raw images + a temp config. Returns (tmp, env)."""
    rng = np.random.default_rng(0)
    for cls, tint in TINTS.items():
        d = tmp / "raw" / "ds" / cls; d.mkdir(parents=True)
        for i in range(14):
            arr = np.clip(rng.normal(tint, 40, (64, 64, 3)), 0, 255).astype(np.uint8)
            Image.fromarray(arr).save(d / f"{cls}_{i}.jpg")
    cfg = copy.deepcopy(yaml.safe_load((ROOT / "config.yaml").read_text()))
    cfg["dataset"].update(raw_dir=str(tmp / "raw"), prepared_dir=str(tmp / "ds"), min_images_per_class=5)
    cfg["paths"].update(models_dir=str(tmp / "models"), results_dir=str(tmp / "results"), database=str(tmp / "h.db"))
    for m in cfg["models"].values():
        m["file"] = str(tmp / "models" / Path(m["file"]).name)
    cfg["training"]["smoke"].update(image_size=32, epochs=1, fine_tune_epochs=1, batch_size=8, max_train_per_class=6)
    cfg_path = tmp / "config.yaml"; cfg_path.write_text(yaml.safe_dump(cfg))
    env = {**os.environ, "SWC_CONFIG": str(cfg_path), "TF_CPP_MIN_LOG_LEVEL": "3", "PYTHONPATH": str(ROOT)}
    return tmp, env
