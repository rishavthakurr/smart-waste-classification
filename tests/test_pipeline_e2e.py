"""End-to-end pipeline test on SYNTHETIC images (tests mechanics only; results are meaningless).

Runs the real CLI scripts in subprocesses against a temporary config: prepare -> train (CNN and a
transfer model including the fine-tune phase) -> evaluate -> check every output file and the stale-split guard.
"""
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image

from helpers import make_sandbox, run_module

ROOT = Path(__file__).resolve().parents[1]
_run = run_module


@pytest.fixture(scope="module")
def sandbox(tmp_path_factory):
    return make_sandbox(tmp_path_factory.mktemp("e2e"))


def test_full_pipeline(sandbox):
    tmp, env = sandbox
    _run(["training.prepare_dataset"], env)
    man = pd.read_csv(tmp / "ds" / "manifest.csv")
    assert set(man["split"]) == {"train", "validation", "test"} and len(man) == 84
    assert (tmp / "results" / "dataset_report.json").exists() and (tmp / "results" / "dataset_inspection.json").exists()

    _run(["training.train_cnn", "--smoke", "--cpu"], env)
    _run(["training.train_mobilenet", "--smoke", "--cpu", "--no-pretrained", "--debug-finetune-random-init"], env)
    meta = json.loads((tmp / "results" / "mobilenetv2" / "meta.json").read_text())
    assert meta["smoke_test"] and meta["fine_tuned"] and meta["pretrained_imagenet"] is False
    hist = pd.read_csv(tmp / "results" / "mobilenetv2" / "history.csv")
    assert list(hist["phase"]) == ["head", "fine_tune"]
    for k in ("custom_cnn", "mobilenetv2"):
        assert (tmp / "models" / f"{k}.keras").exists()

    _run(["training.evaluate_models"], env)
    res = tmp / "results"
    summ = pd.read_csv(res / "model_comparison.csv")
    assert set(summ["model_key"]) == {"cnn", "mobilenetv2"} and summ["smoke_test"].all()
    assert ((summ["test_accuracy"] >= 0) & (summ["test_accuracy"] <= 1)).all()
    ev = json.loads((res / "evaluation_summary.json").read_text())
    assert ev["test_images"] == int((man["split"] == "test").sum())
    m = json.loads((res / "cnn" / "metrics.json").read_text())
    assert np.array(m["confusion_matrix"]).sum() == ev["test_images"]
    preds = pd.read_csv(res / "cnn" / "test_predictions.csv")
    assert len(preds) == ev["test_images"] and np.allclose(preds.filter(like="prob_").sum(axis=1), 1, atol=1e-4)

    # stale-split guard: changing the split must make evaluation refuse the old models
    manifest = tmp / "ds" / "manifest.csv"
    manifest.write_text(manifest.read_text() + "\n")
    r = _run(["training.evaluate_models"], env, expect_ok=False)
    assert r.returncode != 0 and "different dataset split" in r.stdout


def test_prepare_aborts_when_a_class_is_missing(sandbox, tmp_path):
    tmp, env = sandbox
    import shutil
    raw = tmp_path / "raw"; shutil.copytree(tmp / "raw", raw); shutil.rmtree(raw / "ds" / "glass")
    cfg = yaml.safe_load(Path(env["SWC_CONFIG"]).read_text())
    cfg["dataset"].update(raw_dir=str(raw), prepared_dir=str(tmp_path / "ds"))
    cfg["paths"]["results_dir"] = str(tmp_path / "res")
    p = tmp_path / "c.yaml"; p.write_text(yaml.safe_dump(cfg))
    r = _run(["training.prepare_dataset"], {**env, "SWC_CONFIG": str(p)}, expect_ok=False)
    assert r.returncode != 0 and "DATASET CHECK FAILED" in (r.stdout + r.stderr) and "glass" in (r.stdout + r.stderr)
    assert not (tmp_path / "ds" / "manifest.csv").exists()               # nothing written
