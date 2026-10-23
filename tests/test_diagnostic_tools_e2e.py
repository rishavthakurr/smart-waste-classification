"""End-to-end test of the diagnosis / real-world / comparison / promotion tooling on SYNTHETIC data (mechanics only)."""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image

from helpers import ROOT, make_sandbox, run_module


@pytest.fixture(scope="module")
def box(tmp_path_factory):
    tmp, env = make_sandbox(tmp_path_factory.mktemp("diag"))
    run_module(["training.prepare_dataset"], env)
    run_module(["training.train_cnn", "--smoke", "--cpu", "--tag", "improved"], env)
    run_module(["training.evaluate_models", "--tag", "improved"], env)
    return tmp, env


def test_tagged_run_writes_only_under_its_tag(box):
    tmp, _ = box
    assert (tmp / "models" / "improved" / "custom_cnn.keras").exists()
    assert not (tmp / "models" / "custom_cnn.keras").exists()                      # default location untouched
    meta = json.loads((tmp / "results" / "improved" / "cnn" / "meta.json").read_text())
    assert meta["tag"] == "improved" and "weights_source" in meta and meta["photometric_augmentation"] is False
    assert (tmp / "results" / "improved" / "model_comparison.csv").exists()


def test_diagnose_verifies_label_mapping_and_writes_all_outputs(box):
    tmp, env = box
    run_module(["training.diagnose", "--tag", "improved", "--models", "cnn"], env)
    d = tmp / "results" / "diagnostics" / "improved" / "cnn"
    check = json.loads((d / "label_check.json").read_text())
    assert check["consistent"] and check["loader_indices_match_config"] and check["predictor_index_to_name_matches_config"]
    assert check["loader_class_to_index"]["general_waste"] == 5
    s = json.loads((d / "summary.json").read_text())
    assert 0 <= s["clean_train_accuracy"] <= 1 and "flags (heuristic, thresholds above)" in s
    probes = pd.read_csv(d / "probes.csv")
    assert {"clean", "border_only (object area grayed out)", "rotate 90 deg"} <= set(probes["probe"])
    mc = pd.read_csv(d / "misclassified.csv")
    assert (mc["true_class"] != mc["predicted_class"]).all()
    assert json.loads((d / "metrics.json").read_text())["test"]["n"] == len(pd.read_csv(tmp / "ds" / "manifest.csv").query("split == 'test'"))


def test_diagnose_detects_a_corrupted_class_order(box, tmp_path):
    tmp, env = box
    meta_path = tmp / "results" / "improved" / "cnn" / "meta.json"; orig = meta_path.read_text()
    meta = json.loads(orig); meta["classes"] = list(reversed(meta["classes"])); meta_path.write_text(json.dumps(meta))
    stale = tmp / "results" / "diagnostics" / "improved" / "cnn" / "label_check.json"; stale.unlink(missing_ok=True)
    try:
        r = run_module(["training.diagnose", "--tag", "improved", "--models", "cnn"], env, expect_ok=False)
        chk = json.loads((tmp / "results" / "diagnostics" / "improved" / "cnn" / "label_check.json").read_text())
        assert chk["consistent"] is False and chk["metadata_matches_config"] is False and "different class order" in chk["error"]
    finally:
        meta_path.write_text(orig)


def _rw_dir(tmp, labelled=True):
    rw = tmp / "rw"; rng = np.random.default_rng(5)
    for cls in (["plastic", "glass"] if labelled else []):
        (rw / cls).mkdir(parents=True, exist_ok=True)
        for i in range(3):
            Image.fromarray(rng.integers(0, 255, (80, 100, 3), dtype=np.uint8)).save(rw / cls / f"{i}.jpg")
    (rw / "unlabeled").mkdir(parents=True, exist_ok=True)
    Image.fromarray(rng.integers(0, 255, (80, 100, 3), dtype=np.uint8)).save(rw / "unlabeled" / "u.jpg")
    return rw


def test_real_world_accuracy_uses_only_labelled_images(box):
    tmp, env = box; rw = _rw_dir(tmp)
    run_module(["training.evaluate_real_world", "--model", "cnn", "--tag", "improved", "--dir", str(rw)], env)
    m = json.loads((tmp / "results" / "improved" / "real_world" / "cnn" / "metrics.json").read_text())
    assert m["labelled_images"] == 6 and m["unlabeled_images"] == 1 and "accuracy" in m
    assert "distribution_shift_vs_dataset_test_split" in m
    preds = pd.read_csv(tmp / "results" / "improved" / "real_world" / "cnn" / "predictions.csv")
    assert preds[preds["file"].str.startswith("unlabeled")]["true_class"].isna().all()


def test_real_world_unlabeled_only_computes_no_accuracy(box, tmp_path):
    tmp, env = box; rw = tmp_path / "rw2"; (rw / "unlabeled").mkdir(parents=True)
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(rw / "unlabeled" / "a.png")
    run_module(["training.evaluate_real_world", "--model", "cnn", "--tag", "improved", "--dir", str(rw)], env)
    m = json.loads((tmp / "results" / "improved" / "real_world" / "cnn" / "metrics.json").read_text())
    assert "accuracy" not in m and m["labelled_images"] == 0


def test_real_world_rejects_unknown_folder_names_and_warns_on_reuse(box, tmp_path):
    tmp, env = box; rw = tmp_path / "rw3"; (rw / "organic").mkdir(parents=True)
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(rw / "organic" / "a.png")
    r = run_module(["training.evaluate_real_world", "--model", "cnn", "--tag", "improved", "--dir", str(rw)], env, expect_ok=False)
    assert r.returncode != 0 and "never guessed" in (r.stdout + r.stderr)
    rw2 = _rw_dir(tmp_path)                                           # reuse guard after 3 evaluations of the same set
    outs = [run_module(["training.evaluate_real_world", "--model", "cnn", "--tag", "improved", "--dir", str(rw2)], env).stdout for _ in range(3)]
    assert "no longer an independent test set" in outs[-1] and "no longer an independent" not in outs[0]


def test_compare_and_promote(box):
    tmp, env = box
    r = run_module(["scripts.compare_runs", "--tags", "improved"], env) if False else None
    import subprocess, sys
    p = subprocess.run([sys.executable, str(ROOT / "scripts" / "compare_runs.py"), "--tags", "improved"], cwd=ROOT, env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-800:]
    df = pd.read_csv(tmp / "results" / "comparison_runs.csv")
    assert {"accuracy_ci95_low", "accuracy_ci95_high", "recall_glass", "cfg_pretrained_imagenet", "cfg_image_size"} <= set(df.columns)
    assert (df["accuracy_ci95_low"] <= df["accuracy"]).all() and (df["accuracy"] <= df["accuracy_ci95_high"]).all()
    # promote: live model appears, and an existing live model is preserved under models/original
    (tmp / "models" / "custom_cnn.keras").write_bytes(b"previous-live-model")
    q = subprocess.run([sys.executable, str(ROOT / "scripts" / "promote_model.py"), "--tag", "improved", "--model", "cnn"], cwd=ROOT, env=env, capture_output=True, text=True)
    assert q.returncode == 0, q.stderr[-800:]
    assert (tmp / "models" / "original" / "custom_cnn.keras").read_bytes() == b"previous-live-model"
    assert (tmp / "models" / "custom_cnn.keras").stat().st_size > 1000
