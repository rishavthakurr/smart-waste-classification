"""Regression tests for defects found during the pipeline audit."""
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.preprocessing import apply_preprocessing, load_resized, prepare_input, resize_image
from src.utils import load_config
from training.prepare_dataset import inspect_raw_dataset, verify_prepared


def _write_class_dirs(root, counts):
    rng = np.random.default_rng(1)
    for name, n in counts.items():
        (root / name).mkdir(parents=True)
        for i in range(n):
            Image.fromarray(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)).save(root / name / f"{i}.jpg")


def test_inspection_fails_when_a_project_class_has_no_data(tmp_path):
    _write_class_dirs(tmp_path, {"plastic": 25, "paper": 25, "cardboard": 25, "metal": 25, "trash": 25})  # no glass
    rep = inspect_raw_dataset(tmp_path, load_config())
    assert any("glass" in p for p in rep["problems"]) and rep["usable_files_per_class"]["glass"] == 0


def test_inspection_passes_and_reports_unmapped_folders(tmp_path):
    _write_class_dirs(tmp_path, {c: 25 for c in ["plastic", "paper", "cardboard", "glass", "metal", "trash", "extra"]})
    rep = inspect_raw_dataset(tmp_path, load_config())
    assert rep["problems"] == [] and rep["unmapped_folders"] == ["extra"]
    assert rep["usable_files_per_class"]["general_waste"] == 25


def _manifest(tmp_path, splits):
    rows = []
    for i, (split, cluster) in enumerate(splits):
        rel = f"{split}/plastic/{i}.jpg"
        (tmp_path / split / "plastic").mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(b"x")
        rows.append({"path": rel, "class": "plastic", "split": split, "cluster": cluster, "md5": f"m{i}"})
    return pd.DataFrame(rows)


def test_verify_detects_stale_files_and_cluster_leakage(tmp_path):
    m = _manifest(tmp_path, [("train", 1), ("validation", 2), ("test", 3)])
    (tmp_path / "test" / "plastic" / "stale.jpg").write_bytes(b"x")                 # stale file not in manifest
    probs = verify_prepared(m, tmp_path)
    assert any("differ from manifest" in p for p in probs)
    leak = _manifest(tmp_path / "b", [("train", 1), ("test", 1)])                  # same cluster in two splits
    assert any("cluster spans" in p for p in verify_prepared(leak, tmp_path / "b"))


@pytest.mark.parametrize("scheme", ["rescale_01", "mobilenet_v2", "resnet50"])
def test_training_and_app_preprocessing_are_identical(tmp_path, scheme):
    """Defect: training used TF resize, the app used PIL (mean diff 1.7/255, max 71)."""
    rng = np.random.default_rng(3)
    f = tmp_path / "a.jpg"
    Image.fromarray(rng.integers(0, 255, (300, 400, 3), dtype=np.uint8)).save(f)
    from training.common import load_arrays
    train_side = apply_preprocessing(load_arrays(pd.DataFrame({"abs_path": [str(f)]}), 64).astype(np.float32), scheme)
    app_side = prepare_input(Image.open(f), (64, 64), scheme)
    assert np.array_equal(np.asarray(train_side), app_side)
    assert np.array_equal(load_resized(f, (64, 64)).astype(np.float32), resize_image(Image.open(f), (64, 64)))
