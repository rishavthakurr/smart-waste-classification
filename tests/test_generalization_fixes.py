"""Unit tests for the corrections made while diagnosing poor generalisation."""
import inspect

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import training.common as common
import training.models as M
from src.utils import model_path, run_results_dir


def test_failed_weight_download_stops_training_by_default(monkeypatch):
    def boom():
        raise OSError("offline")
    monkeypatch.setattr(M, "_weight_sources", lambda kind: [("official", boom)])
    with pytest.raises(M.PretrainedWeightsError, match="Could not load pretrained"):
        M.build_transfer_model("mobilenetv2", (32, 32, 3), 6, use_pretrained=True)


def test_random_init_only_when_explicitly_requested(monkeypatch):
    monkeypatch.setattr(M, "_weight_sources", lambda kind: [("official", lambda: (_ for _ in ()).throw(OSError("x")))])
    _, loaded, src = M.build_transfer_model("mobilenetv2", (32, 32, 3), 6, use_pretrained=True, allow_random_init=True)
    assert loaded is False and "FALLBACK" in src
    _, loaded, src = M.build_transfer_model("mobilenetv2", (32, 32, 3), 6, use_pretrained=False)
    assert loaded is False and "explicit" in src


def test_working_source_is_used_and_reported(monkeypatch):
    monkeypatch.setattr(M, "_weight_sources", lambda kind: [("bad", lambda: (_ for _ in ()).throw(OSError("x"))),
                                                           ("good", lambda: None)])        # weights=None == valid keras arg
    _, loaded, src = M.build_transfer_model("mobilenetv2", (32, 32, 3), 6, use_pretrained=True)
    assert src == "good" and loaded is True


def test_mirror_checksum_is_pinned():
    m = M.GITHUB_MIRRORS["mobilenetv2"]
    assert len(m["sha256"]) == 64 and m["url"].startswith("https://github.com/")
    assert "resnet50" not in M.GITHUB_MIRRORS            # legacy ResNet50 file is incompatible; must not be offered


def test_photometric_augmentation_changes_only_when_requested(tmp_path):
    rng = np.random.default_rng(0); paths = []
    for i in range(4):
        f = tmp_path / f"{i}.png"; Image.fromarray(rng.integers(60, 200, (40, 40, 3), dtype=np.uint8)).save(f); paths.append(str(f))
    df = pd.DataFrame({"abs_path": paths, "label": [0, 1, 2, 3]})
    grab = lambda **k: next(iter(common.make_dataset(df, 32, "rescale_01", 4, False, 0, **k)))[0].numpy()
    clean1, clean2, aug = grab(), grab(), grab(photometric=True)
    assert np.array_equal(clean1, clean2)                  # evaluation pipeline is deterministic
    assert not np.array_equal(clean1, aug) and aug.min() >= 0 and aug.max() <= 1.0 + 1e-6


def test_photometric_flag_reaches_only_the_training_dataset():
    src = inspect.getsource(common.run_training)
    assert "train_ds = make_dataset(train_df, size, scheme, batch, True, seed, photometric=args.photometric_aug)" in src
    assert "val_ds = make_dataset(val_df, size, scheme, batch, False, seed)" in src       # no photometric for validation
    ev = open(common.__file__.replace("common.py", "evaluate_models.py")).read()
    assert "photometric" not in ev


def test_tagged_paths_never_collide_with_default(tmp_path, monkeypatch):
    import src.utils as U
    monkeypatch.setattr(U, "results_dir", lambda: tmp_path)          # never create folders in the real results/
    assert model_path("cnn") != model_path("cnn", "improved")
    assert model_path("cnn", "improved").parent.name == "improved" and model_path("cnn", "improved").name == model_path("cnn").name
    assert run_results_dir("improved").name == "improved"
