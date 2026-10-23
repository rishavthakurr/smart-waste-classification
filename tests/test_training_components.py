"""Mechanics tests using small randomly-initialised models (no pretrained weights, no real results)."""
import numpy as np
import pandas as pd
import pytest

from src.gradcam import GradCAMNotSupported, compute_gradcam, find_target_layer, overlay_heatmap
from training.models import build_custom_cnn, build_transfer_model, unfreeze_for_fine_tuning
from training.common import compute_class_weights
from training.prepare_dataset import cluster_near_duplicates, grouped_stratified_split, thumbnail
from PIL import Image


def test_custom_cnn_output_is_six_class_softmax():
    m = build_custom_cnn((48, 48, 3), 6)
    out = m(np.random.rand(2, 48, 48, 3).astype("float32"), training=False).numpy()
    assert out.shape == (2, 6) and np.allclose(out.sum(1), 1, atol=1e-5)


def test_gradcam_on_custom_cnn():
    m = build_custom_cnn((48, 48, 3), 6)
    assert find_target_layer(m) == "last_conv_activation"
    cam, idx = compute_gradcam(m, np.random.rand(1, 48, 48, 3).astype("float32"))
    assert cam.ndim == 2 and 0 <= idx < 6 and np.isfinite(cam).all() and 0 <= cam.min() <= cam.max() <= 1
    colored, overlay = overlay_heatmap(Image.new("RGB", (80, 60)), cam)
    assert overlay.shape == (60, 80, 3) and colored.shape == (60, 80, 3)


def test_gradcam_on_nested_backbone_and_finetune_unfreeze():
    m, loaded, src = build_transfer_model("mobilenetv2", (64, 64, 3), 6, use_pretrained=False)
    assert loaded is False and "explicit" in src and find_target_layer(m) == "backbone"
    cam, _ = compute_gradcam(m, np.random.rand(1, 64, 64, 3).astype("float32"))
    assert cam.ndim == 2
    n = unfreeze_for_fine_tuning(m, 20)
    bb = m.get_layer("backbone")
    assert n > 0 and all(not l.trainable for l in bb.layers[:-20])
    import keras
    assert all(not l.trainable for l in bb.layers if isinstance(l, keras.layers.BatchNormalization))


def test_gradcam_not_supported_without_conv_layers():
    import keras
    m = keras.Sequential([keras.layers.Input((8,)), keras.layers.Dense(6, activation="softmax")])
    with pytest.raises(GradCAMNotSupported):
        find_target_layer(m)


def test_class_weights_favour_minority():
    w = compute_class_weights(np.array([0] * 90 + [1] * 10), 2)
    assert w[1] > w[0] and abs(w[0] * 90 + w[1] * 10 - 100) < 1e-6


def test_thumbnail_clustering_groups_same_photo_but_not_flat_lookalikes():
    rng = np.random.default_rng(0)
    base = Image.fromarray(rng.integers(0, 255, (120, 160, 3), dtype=np.uint8))
    near = base.resize((150, 110))                                        # same photo, re-sized
    other = Image.fromarray(rng.integers(0, 255, (120, 160, 3), dtype=np.uint8))
    flat_a = Image.new("RGB", (100, 100), (200, 150, 100))               # brown flat
    flat_b = Image.new("RGB", (100, 100), (240, 240, 240))               # white flat (dHash used to merge these)
    df = pd.DataFrame({"class": ["a", "a", "a", "a", "a"],
                       "thumb": [thumbnail(i) for i in (base, near, other, flat_a, flat_b)]})
    cl, cross = cluster_near_duplicates(df, 4.5)
    assert cl[0] == cl[1] and len({cl[0], cl[2], cl[3], cl[4]}) == 4 and cross == 0


def test_split_keeps_duplicate_clusters_together_and_stratified():
    rows = [{"class": c, "cluster": f"{c}{i // 3}", "path": f"{c}{i}"} for c in ("a", "b") for i in range(60)]
    df = pd.DataFrame(rows)
    df["split"] = grouped_stratified_split(df, {"train": .7, "validation": .15, "test": .15}, seed=1)
    assert df.groupby("cluster")["split"].nunique().max() == 1          # no leakage across splits
    for c in ("a", "b"):
        frac = (df[df["class"] == c]["split"] == "train").mean()
        assert 0.6 <= frac <= 0.8
    again = grouped_stratified_split(df, {"train": .7, "validation": .15, "test": .15}, seed=1)
    assert (again == df["split"]).all()                                  # reproducible
