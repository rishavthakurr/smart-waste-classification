"""Shared training / data-loading logic used by train_*.py and evaluate_models.py.

Key guarantees
 * Images are loaded and resized by src.preprocessing.load_resized - the SAME code the web app uses.
 * Model-specific scaling is applied by src.preprocessing.apply_preprocessing in both places.
 * Every trained model records the sha256 of dataset/manifest.csv it was trained on, so evaluation
   can refuse to run against a different (possibly leaking) split.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.preprocessing import apply_preprocessing, load_resized
from src.utils import (class_names, load_config, model_path, models_dir, resolve, results_dir,
                       run_results_dir, write_json)


# ----------------------------------------------------------------------------- data
def manifest_path() -> Path:
    return resolve(load_config()["dataset"]["prepared_dir"]) / "manifest.csv"


def manifest_sha256() -> str:
    return hashlib.sha256(manifest_path().read_bytes()).hexdigest()


def load_manifest() -> pd.DataFrame:
    path = manifest_path()
    if not path.exists():
        raise SystemExit(f"{path} not found. Run: python -m training.prepare_dataset --download")
    return pd.read_csv(path)


def split_frame(split: str, max_per_class: int | None = None, seed: int = 42) -> pd.DataFrame:
    df = load_manifest()
    df = df[df["split"] == split].copy()
    if df.empty:
        raise SystemExit(f"Split '{split}' is empty in the manifest.")
    if max_per_class:
        df = pd.concat([g.sample(min(len(g), max_per_class), random_state=seed) for _, g in df.groupby("class")])
    root = resolve(load_config()["dataset"]["prepared_dir"])
    df["abs_path"] = df["path"].map(lambda p: str(root / p))
    df["label"] = df["class"].map({c: i for i, c in enumerate(class_names())})
    if df["label"].isna().any():
        raise SystemExit("Manifest contains classes that are not in config.yaml 'classes'.")
    return df.sort_values("path").reset_index(drop=True)      # fixed order => reproducible


def load_arrays(df: pd.DataFrame, image_size: int) -> np.ndarray:
    """Load all images of a split as uint8 (N, S, S, 3) via the shared PIL loader."""
    return np.stack([load_resized(p, (image_size, image_size)) for p in df["abs_path"]])


def make_dataset(df: pd.DataFrame, image_size: int, scheme: str, batch_size: int, shuffle: bool, seed: int,
                 photometric: bool = False):
    """tf.data pipeline: PIL-loaded uint8 arrays -> (shuffle) -> [photometric jitter] -> model-specific scaling -> batch.

    `photometric` must only be True for the TRAINING set; it never touches validation/test/inference images.
    """
    import tensorflow as tf
    x = load_arrays(df, image_size)
    y = df["label"].to_numpy(dtype="int32")
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(len(df), seed=seed, reshuffle_each_iteration=True)
    def jitter(img):                       # operates on 0-255 pixels, BEFORE model-specific scaling
        img = tf.image.random_brightness(img, 25.0)
        img = tf.image.random_contrast(img, 0.75, 1.25)
        img = tf.image.random_saturation(img, 0.7, 1.3)
        return tf.clip_by_value(img, 0.0, 255.0)

    def prep(img, lab):
        img = tf.cast(img, tf.float32)
        if photometric:
            img = jitter(img)
        return apply_preprocessing(img, scheme), lab

    ds = ds.map(prep, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def compute_class_weights(labels: np.ndarray, n_classes: int) -> dict[int, float]:
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    return {i: float(len(labels) / (n_classes * max(c, 1))) for i, c in enumerate(counts)}


# ----------------------------------------------------------------------------- CLI
def parse_args(model_key: str) -> argparse.Namespace:
    cfg = load_config()["training"]
    ap = argparse.ArgumentParser(description=f"Train {model_key}")
    ap.add_argument("--epochs", type=int, default=cfg["epochs"])
    ap.add_argument("--batch-size", type=int, default=cfg["batch_size"])
    ap.add_argument("--lr", type=float, default=cfg["learning_rate"])
    ap.add_argument("--image-size", type=int, default=None, help="square input size (default from config)")
    ap.add_argument("--fine-tune-epochs", type=int, default=cfg["fine_tune"]["epochs"])
    ap.add_argument("--fine-tune-lr", type=float, default=cfg["fine_tune"]["learning_rate"])
    ap.add_argument("--no-fine-tune", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true", help="deliberately train from random weights")
    ap.add_argument("--allow-random-init", action="store_true",
                    help="if ImageNet weights cannot be loaded, continue from random weights instead of stopping")
    ap.add_argument("--photometric-aug", action="store_true", default=cfg.get("photometric_augmentation", False),
                    help="brightness/contrast/saturation jitter on TRAINING images only")
    ap.add_argument("--tag", default="", help="save under models/<tag>/ and results/<tag>/ (e.g. original, improved)")
    ap.add_argument("--cpu", action="store_true", help="force CPU even if a GPU is present")
    ap.add_argument("--deterministic", action="store_true",
                    help="enable TensorFlow op determinism (bit-for-bit repeatable, slower on GPU)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run to verify the pipeline; results are NOT meaningful")
    ap.add_argument("--debug-finetune-random-init", action="store_true",
                    help="TESTING ONLY: run the fine-tune phase even without pretrained weights")
    return ap.parse_args()


def run_cli(model_key: str) -> None:
    run_training(model_key, parse_args(model_key))


# ----------------------------------------------------------------------------- training
def run_training(model_key: str, args: argparse.Namespace) -> dict[str, Any]:
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import keras
    import tensorflow as tf
    from training.models import build_custom_cnn, build_transfer_model, unfreeze_for_fine_tuning

    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)     # avoid grabbing all GPU memory
        except RuntimeError:
            pass
    if args.deterministic:
        tf.config.experimental.enable_op_determinism()

    cfg = load_config(); tcfg = cfg["training"]; mcfg = cfg["models"][model_key]
    seed, scheme, classes = tcfg["seed"], mcfg["preprocessing"], class_names()
    keras.utils.set_random_seed(seed)

    epochs, batch, size = args.epochs, args.batch_size, args.image_size or cfg["image"]["height"]
    ft_epochs, max_per_class = args.fine_tune_epochs, None
    if args.smoke:
        sm = tcfg["smoke"]
        epochs, batch, size, ft_epochs, max_per_class = (sm["epochs"], sm["batch_size"], sm["image_size"],
                                                         sm["fine_tune_epochs"], sm["max_train_per_class"])
        print("*** SMOKE MODE: tiny subset, low resolution. Do not report these numbers. ***")
    print("GPUs visible:", gpus or "none (CPU)")

    train_df = split_frame("train", max_per_class, seed)
    val_df = split_frame("validation", max_per_class, seed)
    print(f"Loading images ({len(train_df)} train, {len(val_df)} validation) at {size}x{size} ...")
    train_ds = make_dataset(train_df, size, scheme, batch, True, seed, photometric=args.photometric_aug)
    val_ds = make_dataset(val_df, size, scheme, batch, False, seed)
    class_weight = compute_class_weights(train_df["label"].values, len(classes)) if tcfg["use_class_weights"] else None
    print(f"batch={batch} preprocessing={scheme} class_weight={ {k: round(v, 2) for k, v in (class_weight or {}).items()} }")

    shape = (size, size, 3)
    pretrained = False
    if mcfg["transfer"]:
        model, pretrained, weights_source = build_transfer_model(
            model_key, shape, len(classes), tcfg["dropout"], use_pretrained=not args.no_pretrained,
            allow_random_init=args.allow_random_init)
        print("Backbone weights:", weights_source)
    else:
        model = build_custom_cnn(shape, len(classes), tcfg["dropout"]); weights_source = "random init (custom CNN)"

    tag = args.tag or None
    out_path = model_path(model_key, tag); out_path.parent.mkdir(parents=True, exist_ok=True)
    res = run_results_dir(tag) / model_key; res.mkdir(parents=True, exist_ok=True)

    def callbacks(threshold: float | None = None):
        return [
            keras.callbacks.ModelCheckpoint(str(out_path), monitor="val_accuracy", mode="max",
                                            save_best_only=True, initial_value_threshold=threshold),
            keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", restore_best_weights=True,
                                          patience=tcfg["early_stopping_patience"]),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=tcfg["reduce_lr_factor"],
                                              patience=tcfg["reduce_lr_patience"], min_lr=tcfg["min_lr"]),
        ]

    def compile_(lr: float):
        model.compile(optimizer=keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])

    t0 = time.time()
    compile_(args.lr)
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=epochs, class_weight=class_weight,
                   callbacks=callbacks(), shuffle=False, verbose=2)
    hist = pd.DataFrame(h1.history); hist["phase"] = "head" if mcfg["transfer"] else "full"
    best_phase1 = float(hist["val_accuracy"].max())

    can_ft = mcfg["transfer"] and tcfg["fine_tune"]["enabled"] and not args.no_fine_tune and ft_epochs > 0
    fine_tuned = False
    if can_ft and (pretrained or args.debug_finetune_random_init):
        if not pretrained:
            print("NOTE: fine-tuning WITHOUT pretrained weights (debug flag) - for pipeline testing only.")
        n = unfreeze_for_fine_tuning(model, tcfg["fine_tune"]["unfreeze_last_layers"])
        print(f"Fine-tuning: {n} backbone layers unfrozen (BatchNorm kept frozen)")
        compile_(args.fine_tune_lr)
        h2 = model.fit(train_ds, validation_data=val_ds, epochs=ft_epochs, class_weight=class_weight,
                       callbacks=callbacks(best_phase1), shuffle=False, verbose=2)
        h2df = pd.DataFrame(h2.history); h2df["phase"] = "fine_tune"
        hist = pd.concat([hist, h2df], ignore_index=True); fine_tuned = True
    elif can_ft:
        print("Fine-tuning skipped: it requires pretrained ImageNet weights (not loaded).")
    hist.insert(0, "epoch", range(1, len(hist) + 1))
    hist.to_csv(res / "history.csv", index=False)

    if not out_path.exists():            # validation accuracy never beat the threshold
        model.save(out_path)
    best = float(hist["val_accuracy"].max())
    meta = {
        "model_key": model_key, "display_name": mcfg["display_name"], "classes": classes,
        "preprocessing": scheme, "input_size": size, "pretrained_imagenet": pretrained,
        "weights_source": weights_source, "photometric_augmentation": bool(args.photometric_aug), "tag": args.tag,
        "fine_tuned": fine_tuned, "smoke_test": bool(args.smoke), "deterministic_mode": bool(args.deterministic),
        "seed": seed, "batch_size": batch, "learning_rate": args.lr, "epochs_requested": epochs,
        "epochs_run": int(len(hist)), "best_val_accuracy": best,
        "best_val_epoch": int(hist["val_accuracy"].idxmax()) + 1,
        "train_images": int(len(train_df)), "val_images": int(len(val_df)),
        "class_weights": class_weight, "dropout": tcfg["dropout"], "fine_tune_lr": args.fine_tune_lr,
        "fine_tune_unfrozen_layers": tcfg["fine_tune"]["unfreeze_last_layers"], "manifest_sha256": manifest_sha256(),
        "training_seconds": round(time.time() - t0, 1), "cpu_only": not gpus,
        "tensorflow": tf.__version__, "keras": keras.__version__, "python": platform.python_version(),
        "command": " ".join(sys.argv), "gradcam_supported": True,
    }
    write_json(res / "meta.json", meta)
    plot_history(hist, res / "training_curves.png", mcfg["display_name"])
    print("\nSaved model:", out_path, "\nSaved results:", res)
    print(f"Best validation accuracy: {best:.4f} at epoch {meta['best_val_epoch']}  "
          f"(pretrained={pretrained}, fine_tuned={fine_tuned}, smoke={args.smoke})")
    return meta


def plot_history(hist: pd.DataFrame, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ft = hist.index[hist["phase"] == "fine_tune"]
    for a, (tr, va, name) in zip(ax, [("accuracy", "val_accuracy", "Accuracy"), ("loss", "val_loss", "Loss")]):
        a.plot(hist["epoch"], hist[tr], label=f"Train {name.lower()}", color="#1B5E20")
        a.plot(hist["epoch"], hist[va], label=f"Validation {name.lower()}", color="#2F80ED")
        if len(ft):
            a.axvline(hist["epoch"].iloc[ft[0]] - 0.5, ls="--", color="gray", label="Fine-tune start")
        a.set_xlabel("Epoch"); a.set_ylabel(name); a.set_title(f"{title}: {name}"); a.legend(); a.grid(alpha=.3)
    plt.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
