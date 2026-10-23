"""Evaluate a model on REAL-WORLD photos you took yourself, and report distribution shift vs. the dataset.

Directory layout (create it with your own photos; nothing is generated or invented):
    real_world/<class>/*.jpg        labelled images, <class> in config.yaml classes (plastic, paper, ...)
    real_world/unlabeled/*.jpg      images without ground truth: predictions + confidence only, NO accuracy
    real_world/conditions.csv       optional: file,condition  (e.g. "dim light", "crumpled", "cluttered background")

Usage: python -m training.evaluate_real_world --model mobilenetv2 [--tag improved] [--dir real_world]

Rules enforced here:
  * accuracy is computed ONLY from labelled folders; unlabeled images never contribute to any metric
  * every run is appended to real_world/.evaluation_log.csv; after 3 evaluations of the same image set a warning
    says it is no longer an independent test set (collect a fresh set instead of tuning to this one)
  * classes with fewer than 10 images are flagged: their recall is statistically unreliable
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.model_loader import load_model
from src.predictor import predict
from src.preprocessing import ImageValidationError, validate_image_bytes
from src.utils import PROJECT_ROOT, class_names, run_results_dir, write_json
from training.common import split_frame

EXT = {".jpg", ".jpeg", ".png"}


def image_stats(img: Image.Image) -> dict:
    a = np.asarray(img.convert("RGB").resize((256, 256))); g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(a, cv2.COLOR_RGB2HSV); b = 24
    border = np.concatenate([g[:b].ravel(), g[-b:].ravel(), g[:, :b].ravel(), g[:, -b:].ravel()])
    return {"brightness": float(g.mean()), "contrast": float(g.std()), "saturation": float(hsv[..., 1].mean()),
            "border_brightness": float(border.mean()), "border_std": float(border.std()),
            "sharpness_lapvar": float(cv2.Laplacian(g, cv2.CV_64F).var()), "aspect_ratio": img.width / img.height}


def shift_report(real: pd.DataFrame, ref: pd.DataFrame) -> dict:
    out = {}
    for c in [c for c in real.columns if c in ref.columns]:
        sd = np.sqrt((real[c].var() + ref[c].var()) / 2) or 1.0
        d = float((real[c].mean() - ref[c].mean()) / sd)
        out[c] = {"dataset_mean": float(ref[c].mean()), "real_world_mean": float(real[c].mean()),
                  "standardised_difference": d, "shift_flag": bool(abs(d) > 0.8)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--tag", default=""); ap.add_argument("--dir", default="real_world")
    args = ap.parse_args(); tag = args.tag or None; classes = class_names()
    root = Path(args.dir) if Path(args.dir).is_absolute() else PROJECT_ROOT / args.dir
    items = []                                              # (path, true_class or None)
    for d in sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []:
        if d.name != "unlabeled" and d.name not in classes:
            raise SystemExit(f"Folder '{d.name}' is neither 'unlabeled' nor one of {classes}. Rename it; labels are never guessed.")
        items += [(f, None if d.name == "unlabeled" else d.name) for f in sorted(d.iterdir()) if f.suffix.lower() in EXT]
    if not items:
        raise SystemExit(f"No images found under {root}. Put photos in {root}/<class>/ (labelled) or {root}/unlabeled/.")
    model = load_model(args.model, tag); rows, stats, bad = [], [], []
    for f, true in items:
        try:
            img = validate_image_bytes(f.read_bytes(), f.name)
        except ImageValidationError as e:
            bad.append((str(f), str(e))); continue
        r = predict(img, args.model, model); st = image_stats(img)
        rows.append({"file": str(f.relative_to(root)), "true_class": true, "predicted_class": r.predicted_class,
                     "confidence": r.confidence, **{f"prob_{c}": p for c, p in r.probabilities.items()}}); stats.append(st)
    df = pd.DataFrame(rows); out = run_results_dir(tag) / "real_world" / args.model; out.mkdir(parents=True, exist_ok=True)
    cond = root / "conditions.csv"
    if cond.exists():
        df = df.merge(pd.read_csv(cond).rename(columns={"file": "file"}), on="file", how="left")
    df.to_csv(out / "predictions.csv", index=False)
    lab = df[df["true_class"].notna()]
    metrics = {"images_total": len(items), "images_unreadable": len(bad), "labelled_images": int(len(lab)),
               "unlabeled_images": int(df["true_class"].isna().sum()), "labelled_per_class": lab["true_class"].value_counts().to_dict()}
    if len(lab):
        y, p = lab["true_class"], lab["predicted_class"]
        metrics.update(accuracy=float(accuracy_score(y, p)),
                       report=classification_report(y, p, labels=classes, output_dict=True, zero_division=0),
                       confusion_matrix=confusion_matrix(y, p, labels=classes).tolist(),
                       unreliable_classes_fewer_than_10_images=[c for c in classes if (y == c).sum() < 10])
        if "condition" in lab:
            metrics["accuracy_by_condition"] = lab.groupby("condition").apply(lambda g: float((g["true_class"] == g["predicted_class"]).mean())).to_dict()
        print(f"Real-world accuracy on {len(lab)} labelled images: {metrics['accuracy']:.4f}")
        if metrics["unreliable_classes_fewer_than_10_images"]:
            print("WARNING: fewer than 10 images for", metrics["unreliable_classes_fewer_than_10_images"], "- treat their recall as unreliable.")
    else:
        print("No labelled images: predictions and confidences saved; accuracy is NOT computed without ground truth.")
    ref = pd.DataFrame([image_stats(Image.open(Path(load_manifest_root()) / p)) for p in (lambda t: t.sample(min(150, len(t)), random_state=0))(split_frame("test")["path"])])
    metrics["distribution_shift_vs_dataset_test_split"] = shift_report(pd.DataFrame(stats), ref)
    write_json(out / "metrics.json", metrics)
    shifted = [k for k, v in metrics["distribution_shift_vs_dataset_test_split"].items() if v["shift_flag"]]
    print("Distribution shift flags (|standardised diff| > 0.8):", shifted or "none")
    # independence guard
    set_hash = hashlib.sha256("".join(sorted(str(f.relative_to(root)) + str(f.stat().st_size) for f, _ in items)).encode()).hexdigest()[:12]
    log = root / ".evaluation_log.csv"; prev = pd.read_csv(log) if log.exists() else pd.DataFrame(columns=["time", "set_hash", "model", "tag", "n_labelled", "accuracy"])
    prev.loc[len(prev)] = [time.strftime("%Y-%m-%d %H:%M:%S"), set_hash, args.model, tag or "", len(lab), metrics.get("accuracy", np.nan)]
    prev.to_csv(log, index=False)
    n_used = int((prev["set_hash"] == set_hash).sum())
    if n_used >= 3:
        print(f"WARNING: this exact image set has now been evaluated {n_used} times. If it guided any tuning decision it is no "
              "longer an independent test set - photograph a fresh set for final numbers.")
    print("Saved", out)


def load_manifest_root() -> str:
    from src.utils import load_config, resolve
    return str(resolve(load_config()["dataset"]["prepared_dir"]))


if __name__ == "__main__":
    main()
