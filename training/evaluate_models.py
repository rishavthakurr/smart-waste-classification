"""Evaluate every trained model on the SAME held-out test split.

Usage: python -m training.evaluate_models [--models cnn mobilenetv2 resnet50]

Inference time procedure (documented, consistent for all models):
  * batch size 1, one fixed test image, preprocessing done once BEFORE timing
  * 10 warm-up forward passes, then 100 timed passes of model(x, training=False)
  * reports the MEDIAN in milliseconds; excludes image loading/decoding/preprocessing/UI
Recommendation rule: highest macro-F1; among models within 0.01 macro-F1 of the best,
choose the one with the lowest inference time.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)

from src.model_loader import load_model, model_meta
from src.utils import class_names, load_config, model_path, run_results_dir, write_json
from training.common import make_dataset, manifest_sha256, split_frame

WARMUP, RUNS = 10, 100


def time_inference(model, sample: np.ndarray) -> float:
    for _ in range(WARMUP):
        model(sample, training=False)
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter(); model(sample, training=False); times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def plot_confusion(cm: np.ndarray, classes: list[str], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Greens")
    labels = [c.replace("_", " ").title() for c in classes]
    ax.set_xticks(range(len(classes)), labels, rotation=35, ha="right"); ax.set_yticks(range(len(classes)), labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046); plt.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def main() -> None:
    cfg = load_config(); classes = class_names()
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(cfg["models"]))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--tag", default="", help="evaluate models/<tag>/ and write to results/<tag>/")
    ap.add_argument("--force", action="store_true",
                    help="evaluate even if the model was trained on a different split (NOT recommended)")
    args = ap.parse_args()
    tag = args.tag or None; res = run_results_dir(tag)

    summary_rows, perclass_rows, skipped = [], [], []
    test_df = split_frame("test")
    cur_hash = manifest_sha256()
    print(f"Test images: {len(test_df)}   split manifest: {cur_hash[:12]}...")
    for key in args.models:
        if not model_path(key, tag).is_file():
            print(f"[skip] {key}: model file not found (train it first)"); continue
        mcfg, meta = cfg["models"][key], model_meta(key, tag)
        trained_hash = meta.get("manifest_sha256")
        if trained_hash != cur_hash and not args.force:
            print(f"[skip] {key}: trained on a different dataset split "
                  f"({'unknown' if not trained_hash else trained_hash[:12] + '...'}), so evaluating it on the "
                  f"current test set could be contaminated. Retrain it, or pass --force to override.")
            skipped.append(key); continue
        model = load_model(key, tag)
        size = int(model.input_shape[1])
        ds = make_dataset(test_df, size, mcfg["preprocessing"], args.batch_size, False, 0)
        probs = model.predict(ds, verbose=0)
        y_true, y_pred = test_df["label"].values, probs.argmax(1)
        p, r, f, sup = precision_recall_fscore_support(y_true, y_pred, labels=range(len(classes)), zero_division=0)
        mp, mr, mf, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        wp, wr, wf, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
        sample = next(iter(make_dataset(test_df.head(1), size, mcfg["preprocessing"], 1, False, 0)))[0].numpy()
        ms = time_inference(model, sample)
        cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
        out = res / key; out.mkdir(parents=True, exist_ok=True)
        pred_df = pd.DataFrame({"path": test_df["path"], "true_class": [classes[i] for i in y_true],
                                "predicted_class": [classes[i] for i in y_pred], "confidence": probs.max(1),
                                "correct": y_true == y_pred})
        for i, c in enumerate(classes):
            pred_df[f"prob_{c}"] = probs[:, i]
        pred_df.to_csv(out / "test_predictions.csv", index=False)
        report = classification_report(y_true, y_pred, labels=range(len(classes)), target_names=classes,
                                       output_dict=True, zero_division=0)
        write_json(out / "metrics.json", {"model_key": key, "classes": classes, "accuracy": float(accuracy_score(y_true, y_pred)),
                                          "classification_report": report, "confusion_matrix": cm.tolist(),
                                          "test_images": int(len(test_df)), "manifest_sha256": cur_hash})
        pd.DataFrame(cm, index=classes, columns=classes).to_csv(out / "confusion_matrix.csv")
        plot_confusion(cm, classes, f"{mcfg['display_name']} - confusion matrix (test)", out / "confusion_matrix.png")
        summary_rows.append({
            "model_key": key, "model": mcfg["display_name"], "test_accuracy": accuracy_score(y_true, y_pred),
            "macro_precision": mp, "macro_recall": mr, "macro_f1": mf,
            "weighted_precision": wp, "weighted_recall": wr, "weighted_f1": wf,
            "inference_ms_per_image": ms, "model_size_mb": model_path(key, tag).stat().st_size / 1e6,
            "parameters": int(model.count_params()), "test_images": len(test_df),
            "pretrained_imagenet": meta.get("pretrained_imagenet"), "fine_tuned": meta.get("fine_tuned"),
            "smoke_test": meta.get("smoke_test", False), "trained_on_this_split": trained_hash == cur_hash})
        for i, c in enumerate(classes):
            perclass_rows.append({"model_key": key, "model": mcfg["display_name"], "class": c,
                                  "precision": p[i], "recall": r[i], "f1": f[i], "support": int(sup[i])})
        print(f"{mcfg['display_name']:<12} acc={summary_rows[-1]['test_accuracy']:.4f} macroF1={mf:.4f} "
              f"{ms:.1f} ms/img  {summary_rows[-1]['model_size_mb']:.1f} MB")

    if not summary_rows:
        raise SystemExit(f"All trained models were skipped because they were trained on a different split ({skipped}). Retrain them." if skipped
                         else "No trained models found. Train at least one model first.")
    summ, per = pd.DataFrame(summary_rows), pd.DataFrame(perclass_rows)
    summ.to_csv(res / "model_comparison.csv", index=False); per.to_csv(res / "per_class_metrics.csv", index=False)

    # Charts
    x = np.arange(len(summ)); colors = ["#1B5E20", "#81C784", "#2F80ED"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for a, col, ttl in zip(axes, ["test_accuracy", "macro_f1", "inference_ms_per_image"],
                           ["Test accuracy", "Macro F1-score", "Inference time (ms/image)"]):
        a.bar(summ["model"], summ[col], color=colors[:len(summ)]); a.set_title(ttl)
        for i, v in enumerate(summ[col]):
            a.text(i, v, f"{v:.3f}" if col != "inference_ms_per_image" else f"{v:.1f}", ha="center", va="bottom")
    plt.tight_layout(); fig.savefig(res / "model_comparison.png", dpi=140); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.5)); w = 0.8 / len(summ)
    for i, (k, g) in enumerate(per.groupby("model", sort=False)):
        ax.bar(np.arange(len(classes)) + i * w, g["f1"], w, label=k, color=colors[i % 3])
    ax.set_xticks(np.arange(len(classes)) + w * (len(summ) - 1) / 2, [c.replace("_", " ").title() for c in classes])
    ax.set_ylabel("F1-score"); ax.set_title("Per-class F1-score (test set)"); ax.legend(); plt.tight_layout()
    fig.savefig(res / "per_class_f1.png", dpi=140); plt.close(fig)

    best_f1 = summ["macro_f1"].max()
    close = summ[summ["macro_f1"] >= best_f1 - 0.01]
    pick = close.sort_values("inference_ms_per_image").iloc[0]
    write_json(res / "recommended_model.json", {
        "model_key": pick["model_key"], "model": pick["model"],
        "rule": "highest macro-F1; among models within 0.01 of best, lowest inference time",
        "macro_f1": float(pick["macro_f1"]), "inference_ms_per_image": float(pick["inference_ms_per_image"]),
        "based_on_smoke_test_model": bool(pick["smoke_test"])})
    write_json(res / "evaluation_summary.json", {
        "test_images": int(len(test_df)), "manifest_sha256": cur_hash, "classes": classes,
        "models": summ.to_dict(orient="records"), "recommended_model": pick["model_key"],
        "inference_time_procedure": f"batch size 1, {WARMUP} warm-up + {RUNS} timed forward passes, median ms, model only",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    print("\nRecommended deployment model:", pick["model"])
    print("Saved:", res)


if __name__ == "__main__":
    main()
