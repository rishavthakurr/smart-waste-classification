"""Diagnostic evaluation of trained models: WHY do they fail?  (read-only; never trains or modifies models)

Usage: python -m training.diagnose [--tag repro] [--models cnn mobilenetv2] [--split test]

Per model it writes results/diagnostics/<tag|default>/<model>/ :
  label_check.json   class order consistency: config vs manifest/loader vs model metadata vs output size vs predictor
  metrics.json       train(clean, inference mode) / validation / test accuracy, macro+weighted F1, per-class P/R/F1, confusion
  misclassified.csv + misclassified.png    every error with true/predicted label and confidence
  probes.csv         accuracy under synthetic perturbations and background-bias probes (NOT real-world accuracy)
  summary.json       numeric facts + heuristic flags with the exact thresholds used
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from src.model_loader import ModelLoadError, load_model, model_meta
from src.preprocessing import apply_preprocessing
from src.predictor import result_from_probabilities
from src.utils import class_names, load_config, model_path, resolve, results_dir, run_results_dir, write_json
from training.common import load_arrays, load_manifest, split_frame

# --- heuristic thresholds (stated so the flags are auditable, not magic) ---
UNDERFIT_TRAIN_ACC = 0.80      # clean train accuracy below this => the model has not even fit the training data
OVERFIT_GAP = 0.15             # clean train accuracy minus validation accuracy above this => overfitting
BG_BIAS_FACTOR = 2.0           # border-only accuracy above this x majority-class rate => background/context shortcut


def label_check(model, meta: dict, key: str) -> dict:
    classes, cfg_classes = class_names(), load_config()["classes"]
    man = load_manifest()
    loader_ok, loader_map = True, {}
    for split in ("train", "validation", "test"):
        df = split_frame(split)
        for c, ids in df.groupby("class")["label"].unique().items():
            loader_map.setdefault(c, set()).update(int(i) for i in ids)
    loader_ok = all(len(v) == 1 and list(v)[0] == classes.index(c) for c, v in loader_map.items())
    one_hot_ok = all(result_from_probabilities(np.eye(len(classes))[i], key).predicted_class == classes[i]
                     for i in range(len(classes)))
    out = {"config_classes": cfg_classes, "loader_class_to_index": {c: sorted(v)[0] for c, v in loader_map.items()},
           "classes_in_manifest_not_in_config": sorted(set(man["class"]) - set(classes)),
           "classes_in_config_missing_from_manifest": sorted(set(classes) - set(man["class"])),
           "model_metadata_classes": meta.get("classes"), "model_output_units": int(model.output_shape[-1]),
           "loader_indices_match_config": bool(loader_ok), "predictor_index_to_name_matches_config": bool(one_hot_ok),
           "metadata_matches_config": meta.get("classes") == classes, "output_units_match": int(model.output_shape[-1]) == len(classes)}
    out["consistent"] = bool(loader_ok and one_hot_ok and out["metadata_matches_config"] and out["output_units_match"]
                             and not out["classes_in_manifest_not_in_config"] and not out["classes_in_config_missing_from_manifest"])
    return out


def predict(model, x: np.ndarray, scheme: str, batch: int = 64) -> np.ndarray:
    return np.concatenate([model.predict(np.asarray(apply_preprocessing(x[i:i + batch].astype("float32"), scheme)),
                                         verbose=0) for i in range(0, len(x), batch)])


def split_metrics(y, probs, classes) -> dict:
    pred = probs.argmax(1); labels = range(len(classes))
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    return {"n": int(len(y)), "accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
            "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                          for i, c in enumerate(classes)},
            "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
            "predicted_class_counts": {c: int((pred == i).sum()) for i, c in enumerate(classes)}}


# --- synthetic probes (operate on uint8 arrays at model input size) ---
def _mask(x, keep_center: bool):
    n, h, w, _ = x.shape; ch, cw = int(h * 0.3), int(w * 0.3)       # central 40% x 40%... defined below
    out = np.full_like(x, 128)
    box = (slice(h // 2 - int(h * .3), h // 2 + int(h * .3)), slice(w // 2 - int(w * .3), w // 2 + int(w * .3)))   # central 60%
    if keep_center:
        out[:, box[0], box[1]] = x[:, box[0], box[1]]
    else:
        out[:] = x; out[:, box[0], box[1]] = 128
    return out


PROBES = {
    "clean": lambda x: x,
    "center_only (border grayed out)": lambda x: _mask(x, True),
    "border_only (object area grayed out)": lambda x: _mask(x, False),
    "brightness x0.6": lambda x: np.clip(x.astype("float32") * 0.6, 0, 255).astype("uint8"),
    "brightness x1.4": lambda x: np.clip(x.astype("float32") * 1.4, 0, 255).astype("uint8"),
    "low contrast (x0.5 around mean)": lambda x: np.clip((x.astype("float32") - x.mean((1, 2, 3), keepdims=True)) * 0.5
                                                       + x.mean((1, 2, 3), keepdims=True), 0, 255).astype("uint8"),
    "gaussian blur": lambda x: np.stack([cv2.GaussianBlur(i, (7, 7), 0) for i in x]),
    "gaussian noise (sigma 15)": lambda x: np.clip(x.astype("float32") + np.random.default_rng(0).normal(0, 15, x.shape), 0, 255).astype("uint8"),
    "grayscale (colour removed)": lambda x: np.repeat(np.stack([cv2.cvtColor(i, cv2.COLOR_RGB2GRAY) for i in x])[..., None], 3, axis=3),
    "rotate 90 deg": lambda x: np.rot90(x, 1, axes=(1, 2)).copy(),
    "zoom: centre crop 70%": lambda x: np.stack([cv2.resize(i[int(i.shape[0] * .15):int(i.shape[0] * .85), int(i.shape[1] * .15):int(i.shape[1] * .85)],
                                                           (i.shape[1], i.shape[0])) for i in x]),
}


def montage(df: pd.DataFrame, root: Path, path: Path, n: int = 24) -> None:
    sel = df.sort_values("confidence", ascending=False).head(n)
    if sel.empty:
        return
    cols = 6; rows = int(np.ceil(len(sel) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.5)); axes = np.atleast_2d(axes)
    for a in axes.ravel():
        a.axis("off")
    for a, (_, r) in zip(axes.ravel(), sel.iterrows()):
        a.imshow(cv2.cvtColor(cv2.imread(str(root / r["path"])), cv2.COLOR_BGR2RGB))
        a.set_title(f"true: {r['true_class']}\npred: {r['predicted_class']} ({r['confidence']:.2f})", fontsize=7)
    fig.suptitle("Most confident misclassifications", fontsize=10); plt.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def diagnose_model(key: str, tag: str | None, out_root: Path) -> dict | None:
    if not model_path(key, tag).is_file():
        print(f"[skip] {key}: no model at {model_path(key, tag)}"); return None
    cfg = load_config(); classes = class_names(); scheme = cfg["models"][key]["preprocessing"]
    meta = model_meta(key, tag)
    try:
        model = load_model(key, tag)
    except ModelLoadError as exc:                 # e.g. class order in metadata differs from config.yaml
        out0 = out_root / key; out0.mkdir(parents=True, exist_ok=True)
        write_json(out0 / "label_check.json", {"consistent": False, "error": str(exc), "config_classes": class_names(),
                                               "model_metadata_classes": meta.get("classes"), "metadata_matches_config": meta.get("classes") == class_names()})
        print(f"[FAIL] {key}: model rejected by the loader -> {exc}"); return None
    size = int(model.input_shape[1]); out = out_root / key; out.mkdir(parents=True, exist_ok=True)
    root = resolve(load_config()["dataset"]["prepared_dir"])
    check = label_check(model, meta, key); write_json(out / "label_check.json", check)
    data, dfs, metrics = {}, {}, {}
    for split in ("train", "validation", "test"):
        dfs[split] = split_frame(split); data[split] = load_arrays(dfs[split], size)
        probs = predict(model, data[split], scheme); metrics[split] = split_metrics(dfs[split]["label"].to_numpy(), probs, classes)
        if split == "test":
            test_probs = probs
    hist_path = run_results_dir(tag) / key / "history.csv"
    hist = pd.read_csv(hist_path) if hist_path.exists() else None
    metrics["training_history_last_epoch"] = ({k: float(hist[k].iloc[-1]) for k in ("accuracy", "loss", "val_accuracy", "val_loss") if k in hist}
                                              if hist is not None else None)
    write_json(out / "metrics.json", metrics)
    # misclassified
    t = dfs["test"]; pred = test_probs.argmax(1)
    err = pd.DataFrame({"path": t["path"], "true_class": [classes[i] for i in t["label"]], "predicted_class": [classes[i] for i in pred],
                        "confidence": test_probs.max(1), "correct": pred == t["label"].to_numpy()})
    wrong = err[~err["correct"]].drop(columns="correct").sort_values("confidence", ascending=False)
    wrong.to_csv(out / "misclassified.csv", index=False); montage(wrong, root, out / "misclassified.png")
    pairs = wrong.groupby(["true_class", "predicted_class"]).size().sort_values(ascending=False).head(8)
    # probes
    rows = []
    y = t["label"].to_numpy(); chance = float(np.bincount(y).max() / len(y))
    for name, fn in PROBES.items():
        acc = float(accuracy_score(y, predict(model, np.asarray(fn(data["test"])), scheme).argmax(1)))
        rows.append({"probe": name, "test_accuracy": acc})
    probes = pd.DataFrame(rows); clean = probes.loc[probes["probe"] == "clean", "test_accuracy"].iloc[0]
    probes["delta_vs_clean"] = probes["test_accuracy"] - clean; probes["majority_class_rate"] = chance
    probes.to_csv(out / "probes.csv", index=False)
    tr, va, te = (metrics[s]["accuracy"] for s in ("train", "validation", "test"))
    border = float(probes.loc[probes["probe"].str.startswith("border_only"), "test_accuracy"].iloc[0])
    summary = {
        "model": key, "tag": tag, "weights_source": meta.get("weights_source"), "pretrained_imagenet": meta.get("pretrained_imagenet"),
        "smoke_test": meta.get("smoke_test"), "input_size": size, "label_check_consistent": check["consistent"],
        "clean_train_accuracy": tr, "validation_accuracy": va, "test_accuracy": te, "majority_class_rate_test": chance,
        "train_minus_validation_gap": tr - va, "test_class_counts": {c: metrics["test"]["per_class"][c]["support"] for c in classes},
        "weakest_classes_by_recall": sorted(classes, key=lambda c: metrics["test"]["per_class"][c]["recall"])[:3],
        "top_confusions": {f"{a} -> {b}": int(n) for (a, b), n in pairs.items()},
        "predicts_only_few_classes": sum(v > 0 for v in metrics["test"]["predicted_class_counts"].values()) < len(classes) - 1,
        "flags (heuristic, thresholds above)": {
            "underfitting_suspected": bool(tr < UNDERFIT_TRAIN_ACC),
            "overfitting_suspected": bool(tr - va > OVERFIT_GAP),
            "background_or_context_shortcut_suspected": bool(border > BG_BIAS_FACTOR * chance),
            "random_weight_backbone": meta.get("pretrained_imagenet") is False and cfg["models"][key]["transfer"]},
        "border_only_accuracy": border, "worst_perturbation": probes.iloc[probes["delta_vs_clean"].idxmin()]["probe"],
    }
    write_json(out / "summary.json", summary)
    print(f"{key:<12} train(clean)={tr:.3f} val={va:.3f} test={te:.3f} | majority={chance:.3f} border-only={border:.3f} | "
          f"labels consistent={check['consistent']} | flags={ {k: v for k, v in summary['flags (heuristic, thresholds above)'].items() if v} }")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default=""); ap.add_argument("--models", nargs="*", default=list(load_config()["models"]))
    args = ap.parse_args(); tag = args.tag or None
    out_root = results_dir() / "diagnostics" / (tag or "default")
    summaries = [s for k in args.models if (s := diagnose_model(k, tag, out_root))]
    if not summaries:
        raise SystemExit("No models found to diagnose.")
    pd.DataFrame([{k: v for k, v in s.items() if not isinstance(v, (dict, list))} for s in summaries]).to_csv(out_root / "summary.csv", index=False)
    print("Saved:", out_root)


if __name__ == "__main__":
    main()
