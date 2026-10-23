"""Linear probe on FROZEN backbone features (no backprop, cheap, no augmentation).

Answers two diagnostic questions with measurements:
  1. How good are genuine pretrained ImageNet features for this dataset, versus a random-weight backbone?
  2. How much accuracy is lost by feeding the backbone the WRONG preprocessing?
Hyper-parameter C is chosen on the validation split; the test split is only read once, for the chosen C.

Usage: python -m training.probe_features [--sizes 96 224] [--model mobilenetv2]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from src.preprocessing import apply_preprocessing
from src.utils import load_config, results_dir
from training.common import load_arrays, split_frame

VARIANTS = [  # (label, backbone weights, preprocessing applied to the images)
    ("pretrained + CORRECT preprocessing", True, "mobilenet_v2"),
    ("pretrained + wrong: rescale to [0,1]", True, "rescale_01"),
    ("pretrained + wrong: raw 0-255", True, "raw_255"),
    ("pretrained + wrong: ResNet BGR mean-subtract", True, "resnet50"),
    ("random weights + correct preprocessing", False, "mobilenet_v2"),
]


def preprocess(x: np.ndarray, scheme: str) -> np.ndarray:
    return x.astype("float32") if scheme == "raw_255" else np.asarray(apply_preprocessing(x.astype("float32"), scheme))


def features(backbone, x: np.ndarray, scheme: str, batch: int = 64) -> np.ndarray:
    out = [backbone.predict(preprocess(x[i:i + batch], scheme), verbose=0).mean(axis=(1, 2))
           for i in range(0, len(x), batch)]
    return np.concatenate(out)


def main() -> None:
    import keras
    from training.models import _load_backbone
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[96, 224])
    ap.add_argument("--model", default="mobilenetv2", choices=["mobilenetv2"])
    ap.add_argument("--no-standardise", action="store_true",
                    help="do NOT standardise features (mirrors the real softmax head; exposes preprocessing sensitivity)")
    args = ap.parse_args()
    dfs = {s: split_frame(s) for s in ("train", "validation", "test")}
    rows = []
    for size in args.sizes:
        data = {s: load_arrays(df, size) for s, df in dfs.items()}
        variants = VARIANTS if size == min(args.sizes) else VARIANTS[:1]     # full study at the small size only
        for label, pretrained, scheme in variants:
            t0 = time.time(); keras.utils.set_random_seed(0)
            backbone, loaded, source = _load_backbone(keras.applications.MobileNetV2, "mobilenetv2", (size, size, 3),
                                                      pretrained, allow_random_init=False)
            f = {s: features(backbone, data[s], scheme) for s in data}
            sc = None if args.no_standardise else StandardScaler().fit(f["train"])
            X = {s: (v if sc is None else sc.transform(v)) for s, v in f.items()}
            y = {s: dfs[s]["label"].to_numpy() for s in dfs}
            best = max(((C, LogisticRegression(C=C, max_iter=3000).fit(X["train"], y["train"])) for C in (0.01, 0.1, 1.0, 10.0)),
                       key=lambda cm: accuracy_score(y["validation"], cm[1].predict(X["validation"])))
            C, clf = best
            pv, pt = clf.predict(X["validation"]), clf.predict(X["test"])
            rows.append({"image_size": size, "backbone": label, "weights_source": source, "chosen_C": C, "standardised_features": not args.no_standardise,
                         "val_accuracy": accuracy_score(y["validation"], pv), "test_accuracy": accuracy_score(y["test"], pt),
                         "test_macro_f1": f1_score(y["test"], pt, average="macro"), "seconds": round(time.time() - t0, 1)})
            print(f"{size}px | {label:<46} val={rows[-1]['val_accuracy']:.4f} test={rows[-1]['test_accuracy']:.4f} "
                  f"macroF1={rows[-1]['test_macro_f1']:.4f}  ({rows[-1]['seconds']}s)", flush=True)
    out = results_dir() / "diagnostics"; out.mkdir(exist_ok=True)
    name = "feature_probe_unstandardised.csv" if args.no_standardise else "feature_probe.csv"
    pd.DataFrame(rows).to_csv(out / name, index=False)
    print("Saved", out / name)


if __name__ == "__main__":
    main()
