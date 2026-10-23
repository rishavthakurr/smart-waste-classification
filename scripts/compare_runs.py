"""Compare tagged runs (e.g. original vs improved) on the SAME held-out test split, in one CSV.

Usage: python scripts/compare_runs.py --tags original repro improved
Needs `python -m training.evaluate_models --tag <tag>` to have been run for each tag.
Adds a 95% Wilson interval for test accuracy: with a few hundred test images, differences smaller than the
interval width are NOT evidence of improvement. Real-world metrics are included when they exist.
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.utils import class_names, run_results_dir


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    d = 1 + z * z / n; c = p + z * z / (2 * n); m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - m) / d, (c + m) / d


def load_json(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--tags", nargs="+", required=True)
    args = ap.parse_args(); classes = class_names(); rows = []
    for tag in args.tags:
        base = run_results_dir(None if tag == "default" else tag); cmp = base / "model_comparison.csv"
        if not cmp.exists():
            print(f"[skip] {tag}: {cmp} missing (run: python -m training.evaluate_models --tag {tag})"); continue
        for _, r in pd.read_csv(cmp).iterrows():
            key = r["model_key"]; meta = load_json(base / key / "meta.json") or {}; mt = load_json(base / key / "metrics.json") or {}
            rep = mt.get("classification_report", {}); lo, hi = wilson(r["test_accuracy"], int(r["test_images"]))
            rw = load_json(base / "real_world" / key / "metrics.json") or {}
            rows.append({"run_tag": tag, "model": r["model"], "evaluation_split": "test (held-out, same for all runs)",
                         "n_test": int(r["test_images"]), "accuracy": r["test_accuracy"], "accuracy_ci95_low": lo, "accuracy_ci95_high": hi,
                         "macro_f1": r["macro_f1"], "weighted_f1": r["weighted_f1"],
                         **{f"recall_{c}": rep.get(c, {}).get("recall") for c in classes},
                         "inference_ms_per_image": r["inference_ms_per_image"], "model_size_mb": r["model_size_mb"],
                         "real_world_accuracy": rw.get("accuracy"), "real_world_n_labelled": rw.get("labelled_images"),
                         "cfg_image_size": meta.get("input_size"), "cfg_pretrained_imagenet": meta.get("pretrained_imagenet"),
                         "cfg_weights_source": meta.get("weights_source"), "cfg_fine_tuned": meta.get("fine_tuned"),
                         "cfg_photometric_aug": meta.get("photometric_augmentation"), "cfg_epochs_run": meta.get("epochs_run"),
                         "cfg_batch_size": meta.get("batch_size"), "cfg_lr": meta.get("learning_rate"), "cfg_seed": meta.get("seed"),
                         "cfg_smoke_test": meta.get("smoke_test"), "cfg_command": meta.get("command")})
    if not rows:
        raise SystemExit("Nothing to compare.")
    df = pd.DataFrame(rows); out = run_results_dir() / "comparison_runs.csv"; df.to_csv(out, index=False)
    show = df[["run_tag", "model", "accuracy", "accuracy_ci95_low", "accuracy_ci95_high", "macro_f1", "weighted_f1", "inference_ms_per_image"]]
    print(show.round(4).to_string(index=False)); print("\nSaved", out)
    print("Differences smaller than the CI width are not evidence of improvement.")


if __name__ == "__main__":
    main()
