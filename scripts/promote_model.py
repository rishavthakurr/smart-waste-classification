"""Make a tagged model the one the Streamlit app uses, WITHOUT losing what was there.

Usage: python scripts/promote_model.py --tag improved --model mobilenetv2
 1. if models/<file> exists and models/original/<file> does not, the current model is preserved in models/original/
 2. models/<tag>/<file> is copied to models/<file>; its meta/history/metrics are copied to results/<model>/
 3. run `python -m training.evaluate_models` afterwards so the dashboard comparison reflects the live models.
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import model_path, results_dir, run_results_dir


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--model", required=True)
    a = ap.parse_args(); src, live, keep = model_path(a.model, a.tag), model_path(a.model), model_path(a.model, "original")
    if not src.is_file():
        raise SystemExit(f"{src} does not exist.")
    if live.is_file() and not keep.is_file():
        keep.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(live, keep)
        old = results_dir() / a.model
        if old.exists():
            shutil.copytree(old, run_results_dir("original") / a.model, dirs_exist_ok=True)
        print(f"Preserved current model in {keep}")
    shutil.copy2(src, live)
    shutil.copytree(run_results_dir(a.tag) / a.model, results_dir() / a.model, dirs_exist_ok=True)
    print(f"Promoted {src} -> {live}. Now run: python -m training.evaluate_models")


if __name__ == "__main__":
    main()
