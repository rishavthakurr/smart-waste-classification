"""One-shot setup helper.  Usage:  python scripts/setup_project.py [--download-data]

Checks the Python/TensorFlow environment, creates working folders and (optionally)
downloads and prepares the dataset. It never trains models.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download-data", action="store_true", help="download + prepare TrashNet (needs internet)")
    args = ap.parse_args()
    print(f"Python {sys.version.split()[0]}")
    missing = []
    for mod in ("tensorflow", "keras", "streamlit", "plotly", "cv2", "PIL", "pandas", "sklearn", "yaml"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        raise SystemExit(f"Missing packages: {missing}. Run: pip install -r requirements.txt")
    import tensorflow as tf
    print("TensorFlow", tf.__version__, "| GPUs:", tf.config.list_physical_devices("GPU") or "none (CPU mode)")
    for d in ("models", "dataset", "results", "assets"):
        (ROOT / d).mkdir(exist_ok=True)
    from src.database import init_db
    init_db()
    print("Folders and database ready.")
    if args.download_data:
        subprocess.check_call([sys.executable, "-m", "training.prepare_dataset", "--download"], cwd=ROOT)
    else:
        print("Next: python -m training.prepare_dataset --download")


if __name__ == "__main__":
    main()
