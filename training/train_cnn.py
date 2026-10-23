"""Train the custom CNN model.  Run:  python -m training.train_cnn [--epochs N] [--smoke] [--cpu]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.common import run_cli

if __name__ == "__main__":
    run_cli("cnn")
