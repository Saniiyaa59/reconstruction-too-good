"""Where data and results live.

Results are grouped by dataset and model, results/<dataset>/<model>/, with each
figure next to the CSV it was made from. results/README.md lists them all.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"  # saved models, shared by both datasets
