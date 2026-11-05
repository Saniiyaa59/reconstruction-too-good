"""Where data and results live. Each dataset writes its metrics and figures to its own subfolder."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "results" / "runs"      # saved models, shared by both datasets
METRICS_DIR = ROOT / "results" / "metrics"
FIGURES_DIR = ROOT / "results" / "figures"
