"""Download ECG5000 and build the normal-only train / val / test splits.

The UCR train/test files are pooled (5000 beats) and re-split, since the
original 500-row train file holds only ~290 normal beats.

    train : 70% of normal beats
    val   : 15% of normal beats
    test  : remaining 15% of normal beats + all anomalous beats

UCR distributes every beat already z-normalized (mean 0, std ~1 per beat), so
amplitude and variance carry no information; we keep that preprocessing.
"""
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

BASE_URL = ("https://raw.githubusercontent.com/cerenyildiiz/"
            "ecg5000-anomaly-detection-time-series/main/data/")
FILES = ["ECG5000_TRAIN.txt", "ECG5000_TEST.txt"]

NORMAL_CLASS = 1
SEED = 0


def download():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        path = RAW_DIR / name
        if not path.exists():
            print(f"downloading {name}")
            urllib.request.urlretrieve(BASE_URL + name, path)


def load_raw():
    """Return (X, y) for all 5000 beats; X is (5000, 140), y is class 1-5."""
    data = np.concatenate([np.loadtxt(RAW_DIR / name) for name in FILES])
    return data[:, 1:].astype(np.float32), data[:, 0].astype(int)


def make_splits(X, y, seed=SEED):
    rng = np.random.default_rng(seed)
    normal_idx = rng.permutation(np.flatnonzero(y == NORMAL_CLASS))
    anomaly_idx = np.flatnonzero(y != NORMAL_CLASS)

    n_train = int(0.7 * len(normal_idx))
    n_val = int(0.15 * len(normal_idx))
    train_idx = normal_idx[:n_train]
    val_idx = normal_idx[n_train:n_train + n_val]
    test_idx = np.concatenate([normal_idx[n_train + n_val:], anomaly_idx])

    return {
        "X_train": X[train_idx],
        "X_val": X[val_idx],
        "X_test": X[test_idx],
        "y_test": y[test_idx],  # original class labels, 1 = normal
    }


def resample(X, length):
    """Linearly interpolate each signal (row) to `length` points over the same time span."""
    old_t = np.linspace(0, 1, X.shape[1])
    new_t = np.linspace(0, 1, length)
    return np.stack([np.interp(new_t, old_t, row) for row in X]).astype(np.float32)


def load_splits():
    """Load processed splits, building them first if needed."""
    path = PROCESSED_DIR / "splits.npz"
    if not path.exists():
        prepare()
    return dict(np.load(path))


def prepare():
    download()
    X, y = load_raw()
    classes, counts = np.unique(y, return_counts=True)
    print("class counts:", dict(zip(classes.tolist(), counts.tolist())))

    splits = make_splits(X, y)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(PROCESSED_DIR / "splits.npz", **splits)
    np.save(PROCESSED_DIR / "normal_signals.npy", X[y == NORMAL_CLASS])
    np.save(PROCESSED_DIR / "anomaly_signals.npy", X[y != NORMAL_CLASS])

    n_test_normal = int((splits["y_test"] == NORMAL_CLASS).sum())
    print(f"train {len(splits['X_train'])} | val {len(splits['X_val'])} | "
          f"test {len(splits['X_test'])} "
          f"({n_test_normal} normal, {len(splits['X_test']) - n_test_normal} anomalous)")


if __name__ == "__main__":
    prepare()
