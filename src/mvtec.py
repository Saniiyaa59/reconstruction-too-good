"""Load MVTec AD categories as flat image vectors, like the ECG beats.

Each image is resized to `size` x `size`, kept in RGB (carpet has a "color"
defect that grayscale would hide), scaled to [0, 1] and flattened to
3 * size * size values. There is no per-image normalization, so brightness
and contrast still carry information.

    train : 85% of the normal training images
    val   : the other 15%
    test  : the official test set (normal + every defect type)

Download the categories from mvtec.com into data/mvtec/<category>/ first.

    python src/mvtec.py    # build the splits for the downloaded categories and plot examples
"""
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MVTEC_DIR = ROOT / "data" / "mvtec"
PROCESSED_DIR = ROOT / "data" / "processed"

SIZE = 64
SEED = 0


def load_images(paths, size):
    """(n, 3 * size * size) array in [0, 1]."""
    images = [np.asarray(Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR))
              for p in paths]
    return (np.stack(images).reshape(len(images), -1) / 255.0).astype(np.float32)


def make_splits(category, size=SIZE, seed=SEED):
    cat_dir = MVTEC_DIR / category
    train_paths = sorted((cat_dir / "train" / "good").glob("*.png"))
    test_paths = sorted((cat_dir / "test").glob("*/*.png"))
    defect = np.array([p.parent.name for p in test_paths])

    idx = np.random.default_rng(seed).permutation(len(train_paths))
    n_train = int(0.85 * len(idx))
    X = load_images(train_paths, size)
    return {
        "X_train": X[idx[:n_train]],
        "X_val": X[idx[n_train:]],
        "X_test": load_images(test_paths, size),
        "y_test": (defect != "good").astype(int),  # 1 = anomaly
        "defect": defect,                          # "good" or the defect type
    }


def load_splits(category, size=SIZE):
    """Load processed splits for one category, building them first if needed."""
    path = PROCESSED_DIR / f"mvtec_{category}_{size}.npz"
    if not path.exists():
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(path, **make_splits(category, size))
    return dict(np.load(path))


def plot_examples(category, splits, path, size=SIZE):
    """One normal test image and one image per defect type, at the resolution the models see."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    types = ["good"] + sorted(set(splits["defect"]) - {"good"})
    fig, axes = plt.subplots(1, len(types), figsize=(1.8 * len(types), 2.1))
    for ax, t in zip(axes, types):
        img = splits["X_test"][np.flatnonzero(splits["defect"] == t)[0]].reshape(size, size, 3)
        ax.imshow(img)
        ax.set_title(t, fontsize=8)
        ax.axis("off")
    fig.suptitle(f"MVTec {category} at {size}x{size}", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    figures = ROOT / "results" / "figures"
    for cat_dir in sorted(p for p in MVTEC_DIR.iterdir() if p.is_dir()):
        s = load_splits(cat_dir.name)
        n_normal = int((s["y_test"] == 0).sum())
        print(f"{cat_dir.name:<8} train {len(s['X_train'])} | val {len(s['X_val'])} | "
              f"test {len(s['X_test'])} ({n_normal} normal, {len(s['X_test']) - n_normal} anomalous) | "
              f"{s['X_train'].shape[1]} features")
        plot_examples(cat_dir.name, s, figures / f"mvtec_{cat_dir.name}_examples.png")
