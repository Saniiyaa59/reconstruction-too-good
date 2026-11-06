"""Load MVTec AD categories as flat image vectors, like the ECG beats.

Each image is resized to `size` x `size`, kept in RGB (carpet has a "color"
defect that grayscale would hide), scaled to [0, 1] and flattened to
3 * size * size values. There is no per-image normalization, so brightness
and contrast still carry information.

    train : 85% of the normal training images
    val   : the other 15%
    test  : the official test set (normal + every defect type), with its
            ground-truth defect masks at the same resolution (all zero for normals)

Download the categories from mvtec.com into data/mvtec/<category>/ first.

    python -m src.mvtec.data              # build the 128x128 splits for every category and plot examples
    python -m src.mvtec.data --size 64    # the same at 64x64
"""
import argparse

import numpy as np
from PIL import Image

from src.common.paths import DATA_DIR, RESULTS_DIR

MVTEC_DIR = DATA_DIR / "mvtec"
PROCESSED_DIR = DATA_DIR / "processed"

SIZE = 128  # the first MVTec runs used 64, where the smallest defects are only a few pixels
SEED = 0


def categories():
    """Names of the downloaded categories (none if only the cached splits are present, as on the SCC)."""
    return sorted(p.name for p in MVTEC_DIR.iterdir() if p.is_dir()) if MVTEC_DIR.exists() else []


def load_images(paths, size):
    """(n, 3 * size * size) array in [0, 1]."""
    images = [np.asarray(Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR))
              for p in paths]
    return (np.stack(images).reshape(len(images), -1) / 255.0).astype(np.float32)


def load_masks(test_paths, size):
    """(n, size, size) boolean defect masks for the test images; all False for normal ones.

    Nearest-neighbour resizing keeps the mask binary; at 128x128 the smallest
    defect still covers about 6 pixels.
    """
    masks = np.zeros((len(test_paths), size, size), dtype=bool)
    for i, p in enumerate(test_paths):
        if p.parent.name != "good":
            mask_path = p.parents[2] / "ground_truth" / p.parent.name / f"{p.stem}_mask.png"
            masks[i] = np.asarray(Image.open(mask_path).convert("L").resize((size, size), Image.NEAREST)) > 0
    return masks


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
        "masks": load_masks(test_paths, size),
    }


def load_splits(category, size=SIZE):
    """Load processed splits for one category, building them first if needed."""
    path = PROCESSED_DIR / f"mvtec_{category}_{size}.npz"
    if path.exists():
        splits = dict(np.load(path))
        if "masks" not in splits:  # cached before masks were added
            test_paths = sorted((MVTEC_DIR / category / "test").glob("*/*.png"))
            splits["masks"] = load_masks(test_paths, size)
            np.savez(path, **splits)
        return splits
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=SIZE)
    size = parser.parse_args().size
    for category in categories():
        s = load_splits(category, size)
        n_normal = int((s["y_test"] == 0).sum())
        print(f"{category:<10} train {len(s['X_train'])} | val {len(s['X_val'])} | "
              f"test {len(s['X_test'])} ({n_normal} normal, {len(s['X_test']) - n_normal} anomalous) | "
              f"{s['X_train'].shape[1]} features")
        plot_examples(category, s, RESULTS_DIR / "mvtec" / "data" / f"examples_{category}_{size}.png", size)
