"""PCA capacity sweep on MVTec AD categories.

Same setup as the ECG sweep in src/ecg/pca.py: fit PCA on normal training images
only, score each test image by its reconstruction error (see scoring.py for the
mean and max scores). With n training images PCA has at most n - 1 useful
components, so the sweep ends there.

    python -m src.mvtec.pca                     # every category in data/mvtec/, 128x128
    python -m src.mvtec.pca --categories bottle --size 64
"""
import argparse
import math

from sklearn.decomposition import PCA

from src.common import paths
from src.common.evaluate import write_csv
from src.common.plots import PALETTE, save, style_auroc_axis
from src.mvtec import data as mvtec
from src.mvtec.scoring import evaluate, reconstruction_scores

OUT_DIR = paths.RESULTS_DIR / "mvtec" / "pca"

KS = [1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120]


def reconstruct(pca, X):
    return pca.inverse_transform(pca.transform(X))


def run(category, size):
    splits = mvtec.load_splits(category, size)
    X_train = splits["X_train"]
    ks = [k for k in KS if k < len(X_train) - 1] + [len(X_train) - 1]

    rows = []
    for k in ks:
        # The exact solver: sklearn's default switches to a randomized one with no fixed seed on images
        pca = PCA(n_components=k, svd_solver="full").fit(X_train)
        train_err = float(((X_train - reconstruct(pca, X_train)) ** 2).mean())
        scores, maps = reconstruction_scores(splits["X_test"], reconstruct(pca, splits["X_test"]), size)
        rows += [{"k": k, "train_err": train_err, **r} for r in evaluate(scores, maps, splits)]

    defects = sorted(set(splits["defect"]) - {"good"})
    print(f"\n{category}: {len(X_train)} training images, {X_train.shape[1]} features")
    print(f"{'k':>4} {'train err':>10} {'normal err':>11} {'AUROC mean':>11} {'AUROC max':>10} {'pixel':>6}  "
          + " ".join(f"{d[:10]:>10}" for d in defects) + "   (per type: max score)")
    for k in ks:
        mean, mx = [r for r in rows if r["k"] == k]
        print(f"{k:>4} {mean['train_err']:>10.2e} {mean['normal_mse']:>11.2e} {mean['auroc']:>11.3f} "
              f"{mx['auroc']:>10.3f} {mx['pixel_auroc']:>6.3f}  "
              + " ".join(f"{mx['auroc_class' + d]:>10.3f}" for d in defects))

    write_csv(rows, OUT_DIR / f"{category}_{size}.csv")
    return rows, defects


def plot(results, size, path):
    """Per category: image AUROC of the max score per defect type, and overall for both scores."""
    import matplotlib.pyplot as plt

    ncols = min(len(results), 5)
    nrows = math.ceil(len(results) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows), sharey=True, squeeze=False)
    for ax, (category, (rows, defects)) in zip(axes.flat, results.items()):
        mean = [r for r in rows if r["score"] == "mean"]
        mx = [r for r in rows if r["score"] == "max"]
        ks = [r["k"] for r in mx]
        for color, d in zip(PALETTE, defects):
            ax.plot(ks, [r["auroc_class" + d] for r in mx], "-", color=color, lw=1.2, alpha=0.8, label=d)
        ax.plot(ks, [r["auroc"] for r in mx], "o-", color="black", lw=2, ms=4, label="all defects (max)")
        ax.plot(ks, [r["auroc"] for r in mean], "s--", color="0.5", lw=1.5, ms=3, label="all defects (mean)")
        style_auroc_axis(ax)
        ax.set_xscale("log")
        ax.set_xlabel("PCA components (k)")
        ax.set_title(f"{category} (max k = {ks[-1]})", fontsize=10)
        ax.legend(fontsize=6, frameon=False, loc="lower left")
    for ax in axes.flat[len(results):]:
        ax.axis("off")
    for row in axes:
        row[0].set_ylabel("image AUROC")
    fig.suptitle(f"PCA capacity sweep on MVTec AD ({size}x{size} RGB); per type: max of smoothed error map",
                 fontsize=11)
    fig.tight_layout()
    save(fig, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", nargs="+", default=mvtec.categories())
    parser.add_argument("--size", type=int, default=mvtec.SIZE, help="image size in pixels")
    args = parser.parse_args()
    results = {c: run(c, args.size) for c in args.categories}
    plot(results, args.size, OUT_DIR / f"sweep_{args.size}.png")


if __name__ == "__main__":
    main()
