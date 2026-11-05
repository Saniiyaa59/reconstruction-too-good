"""PCA capacity sweep on MVTec AD categories (image-level AUROC).

Same setup as the ECG sweep in pca_model.py: fit PCA on normal training images
only, score each test image by its reconstruction error. With n training
images PCA has at most n - 1 useful components, so the sweep ends there.

    python src/pca_mvtec.py                     # every category in data/mvtec/
    python src/pca_mvtec.py --categories bottle
"""
import argparse

import numpy as np
from sklearn.decomposition import PCA

import mvtec
from evaluate import summarize, write_csv
from pca_model import FIGURES_DIR, METRICS_DIR, TYPE_COLORS, score

KS = [1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120]


def run(category):
    splits = mvtec.load_splits(category)
    X_train = splits["X_train"]
    ks = [k for k in KS if k < len(X_train) - 1] + [len(X_train) - 1]

    rows = []
    for k in ks:
        pca = PCA(n_components=k).fit(X_train)
        rows.append({"k": k, "train_err": float(score(pca, X_train).mean()),
                     **summarize(score(pca, splits["X_test"]), splits["defect"], "good")})

    defects = sorted(set(splits["defect"]) - {"good"})
    print(f"\n{category}: {len(X_train)} training images, {X_train.shape[1]} features")
    print(f"{'k':>4} {'train err':>10} {'normal err':>11} {'anomaly err':>12} {'AUROC':>6}  "
          + " ".join(f"{d[:10]:>10}" for d in defects))
    for r in rows:
        print(f"{r['k']:>4} {r['train_err']:>10.2e} {r['normal_err']:>11.2e} {r['anomaly_err']:>12.2e} "
              f"{r['auroc']:>6.3f}  " + " ".join(f"{r['auroc_class' + d]:>10.3f}" for d in defects))

    write_csv(rows, METRICS_DIR / f"pca_mvtec_{category}.csv")
    return rows, defects


def plot(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(results), figsize=(4.2 * len(results), 3.6), sharey=True, squeeze=False)
    for ax, (category, (rows, defects)) in zip(axes[0], results.items()):
        ks = [r["k"] for r in rows]
        for color, d in zip(TYPE_COLORS, defects):
            ax.plot(ks, [r["auroc_class" + d] for r in rows], "-", color=color, lw=1.2, alpha=0.8, label=d)
        ax.plot(ks, [r["auroc"] for r in rows], "o-", color="black", lw=2, ms=4, label="all defects")
        ax.axhline(0.5, color="gray", ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", color="0.9", lw=0.8)
        ax.set_xlabel("PCA components (k)")
        ax.set_title(f"{category} (max k = {ks[-1]})", fontsize=10)
        ax.legend(fontsize=7, frameon=False, loc="lower left")
    axes[0][0].set_ylabel("image-level AUROC")
    fig.suptitle("PCA capacity sweep on MVTec AD (64x64 RGB)", fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", nargs="+",
                        default=sorted(p.name for p in mvtec.MVTEC_DIR.iterdir() if p.is_dir()))
    args = parser.parse_args()
    results = {c: run(c) for c in args.categories}
    plot(results, FIGURES_DIR / "pca_mvtec.png")


if __name__ == "__main__":
    main()
