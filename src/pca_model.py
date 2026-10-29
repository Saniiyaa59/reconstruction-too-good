"""PCA capacity sweep: a linear autoencoder as a sanity check for the paradox.

Fits PCA on normal training data only, reconstructs test samples with k
components, and scores each sample by reconstruction error.

    python src/pca_model.py                          # ECG5000
    python src/pca_model.py --dataset mnist --normal 8
    python src/pca_model.py --sweep resolution       # ECG5000, resample to L points
"""
import argparse
import csv
from pathlib import Path

from sklearn.decomposition import PCA

import data
import mnist_data
from evaluate import plot_paradox_curve, reconstruction_error, summarize

ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = ROOT / "results" / "metrics"
FIGURES_DIR = ROOT / "results" / "figures"

KS = {
    "ecg": [1, 2, 5, 10, 20, 30, 50, 80, 100, 120],
    "mnist": [1, 2, 5, 10, 20, 50, 100, 200, 300, 500, 700],
}


def reconstruct(pca, X):
    return pca.inverse_transform(pca.transform(X))


def run_sweep(X_train, X_test, y_test, normal_class, ks):
    rows = []
    for k in ks:
        pca = PCA(n_components=k).fit(X_train)
        scores = reconstruction_error(X_test, reconstruct(pca, X_test))
        rows.append({"k": k, **summarize(scores, y_test, normal_class)})
    return rows


RESOLUTIONS = [20, 35, 50, 80, 100, 140, 200, 280, 560]
FIXED_KS = [5, 20, 50]
K_FRACTION = 1 / 7  # k grows with resolution: k = 20 at L = 140


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def run_resolution_sweep():
    """Resample ECG beats to L points, then fit PCA with fixed k or with k proportional to L."""
    splits = data.load_splits()
    rows = []
    for L in RESOLUTIONS:
        X_train = data.resample(splits["X_train"], L)
        X_test = data.resample(splits["X_test"], L)
        settings = [(f"k={k}", k) for k in FIXED_KS if k < L]
        settings.append(("k=L/7", max(1, round(L * K_FRACTION))))
        for label, k in settings:
            row = run_sweep(X_train, X_test, splits["y_test"], data.NORMAL_CLASS, [k])[0]
            rows.append({"resolution": L, "setting": label, **row})

    print(f"{'L':>4} {'setting':>8} {'k':>4} {'normal err':>11} {'anomaly err':>12} {'AUROC':>7}")
    for r in rows:
        print(f"{r['resolution']:>4} {r['setting']:>8} {r['k']:>4} {r['normal_err']:>11.4g} "
              f"{r['anomaly_err']:>12.4g} {r['auroc']:>7.3f}")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(rows, METRICS_DIR / "pca_ecg_resolution.csv")
    plot_resolution_sweep(rows, FIGURES_DIR / "pca_ecg_resolution.png")


def plot_resolution_sweep(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.2))
    labels = [f"k={k}" for k in FIXED_KS] + ["k=L/7"]
    for label in labels:
        pts = [r for r in rows if r["setting"] == label]
        style = "o--" if label == "k=L/7" else "o-"
        ax.plot([r["resolution"] for r in pts], [r["auroc"] for r in pts], style,
                label=label + (" (grows with L)" if label == "k=L/7" else ""))
    ax.axvline(140, color="gray", ls=":", lw=1)
    ax.text(140, 0.03, " original (140)", color="gray", fontsize=8)
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xticks(RESOLUTIONS)
    ax.set_xticklabels(RESOLUTIONS)
    ax.set_xlabel("resolution L (points per beat, linear interpolation)")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0, 1.02)
    ax.set_title("PCA resolution sweep on ECG5000")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ecg", "mnist"], default="ecg")
    parser.add_argument("--normal", type=int, default=0, help="normal digit (mnist only)")
    parser.add_argument("--sweep", choices=["capacity", "resolution"], default="capacity")
    args = parser.parse_args()

    if args.sweep == "resolution":
        run_resolution_sweep()
        return

    if args.dataset == "ecg":
        splits = data.load_splits()
        normal_class = data.NORMAL_CLASS
        name, fig_name = "pca_ecg", "paradox_sanity_check.png"
        title = "PCA capacity sweep on ECG5000"
    else:
        splits = mnist_data.load_one_class(args.normal)
        normal_class = args.normal
        name = f"pca_mnist_normal{args.normal}"
        fig_name = f"{name}.png"
        title = f"PCA one-class MNIST (normal = {args.normal})"

    rows = run_sweep(splits["X_train"], splits["X_test"], splits["y_test"],
                     normal_class, KS[args.dataset])

    print(f"{'k':>4} {'normal err':>11} {'anomaly err':>12} {'AUROC':>7}")
    for r in rows:
        print(f"{r['k']:>4} {r['normal_err']:>11.4g} {r['anomaly_err']:>12.4g} {r['auroc']:>7.3f}")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(rows, METRICS_DIR / f"{name}.csv")

    plot_paradox_curve([r["k"] for r in rows], rows, xlabel="PCA components (k)",
                       title=title, path=FIGURES_DIR / fig_name)


if __name__ == "__main__":
    main()
