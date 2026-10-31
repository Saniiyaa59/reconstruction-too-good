"""PCA baseline: a linear autoencoder swept over the number of components k.

Fits PCA on normal training beats only and scores each test beat by its
reconstruction error.

    python src/pca_model.py            # E1: test normals vs. real anomaly classes
    python src/pca_model.py --exp E2   # E2: real anomalies split by complexity quartile
    python src/pca_model.py --exp E3   # E3: test normals vs. distance-matched synthetic anomalies
"""
import argparse
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

import anomalies
import complexity
import data
from evaluate import plot_paradox_curve, reconstruction_error, summarize, write_csv

ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = ROOT / "results" / "metrics"
FIGURES_DIR = ROOT / "results" / "figures"

KS = [1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120]
E3_PANELS = [2, 20, 50, 80, 120]  # k values shown in the E3 figure
# Default categorical palette, fixed order; line style also marks the direction
TYPE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
# One blue ramp, light -> dark = simplest -> most complex quartile
QUARTILE_COLORS = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]


def score(pca, X):
    return reconstruction_error(X, pca.inverse_transform(pca.transform(X)))


def run_e1(splits):
    rows = []
    for k in KS:
        pca = PCA(n_components=k).fit(splits["X_train"])
        scores = score(pca, splits["X_test"])
        rows.append({"k": k, **summarize(scores, splits["y_test"], data.NORMAL_CLASS)})

    print(f"{'k':>4} {'normal err':>11} {'anomaly err':>12} {'AUROC':>7}")
    for r in rows:
        print(f"{r['k']:>4} {r['normal_err']:>11.4g} {r['anomaly_err']:>12.4g} {r['auroc']:>7.3f}")

    write_csv(rows, METRICS_DIR / "pca_ecg.csv")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_paradox_curve(KS, rows, xlabel="PCA components (k)",
                       title="PCA capacity sweep on ECG5000",
                       path=FIGURES_DIR / "paradox_sanity_check.png")


def run_e2(splits):
    """AUROC of test normals vs. real anomalies split into complexity quartiles, for every k."""
    X, y = splits["X_test"], splits["y_test"]
    is_normal = y == data.NORMAL_CLASS
    comp = complexity.compute_all(X)

    # Quartiles of each measure over the anomalous beats; Q1 = simplest
    quartile = {}
    for name, c in comp.items():
        edges = np.quantile(c[~is_normal], [0.25, 0.5, 0.75])
        quartile[name] = np.searchsorted(edges, c, side="right") + 1

    rows = []
    for k in KS:
        pca = PCA(n_components=k).fit(splits["X_train"])
        s = score(pca, X)
        for name in comp:
            for q in range(1, 5):
                m = is_normal | (quartile[name] == q)
                in_q = m & ~is_normal
                rows.append({"k": k, "measure": name.removeprefix("complexity_"), "quartile": q,
                             "n": int(in_q.sum()),
                             "range": f"{comp[name][in_q].min():.3g}-{comp[name][in_q].max():.3g}",
                             "auroc": roc_auc_score(~is_normal[m], s[m])})

    measures = [n.removeprefix("complexity_") for n in comp]
    print("median complexity per class (class 1 = normal)")
    print(f"{'class':>6} {'n':>5} " + " ".join(f"{n:>9}" for n in measures))
    for c in np.unique(y):
        print(f"{c:>6} {(y == c).sum():>5} " + " ".join(f"{np.median(v[y == c]):>9.3f}" for v in comp.values()))

    print("\nAUROC of test normals vs. anomaly quartile (Q1 = simplest)")
    print(f"{'k':>4} " + " ".join(f"{m + ' Q' + str(q):>11}" for m in measures for q in range(1, 5)))
    for k in KS:
        print(f"{k:>4} " + " ".join(f"{r['auroc']:>11.3f}" for r in rows if r["k"] == k))

    write_csv(rows, METRICS_DIR / "pca_ecg_E2.csv")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_e2_complexity(comp, y, FIGURES_DIR / "ecg_complexity_by_class.png")
    plot_e2(rows, measures, FIGURES_DIR / "pca_ecg_E2.png")


def plot_e2_complexity(comp, y, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    classes = np.unique(y)
    fig, axes = plt.subplots(1, len(comp), figsize=(3.6 * len(comp), 3.2))
    for ax, (name, c) in zip(axes, comp.items()):
        ax.boxplot([c[y == k] for k in classes], showfliers=False, widths=0.6,
                   medianprops={"color": "black"})
        ax.set_xticks(range(1, len(classes) + 1),
                      [f"{k}\n(normal)" if k == data.NORMAL_CLASS else str(k) for k in classes])
        ax.set_xlabel("class")
        ax.set_title(name.removeprefix("complexity_"), fontsize=10)
        ax.grid(axis="y", color="0.9", lw=0.8)
    fig.suptitle("Complexity of ECG5000 test beats by class (higher = more complex)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_e2(rows, measures, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(measures), figsize=(3.6 * len(measures), 3.4), sharey=True)
    for ax, measure in zip(axes, measures):
        for q, color in zip(range(1, 5), QUARTILE_COLORS):
            pts = [r for r in rows if r["measure"] == measure and r["quartile"] == q]
            ax.plot([r["k"] for r in pts], [r["auroc"] for r in pts], "o-", color=color, lw=2, ms=5,
                    label=f"Q{q}" + (" (simplest)" if q == 1 else " (most complex)" if q == 4 else ""))
        ax.axhline(0.5, color="gray", ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("PCA components (k)")
        ax.set_title(f"anomalies split by {measure}", fontsize=10)
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", color="0.9", lw=0.8)
    axes[0].set_ylabel("AUROC (test normals vs. quartile)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4,
               fontsize=8, frameon=False)
    fig.suptitle("PCA on ECG5000: real anomalies by complexity quartile (E2)", fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_e3(splits):
    """AUROC of test normals vs. each synthetic type at each distance, for every k."""
    X_normal = splits["X_test"][splits["y_test"] == data.NORMAL_CLASS]
    anoms = anomalies.generate(X_normal)

    rows = []
    for k in KS:
        pca = PCA(n_components=k).fit(splits["X_train"])
        s_normal, s_anom = score(pca, X_normal), score(pca, anoms["X"])
        for name, (_, direction, _, _) in anomalies.TYPES.items():
            for d in anomalies.DISTANCES:
                m = (anoms["anomaly_type"] == name) & (anoms["distance"] == d)
                labels = np.r_[np.zeros(len(s_normal)), np.ones(m.sum())]
                rows.append({"k": k, "anomaly_type": name, "direction": direction,
                             "distance": d, "n": int(m.sum()),
                             "auroc": roc_auc_score(labels, np.r_[s_normal, s_anom[m]])})

    print(f"{'k':>4} {'type':>9} " + " ".join(f"{'d=' + format(d, 'g'):>6}" for d in anomalies.DISTANCES))
    for k in KS:
        for name in anomalies.TYPES:
            cells = [r["auroc"] for r in rows if r["k"] == k and r["anomaly_type"] == name]
            print(f"{k:>4} {name:>9} " + " ".join(f"{a:>6.3f}" for a in cells))

    write_csv(rows, METRICS_DIR / "pca_ecg_E3.csv")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_e3(rows, FIGURES_DIR / "pca_ecg_E3.png")


def plot_e3(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(E3_PANELS), figsize=(3 * len(E3_PANELS), 3.2), sharey=True)
    for ax, k in zip(axes, E3_PANELS):
        for color, (name, (_, direction, _, _)) in zip(TYPE_COLORS, anomalies.TYPES.items()):
            pts = [r for r in rows if r["k"] == k and r["anomaly_type"] == name]
            ax.plot([r["distance"] for r in pts], [r["auroc"] for r in pts],
                    "o-" if direction == "simpler" else "s--", color=color, lw=2, ms=6,
                    label=f"{name} ({direction})")
        ax.axhline(0.5, color="gray", ls=":", lw=1)
        ax.set_title(f"k = {k}", fontsize=10)
        ax.set_xlabel("distance d")
        ax.set_xticks(anomalies.DISTANCES)
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", color="0.9", lw=0.8)
    axes[0].set_ylabel("AUROC (test normals vs. synthetic)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center",
               ncol=len(anomalies.TYPES), fontsize=8, frameon=False)
    fig.suptitle("PCA on ECG5000: distance-matched synthetic anomalies (E3)", fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", choices=["E1", "E2", "E3"], default="E1")
    args = parser.parse_args()

    splits = data.load_splits()
    {"E1": run_e1, "E2": run_e2, "E3": run_e3}[args.exp](splits)


if __name__ == "__main__":
    main()
