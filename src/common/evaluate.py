"""Shared scoring and plotting for every model in the capacity sweeps."""
import csv

import numpy as np
from sklearn.metrics import roc_auc_score


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def reconstruction_error(X, X_hat):
    """Per-sample mean squared error."""
    return ((X - X_hat) ** 2).mean(axis=1)


def summarize(scores, y, normal_class):
    """Metrics for one sweep setting, given per-sample anomaly scores and class labels."""
    is_anomaly = y != normal_class
    row = {
        "normal_err": scores[~is_anomaly].mean(),
        "anomaly_err": scores[is_anomaly].mean(),
        "auroc": roc_auc_score(is_anomaly, scores),
    }
    # AUROC of each anomaly class against the normal test samples
    for c in np.unique(y[is_anomaly]):
        mask = (y == normal_class) | (y == c)
        row[f"auroc_class{c}"] = roc_auc_score(y[mask] == c, scores[mask])
    return row


def plot_paradox_curve(capacities, rows, xlabel, title, path):
    """AUROC (left axis) and normal/anomaly error (right axis, log) against capacity."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(capacities, [r["auroc"] for r in rows], "o-", color="black", label="AUROC")
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("AUROC")
    ax.set_ylim(0, 1.02)
    ax.set_xscale("log")

    ax2 = ax.twinx()
    ax2.plot(capacities, [r["normal_err"] for r in rows], "s--", color="tab:blue",
             label="normal error")
    ax2.plot(capacities, [r["anomaly_err"] for r in rows], "^--", color="tab:red",
             label="anomaly error")
    ax2.set_yscale("log")
    ax2.set_ylabel("mean reconstruction error (MSE)")

    lines = ax.get_lines()[:1] + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="lower left")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
