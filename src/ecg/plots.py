"""Figures for the ECG VAE experiments. `pca_rows` are rows read from the PCA CSV files."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.common.plots import (PALETTE, QUARTILE_COLORS, plot_pca_curve, save, style_auroc_axis,
                              style_error_axis)
from src.common.vae import SCORES
from src.ecg import anomalies


def plot_e1(rows, pca_rows, latent_dims, path, title):
    """Left: AUROC vs. latent size per score. Right: AUROC vs. normal test error, VAE vs. PCA."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    # Dashed neg_elbo so it stays visible when it lies on top of recon
    for s, color, style in zip(SCORES, PALETTE, ["o-", "o-", "s--"]):
        pts = [r for r in rows if r["score"] == s]
        ax.plot([r["latent_dim"] for r in pts], [r["auroc"] for r in pts], style,
                color=color, lw=2, ms=6, label=s)
    style_auroc_axis(ax)
    ax.set_xscale("log")
    ax.set_xticks(latent_dims, [str(d) for d in latent_dims])
    ax.set_xlabel("latent dimension")
    ax.set_ylabel("AUROC (test normals vs. all anomalies)")
    ax.legend(title="score", loc="lower left", frameon=False)
    ax.set_title(title, fontsize=11)

    vae_pts = [r for r in rows if r["score"] == "recon"]
    ax2.plot([r["normal_err"] for r in vae_pts], [r["auroc"] for r in vae_pts], "o-",
             color=PALETTE[0], lw=2, ms=6, label="VAE (latent dim)")
    # The larger latent sizes sit almost on one spot, so label them as a group
    labels = [(vae_pts[0], f"z={vae_pts[0]['latent_dim']}"),
              (vae_pts[-1], f"z={vae_pts[1]['latent_dim']}–{vae_pts[-1]['latent_dim']}")]
    for r, text in labels[:len(vae_pts)]:
        ax2.annotate(text, (r["normal_err"], r["auroc"]), textcoords="offset points",
                     xytext=(0, -14), ha="center", fontsize=8, color=PALETTE[0])
    if pca_rows:
        plot_pca_curve(ax2, pca_rows, "PCA (k)")
        for r in pca_rows:
            if r["k"] in ("1", "20", "50", "80", "120"):
                ax2.annotate(r["k"], (float(r["normal_err"]), float(r["auroc"])),
                             textcoords="offset points", xytext=(6, -10), fontsize=8, color="0.35")
    style_auroc_axis(ax2)
    style_error_axis(ax2)
    ax2.legend(loc="lower left", frameon=False)
    ax2.set_title("AUROC (recon score) at matched reconstruction quality", fontsize=11)

    fig.tight_layout()
    save(fig, path)


def plot_beta_sweep(rows, pca_rows, path):
    """AUROC (recon) vs. normal test error, VAE seeds against the PCA curve."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    plot_pca_curve(ax, pca_rows, "PCA (k = 1 to 120)")
    ax.plot([r["normal_err"] for r in rows], [r["auroc_recon"] for r in rows], "o",
            color=PALETTE[0], ms=7, label="VAE (one point per β and seed)")
    style_auroc_axis(ax)
    style_error_axis(ax)
    ax.set_ylabel("AUROC (recon score)")
    ax.legend(loc="lower left", frameon=False)
    ax.set_title("VAE β sweep vs. PCA on ECG5000", fontsize=11)
    fig.tight_layout()
    save(fig, path)


def plot_e2(rows, pca_e2_rows, pca_rows, path):
    """AUROC per complexity quartile vs. normal test error: PCA as lines, each VAE run as points."""
    pca_err = {r["k"]: float(r["normal_err"]) for r in pca_rows}
    measures = list(dict.fromkeys(r["measure"] for r in rows))

    fig, axes = plt.subplots(1, len(measures), figsize=(4.2 * len(measures), 3.8), sharey=True, squeeze=False)
    for ax, measure in zip(axes[0], measures):
        for q, color in zip(range(1, 5), QUARTILE_COLORS):
            pca = [r for r in pca_e2_rows if r["measure"] == measure and int(r["quartile"]) == q]
            ax.plot([pca_err[r["k"]] for r in pca], [float(r["auroc"]) for r in pca], "--",
                    color=color, lw=1.5)
            vae = [r for r in rows if r["measure"] == measure and r["quartile"] == q]
            ax.plot([r["normal_err"] for r in vae], [r["auroc"] for r in vae], "o", color=color, ms=5,
                    label=f"Q{q}" + (" (simplest)" if q == 1 else " (most complex)" if q == 4 else ""))
        style_auroc_axis(ax)
        style_error_axis(ax)
        ax.set_xlabel("normal test MSE (log)")
        ax.set_title(f"anomalies split by {measure}", fontsize=10)
    axes[0][0].set_ylabel("AUROC (test normals vs. quartile)")
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle("Real anomalies by complexity quartile: PCA (dashed lines, k = 1 to 120) vs. VAE runs (points)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, path)


def plot_e3(all_rows, pca_rows, path):
    """AUROC (recon) vs. distance, one line per type: PCA k=50 next to each saved run."""
    pca = [{**r, "auroc_recon": float(r["auroc"]), "distance": float(r["distance"])}
           for r in pca_rows if r["k"] == "50"]
    panels = {"PCA k=50": pca, **all_rows}

    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.4), sharey=True, squeeze=False)
    for ax, (name, rows) in zip(axes[0], panels.items()):
        for color, (t, (_, direction, _, _)) in zip(PALETTE, anomalies.TYPES.items()):
            pts = [r for r in rows if r["anomaly_type"] == t]
            ax.plot([r["distance"] for r in pts], [r["auroc_recon"] for r in pts],
                    "o-" if direction == "simpler" else "s--", color=color, lw=2, ms=6,
                    label=f"{t} ({direction})")
        style_auroc_axis(ax)
        ax.set_title(name.replace("vae_ecg5000_sweep_", ""), fontsize=10)
        ax.set_xlabel("distance d")
        ax.set_xticks(anomalies.DISTANCES)
    axes[0][0].set_ylabel("AUROC (test normals vs. synthetic)")
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="lower center",
               ncol=len(anomalies.TYPES), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, path)
