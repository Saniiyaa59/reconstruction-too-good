"""Conv autoencoder capacity sweep on MVTec AD, compared with PCA at equal error.

Trains ConvVAE2D with beta = 0 (a plain autoencoder) on normal training images,
sweeping the size of the latent code, and scores each test image by its
reconstruction error (image-level AUROC). As in the ECG beta sweep, the best
epoch is the one with the lowest validation MSE.

    python src/ae_mvtec.py --device mps                  # every category in data/mvtec/
    python src/ae_mvtec.py --categories bottle --latents 64 12288
"""
import argparse
import csv
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import mvtec
from evaluate import summarize, write_csv
from pca_model import FIGURES_DIR, METRICS_DIR
from train import save_run, train_vae
from vae_plots import PALETTE, plot_pca_curve, save, style_auroc_axis, style_error_axis

# Latent code = an 8x8 grid with 1, 4, 16, 64 or 192 channels; 12288 equals the input size
LATENTS = [64, 256, 1024, 4096, 12288]


def run(category, latents, seed, device):
    splits = mvtec.load_splits(category)
    to_tensor = lambda X: torch.tensor(X, dtype=torch.float32)
    rows = []
    for latent_dim in latents:
        start = time.time()
        result = train_vae(splits["X_train"], splits["X_val"], latent_dim, beta=0.0, seed=seed,
                           select="mse", device=device, image_size=mvtec.SIZE)
        save_run(f"mvtec_{category}_ae_z{latent_dim}_seed{seed}", result,
                 {"latent_dim": latent_dim, "hidden": [], "channels": 32, "beta": 0.0, "seed": seed,
                  "select": "mse", "image_size": mvtec.SIZE})

        recon = result.model.score(to_tensor(splits["X_test"]))["recon"].numpy()
        train_err = float(result.model.score(to_tensor(splits["X_train"]))["recon"].mean())
        row = {"latent_dim": latent_dim, "seed": seed, "best_epoch": result.best_epoch, "train_err": train_err,
               **summarize(recon, splits["defect"], "good")}
        rows.append(row)
        print(f"{category:<7} latent {latent_dim:>5}: best epoch {result.best_epoch:>4}, train mse {train_err:.2e}, "
              f"normal mse {row['normal_err']:.2e}, anomaly mse {row['anomaly_err']:.2e}, "
              f"AUROC {row['auroc']:.3f}  ({time.time() - start:.0f}s)", flush=True)
        write_csv(rows, METRICS_DIR / f"ae_mvtec_{category}.csv")  # after each run, in case of a stop
    return rows


def plot(results, path):
    """AUROC vs. normal test error per category: AE points (labelled by latent size) and the PCA curve."""
    fig, axes = plt.subplots(1, len(results), figsize=(4.4 * len(results), 3.8), sharey=True, squeeze=False)
    for ax, (category, rows) in zip(axes[0], results.items()):
        with open(METRICS_DIR / f"pca_mvtec_{category}.csv") as f:
            plot_pca_curve(ax, list(csv.DictReader(f)), "PCA (k)")
        ax.plot([r["normal_err"] for r in rows], [r["auroc"] for r in rows], "o-", color=PALETTE[0],
                lw=2, ms=6, label="conv AE (latent size)")
        for r in rows:
            ax.annotate(str(r["latent_dim"]), (r["normal_err"], r["auroc"]), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=7, color=PALETTE[0])
        style_auroc_axis(ax)
        style_error_axis(ax)
        ax.set_xlabel("normal test MSE (log)")
        ax.set_title(category, fontsize=10)
        ax.legend(fontsize=7, frameon=False, loc="lower left")
    axes[0][0].set_ylabel("image-level AUROC")
    fig.suptitle("Conv autoencoder vs. PCA on MVTec AD at equal reconstruction error", fontsize=11)
    fig.tight_layout()
    save(fig, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", nargs="+",
                        default=sorted(p.name for p in mvtec.MVTEC_DIR.iterdir() if p.is_dir()))
    parser.add_argument("--latents", type=int, nargs="+", default=LATENTS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", help="cpu, cuda or mps")
    args = parser.parse_args()
    results = {c: run(c, args.latents, args.seed, args.device) for c in args.categories}
    plot(results, FIGURES_DIR / "ae_mvtec.png")


if __name__ == "__main__":
    main()
