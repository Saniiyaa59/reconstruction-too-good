"""Conv AE and VAE on MVTec AD: train on normal images, score the test set, compare with PCA.

ConvVAE2D trained with beta = 0 is a plain autoencoder (AE), with beta = 1 a VAE.
For each category, beta, latent size and seed this trains one model (or loads it
if it is already saved, so a stopped job can be rerun) and scores the test images
with every score in scoring.py. As in the ECG beta sweep, the best epoch is the
one with the lowest validation MSE.

    python -m src.mvtec.vae_experiments run --categories carpet --device cuda    # AE + VAE, 3 seeds, 128x128
    python -m src.mvtec.vae_experiments run --categories carpet --size 64 --betas 0 --latents 64 12288 --seeds 0
    python -m src.mvtec.vae_experiments plot                                    # figure from the saved CSVs

Results go to results/mvtec/vae/<category>_<size>.csv, one row per model and score.
"""
import argparse
import csv
import math
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.common import paths
from src.common.evaluate import write_csv
from src.common.plots import PALETTE, PCA_COLOR, save, style_auroc_axis, style_error_axis
from src.common.train import load_run, save_run, train_vae
from src.mvtec import data as mvtec
from src.mvtec.scoring import evaluate, model_scores

OUT_DIR = paths.RESULTS_DIR / "mvtec" / "vae"
PCA_DIR = paths.RESULTS_DIR / "mvtec" / "pca"

# Latent code = an 8x8 grid with 1, 4, 16, 64 or 256 channels
LATENTS = [64, 256, 1024, 4096, 16384]
# With ~200 training images a batch of 128 gave 2 steps per epoch and runs hit the
# epoch cap; 32 gives ~7
TRAINING = {"batch_size": 32, "max_epochs": 3000, "patience": 300}


def run_name(category, size, beta, latent_dim, seed):
    return f"mvtec_{category}_{size}_b{beta:g}_z{latent_dim}_seed{seed}"


def get_model(splits, category, size, beta, latent_dim, seed, device):
    """(model, best_epoch, trained): load the saved run if there is one, otherwise train and save it."""
    name = run_name(category, size, beta, latent_dim, seed)
    if (paths.RUNS_DIR / name / "model.pt").exists():
        model, config = load_run(name)
        return model, config["best_epoch"], False
    result = train_vae(splits["X_train"], splits["X_val"], latent_dim, beta=beta, seed=seed, select="mse",
                       device=device, image_size=size, **TRAINING)
    save_run(name, result, {"latent_dim": latent_dim, "hidden": [], "channels": 32, "beta": beta, "seed": seed,
                            "select": "mse", "image_size": size, **TRAINING})
    return result.model, result.best_epoch, True


def run(category, betas, latents, seeds, size, device):
    splits = mvtec.load_splits(category, size)
    rows = []
    for beta in betas:
        for latent_dim in latents:
            for seed in seeds:
                start = time.time()
                model, best_epoch, trained = get_model(splits, category, size, beta, latent_dim, seed, device)
                train_err = float(model_scores(model, splits["X_train"], size)[0]["mean"].mean())
                scores, maps = model_scores(model, splits["X_test"], size)
                run_rows = [{"category": category, "beta": beta, "latent_dim": latent_dim, "seed": seed,
                             "best_epoch": best_epoch, "train_err": train_err, **r}
                            for r in evaluate(scores, maps, splits)]
                rows += run_rows
                by_score = {r["score"]: r for r in run_rows}
                kl = f", kl_max {by_score['kl_max']['auroc']:.3f}" if "kl_max" in by_score else ""
                print(f"{category:<10} beta {beta:g} latent {latent_dim:>5} seed {seed}: "
                      f"{'trained' if trained else 'loaded'}, best epoch {best_epoch:>4}, "
                      f"normal mse {by_score['mean']['normal_mse']:.2e}, AUROC mean {by_score['mean']['auroc']:.3f}, "
                      f"max {by_score['max']['auroc']:.3f}{kl}  ({time.time() - start:.0f}s)", flush=True)
                write_csv(rows, OUT_DIR / f"{category}_{size}.csv")  # after each run, in case of a stop


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def plot(categories, size, path):
    """Image AUROC (max score) vs. normal test MSE per category: the PCA curve, then the AE
    and VAE at each latent size (mean over seeds), and the VAE's kl_max score."""
    categories = [c for c in categories if (OUT_DIR / f"{c}_{size}.csv").exists()]
    ncols = min(len(categories), 5)
    nrows = math.ceil(len(categories) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows), sharey=True, squeeze=False)
    for ax, category in zip(axes.flat, categories):
        pca = [r for r in read_csv(PCA_DIR / f"{category}_{size}.csv") if r["score"] == "max"]
        ax.plot([float(r["normal_mse"]) for r in pca], [float(r["auroc"]) for r in pca], "s--",
                color=PCA_COLOR, lw=2, ms=4, label="PCA (k)")
        rows = read_csv(OUT_DIR / f"{category}_{size}.csv")
        series = [(0, "max", "o-", PALETTE[0], "AE"), (1, "max", "o-", PALETTE[1], "VAE"),
                  (1, "kl_max", "^:", PALETTE[2], "VAE kl_max")]
        for beta, score, style, color, label in series:
            pts = [r for r in rows if float(r["beta"]) == beta and r["score"] == score]
            latents = sorted({int(r["latent_dim"]) for r in pts})
            if not latents:
                continue
            mse = [np.mean([float(r["normal_mse"]) for r in pts if int(r["latent_dim"]) == z]) for z in latents]
            auroc = [np.mean([float(r["auroc"]) for r in pts if int(r["latent_dim"]) == z]) for z in latents]
            ax.plot(mse, auroc, style, color=color, lw=1.8, ms=5, label=label)
        style_auroc_axis(ax)
        style_error_axis(ax)
        ax.set_xlabel("normal test MSE (log)")
        ax.set_title(category, fontsize=10)
        ax.legend(fontsize=6, frameon=False, loc="lower left")
    for ax in axes.flat[len(categories):]:
        ax.axis("off")
    for row in axes:
        row[0].set_ylabel("image AUROC (max score)")
    fig.suptitle(f"Conv AE and VAE vs. PCA on MVTec AD ({size}x{size}); points = latent sizes, mean over seeds",
                 fontsize=11)
    fig.tight_layout()
    save(fig, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    run_cmd = commands.add_parser("run", help="train (or load) and score")
    run_cmd.add_argument("--categories", nargs="+", default=mvtec.categories())
    run_cmd.add_argument("--betas", type=float, nargs="+", default=[0, 1])
    run_cmd.add_argument("--latents", type=int, nargs="+", default=LATENTS)
    run_cmd.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    run_cmd.add_argument("--size", type=int, default=mvtec.SIZE, help="image size in pixels")
    run_cmd.add_argument("--device", default="cpu", help="cpu, cuda or mps")

    plot_cmd = commands.add_parser("plot", help="figure from the saved CSVs")
    plot_cmd.add_argument("--categories", nargs="+", default=mvtec.categories())
    plot_cmd.add_argument("--size", type=int, default=mvtec.SIZE, help="image size in pixels")

    args = parser.parse_args()
    if args.command == "run":
        for category in args.categories:
            run(category, args.betas, args.latents, args.seeds, args.size, args.device)
    else:
        plot(args.categories, args.size, OUT_DIR / f"sweep_{args.size}.png")


if __name__ == "__main__":
    main()
