"""Diffusion model on MVTec AD: train on normal images, score the test set by noising and denoising.

One model per category and seed. Each test image is noised to step t (of 1000) and
denoised back (src/common/diffusion.py); the error between input and reconstruction
is scored exactly like the AE/VAE reconstructions (src/mvtec/scoring.py). The same
trained model is scored at every noise level in NOISE_LEVELS.

    python -m src.mvtec.diffusion_experiments run --categories tile --seeds 0 --device cuda
    python -m src.mvtec.diffusion_experiments run --categories tile --seeds 0 --max-steps 200 --device mps  # smoke test
    python -m src.mvtec.diffusion_experiments maps --category tile --defect gray_stroke --seed 0

Each (category, seed) writes its own CSV, results/mvtec/diffusion/<category>_<size>_seed<seed>.csv,
so jobs running in parallel on the SCC never overwrite each other.
"""
import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.common import paths
from src.common.diffusion import Diffusion, reconstruct_set, train_diffusion
from src.common.evaluate import write_csv
from src.mvtec import data as mvtec
from src.mvtec.scoring import evaluate, reconstruction_scores

OUT_DIR = paths.RESULTS_DIR / "mvtec" / "diffusion"
NOISE_LEVELS = [50, 100, 250, 500]  # of 1000: how much of each test image is destroyed before denoising
TRAINING = {"batch_size": 32, "lr": 2e-4, "max_steps": 40000, "eval_every": 500, "patience": 10}


def run_name(category, size, seed):
    return f"mvtec_{category}_{size}_diffusion_seed{seed}"


def save_run(name, result, config):
    run_dir = paths.RUNS_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(result.model.state_dict(), run_dir / "model.pt")
    (run_dir / "config.json").write_text(json.dumps({**config, "best_step": result.best_step}, indent=2))
    write_csv(result.history, run_dir / "history.csv")


def load_run(name):
    run_dir = paths.RUNS_DIR / name
    config = json.loads((run_dir / "config.json").read_text())
    model = Diffusion(config["image_size"], config["channels"])
    model.load_state_dict(torch.load(run_dir / "model.pt"))
    return model.eval(), config


def get_model(splits, category, size, seed, device, training):
    """(model, best_step, trained): load the saved run if there is one, otherwise train and save it."""
    name = run_name(category, size, seed)
    if (paths.RUNS_DIR / name / "model.pt").exists():
        model, config = load_run(name)
        return model, config["best_step"], False
    result = train_diffusion(splits["X_train"], splits["X_val"], size, seed=seed, device=device, **training)
    save_run(name, result, {"image_size": size, "channels": 32, "seed": seed, **training})
    return result.model, result.best_step, True


def run(category, seeds, size, device, training, train_only=False):
    splits = mvtec.load_splits(category, size)
    for seed in seeds:
        start = time.time()
        model, best_step, trained = get_model(splits, category, size, seed, device, training)
        print(f"{category} diffusion seed {seed}: {'trained' if trained else 'loaded'}, best step {best_step} "
              f"({time.time() - start:.0f}s)", flush=True)
        if train_only:
            continue
        rows = []
        for t in NOISE_LEVELS:
            start = time.time()
            X_hat = reconstruct_set(model, splits["X_test"], t, device)
            scores, maps = reconstruction_scores(splits["X_test"], X_hat, size)
            run_rows = [{"category": category, "seed": seed, "noise": t, "best_step": best_step, **r}
                        for r in evaluate(scores, maps, splits)]
            rows += run_rows
            by_score = {r["score"]: r for r in run_rows}
            print(f"  noise {t:>3}: normal mse {by_score['mean']['normal_mse']:.2e}, AUROC mean "
                  f"{by_score['mean']['auroc']:.3f}, max {by_score['max']['auroc']:.3f}  ({time.time() - start:.0f}s)",
                  flush=True)
        write_csv(rows, OUT_DIR / f"{category}_{size}_seed{seed}.csv")


def maps(category, defect, seed, size, device):
    """Picture check: 1 normal and the first 3 `defect` test images; the input, then the
    reconstruction and smoothed error map at each noise level. Each noise level has its own
    error colour scale (shown in its title), shared by the rows."""
    model, config = load_run(run_name(category, size, seed))
    splits = mvtec.load_splits(category, size)
    rows = [int(np.flatnonzero(splits["defect"] == "good")[0])] + list(np.flatnonzero(splits["defect"] == defect)[:3])
    recons, errs = {}, {}
    for t in NOISE_LEVELS:  # whole test set, so each image gets the same noise as in the CSV scores
        X_hat = reconstruct_set(model, splits["X_test"], t, device)
        recons[t] = X_hat[rows]
        errs[t] = reconstruction_scores(splits["X_test"][rows], X_hat[rows], size)[1]["max"]

    ncols = 1 + 2 * len(NOISE_LEVELS)
    fig, axes = plt.subplots(len(rows), ncols, figsize=(1.6 * ncols, 1.7 * len(rows) + 0.6), layout="constrained")
    for i, idx in enumerate(rows):
        axes[i, 0].imshow(splits["X_test"][idx].reshape(size, size, 3))
        if splits["masks"][idx].any():
            axes[i, 0].contour(splits["masks"][idx], levels=[0.5], colors="red", linewidths=0.8)
        axes[i, 0].set_ylabel(splits["defect"][idx], fontsize=8)
        for j, t in enumerate(NOISE_LEVELS):
            axes[i, 1 + 2 * j].imshow(recons[t][i].reshape(size, size, 3))
            axes[i, 2 + 2 * j].imshow(errs[t][i], cmap="magma", vmin=0, vmax=errs[t].max())
            if i == 0:
                axes[0, 1 + 2 * j].set_title(f"denoised, t={t}", fontsize=7)
                axes[0, 2 + 2 * j].set_title(f"error, t={t}\n(0 to {errs[t].max():.3f})", fontsize=7)
    axes[0, 0].set_title("input (mask in red)", fontsize=7)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{category}, diffusion (seed {seed}, best step {config['best_step']}): "
                 f"noised to step t of 1000, then denoised", fontsize=9)
    out = OUT_DIR / f"defect_{category}_{defect}_denoise.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    run_cmd = commands.add_parser("run", help="train (or load) and score")
    run_cmd.add_argument("--categories", nargs="+", default=["tile", "cable"])
    run_cmd.add_argument("--seeds", type=int, nargs="+", default=[0])
    run_cmd.add_argument("--size", type=int, default=mvtec.SIZE)
    run_cmd.add_argument("--device", default="cpu", help="cpu, cuda or mps")
    run_cmd.add_argument("--max-steps", type=int, default=TRAINING["max_steps"], help="training step budget")
    run_cmd.add_argument("--train-only", action="store_true", help="train (or load) without scoring")

    maps_cmd = commands.add_parser("maps", help="picture check of the denoised images")
    maps_cmd.add_argument("--category", default="tile")
    maps_cmd.add_argument("--defect", default="gray_stroke")
    maps_cmd.add_argument("--seed", type=int, default=0)
    maps_cmd.add_argument("--size", type=int, default=mvtec.SIZE)
    maps_cmd.add_argument("--device", default="cpu", help="cpu, cuda or mps")

    args = parser.parse_args()
    if args.command == "run":
        training = {**TRAINING, "max_steps": args.max_steps}
        for category in args.categories:
            run(category, args.seeds, args.size, args.device, training, args.train_only)
    else:
        maps(args.category, args.defect, args.seed, args.size, args.device)


if __name__ == "__main__":
    main()
