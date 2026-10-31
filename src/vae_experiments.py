"""VAE experiments on ECG5000. Each one trains or loads models, scores them and saves results.

    python src/vae_experiments.py e1                     # AUROC vs. latent size
    python src/vae_experiments.py beta-sweep             # how low can the error go as beta falls?
    python src/vae_experiments.py beta-sweep --channels 32 --betas 1 0 --seeds 0   # the same, conv VAE
    python src/vae_experiments.py matched-error          # KL on vs. off at equal reconstruction error
    python src/vae_experiments.py e3 RUN [RUN ...]       # synthetic anomalies, on saved runs

Add --device cuda to train on a GPU. This speeds up the conv VAE; small MLPs are faster on the CPU.
"""
import argparse
import csv
import time
from functools import partial
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import anomalies
import data
import vae_plots
from evaluate import summarize, write_csv
from train import load_run, save_run, train_vae
from vae import SCORES

ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = ROOT / "results" / "metrics"
FIGURES_DIR = ROOT / "results" / "figures"

# No bottleneck: the hidden layers and latent code are wider than the 140-point input
WIDE_MLP = {"latent_dim": 256, "hidden": (512, 256)}
SMALL_MLP_HIDDEN = (128, 64)
SMALL_LATENTS = (2, 3, 4, 6, 8, 12, 16)


# ---------- shared helpers ----------

def to_tensor(X):
    return torch.tensor(X, dtype=torch.float32)


def score_beats(model, x, seed):
    """All scores as numpy arrays. Seeded because neg_elbo samples z."""
    torch.manual_seed(seed)
    return {name: values.numpy() for name, values in model.score(x).items()}


def read_metrics(filename):
    with open(METRICS_DIR / filename) as f:
        return list(csv.DictReader(f))


class E3Data(NamedTuple):
    x_normal: torch.Tensor
    x_anom: torch.Tensor
    anoms: dict


def load_e3_data(splits):
    """Test normals and the synthetic anomalies made from them (the same ones PCA E3 uses)."""
    X_normal = splits["X_test"][splits["y_test"] == data.NORMAL_CLASS]
    anoms = anomalies.generate(X_normal)
    return E3Data(to_tensor(X_normal), to_tensor(anoms["X"]), anoms)


def e3_rows(model, seed, e3):
    """AUROC of test normals vs. each synthetic type at each distance, for every score."""
    torch.manual_seed(seed)  # neg_elbo samples z
    s_normal = {k: v.numpy() for k, v in model.score(e3.x_normal).items()}
    s_anom = {k: v.numpy() for k, v in model.score(e3.x_anom).items()}
    rows = []
    for name, (_, direction, _, _) in anomalies.TYPES.items():
        for d in anomalies.DISTANCES:
            m = (e3.anoms["anomaly_type"] == name) & (e3.anoms["distance"] == d)
            labels = np.r_[np.zeros(len(e3.x_normal)), np.ones(m.sum())]
            row = {"anomaly_type": name, "direction": direction, "distance": d, "n": int(m.sum())}
            for s in SCORES:
                row[f"auroc_{s}"] = roc_auc_score(labels, np.r_[s_normal[s], s_anom[s][m]])
            rows.append(row)
    return rows


# ---------- E1: AUROC vs. latent size ----------

def run_e1(latent_dims, beta, seed, device):
    """Train one MLP VAE per latent size and score the real anomalies with all three scores."""
    splits = data.load_splits()
    x_test, y_test = to_tensor(splits["X_test"]), splits["y_test"]
    x_val = to_tensor(splits["X_val"])
    tag = f"b{beta:g}_seed{seed}"

    rows = []
    for latent_dim in latent_dims:
        start = time.time()
        result = train_vae(splits["X_train"], splits["X_val"], latent_dim, beta=beta, seed=seed, device=device)
        save_run(f"vae_ecg5000_z{latent_dim}_{tag}", result,
                 {"latent_dim": latent_dim, "hidden": [128, 64], "beta": beta, "seed": seed})

        scores = score_beats(result.model, x_test, seed)
        write_csv([{"sample_id": i, "class": int(c), **{s: float(scores[s][i]) for s in SCORES}}
                   for i, c in enumerate(y_test)],
                  METRICS_DIR / f"ecg5000_E1_vae_z{latent_dim}_{tag}.csv")

        active = result.model.active_units(x_val)
        for s in SCORES:
            rows.append({"latent_dim": latent_dim, "score": s, "best_epoch": result.best_epoch,
                         "val_mse": result.best["mse"], "sigma_x": result.best["sigma_x"],
                         "active_units": active, **summarize(scores[s], y_test, data.NORMAL_CLASS)})
        print(f"z={latent_dim:>4}: best epoch {result.best_epoch:>4}, val mse {result.best['mse']:.4f}, "
              f"sigma_x {result.best['sigma_x']:.3f}, active {active}/{latent_dim}, "
              + ", ".join(f"AUROC {r['score']} {r['auroc']:.3f}" for r in rows[-3:])
              + f"  ({time.time() - start:.0f}s)")

    write_csv(rows, METRICS_DIR / f"vae_ecg_E1_{tag}.csv")
    pca_csv = METRICS_DIR / "pca_ecg.csv"
    vae_plots.plot_e1(rows, read_metrics("pca_ecg.csv") if pca_csv.exists() else [], latent_dims,
                      FIGURES_DIR / f"vae_ecg_E1_{tag}.png",
                      title=f"VAE capacity sweep on ECG5000 (β = {beta:g}, seed {seed})")


# ---------- Beta sweep: can the error reach the range where PCA collapsed? ----------

def run_beta_sweep(betas, seeds, channels, device):
    """Lower beta on one no-bottleneck network and compare with PCA at equal error.

    `channels` switches from the wide MLP to a ConvVAE.
    """
    splits = data.load_splits()
    x_test, y_test = to_tensor(splits["X_test"]), splits["y_test"]
    x_val = to_tensor(splits["X_val"])
    arch = "" if channels is None else f"_conv{channels}"

    rows = []
    for beta in betas:
        for seed in seeds:
            start = time.time()
            result = train_vae(splits["X_train"], splits["X_val"], **WIDE_MLP, channels=channels,
                               beta=beta, seed=seed, select="mse", device=device)
            save_run(f"vae_ecg5000_sweep{arch}_b{beta:g}_seed{seed}", result,
                     {**WIDE_MLP, "channels": channels, "beta": beta, "seed": seed, "select": "mse"})

            scores = score_beats(result.model, x_test, seed)
            row = {"beta": beta, "seed": seed, "best_epoch": result.best_epoch,
                   "sigma_x": result.best["sigma_x"], "active_units": result.model.active_units(x_val)}
            for s in SCORES:
                summary = summarize(scores[s], y_test, data.NORMAL_CLASS)
                row[f"auroc_{s}"] = summary["auroc"]
                if s == "recon":
                    row["normal_err"], row["anomaly_err"] = summary["normal_err"], summary["anomaly_err"]
            rows.append(row)
            print(f"beta={beta:<6g} seed={seed}: best epoch {result.best_epoch:>4}, "
                  f"normal mse {row['normal_err']:.4f}, anomaly mse {row['anomaly_err']:.4f}, "
                  f"AUROC recon {row['auroc_recon']:.3f} kl {row['auroc_kl']:.3f} "
                  f"neg_elbo {row['auroc_neg_elbo']:.3f}  ({time.time() - start:.0f}s)", flush=True)
            write_csv(rows, METRICS_DIR / f"vae_ecg_beta_sweep{arch}.csv")  # after each run, in case of a stop

    vae_plots.plot_beta_sweep(rows, read_metrics("pca_ecg.csv"), FIGURES_DIR / f"vae_ecg_beta_sweep{arch}.png")


# ---------- Matched error: does the KL help beyond limiting reconstruction? ----------

def run_matched_error(seeds, device):
    """Compare models with and without the KL at about the same validation MSE.

    Arms, all chosen on validation data only:
        beta1           : wide MLP, KL on. Its validation MSE is the target error
        beta0_earlystop : same network, KL off, stopped when it reaches the target
        beta0_small     : small MLP, KL off, latent size closest to the target
        beta0_full      : same network as beta1, KL off, not matched (baseline)
    """
    splits = data.load_splits()
    e3 = load_e3_data(splits)
    x_test = to_tensor(splits["X_test"])
    train = partial(train_vae, splits["X_train"], splits["X_val"], select="mse", device=device)
    rows = []

    def record(arm, seed, result, **config):
        save_run(f"vae_ecg5000_matched_{arm}_seed{seed}", result, {**config, "seed": seed, "select": "mse"})
        e1 = summarize(result.model.score(x_test)["recon"].numpy(), splits["y_test"], data.NORMAL_CLASS)
        row = {"arm": arm, "seed": seed, "latent_dim": config["latent_dim"],
               "hidden": "x".join(map(str, config["hidden"])), "beta": config["beta"],
               "best_epoch": result.best_epoch, "val_mse": result.best["val_recon"],
               "normal_err": e1["normal_err"], "anomaly_err": e1["anomaly_err"], "auroc_e1": e1["auroc"]}
        for r in e3_rows(result.model, seed, e3):
            row[f"{r['anomaly_type']}_d{r['distance']:g}"] = r["auroc_recon"]
        rows.append(row)
        write_csv(rows, METRICS_DIR / "vae_ecg_matched_error.csv")  # after each run, in case of a stop
        print(f"{arm:<16} seed={seed}: val mse {row['val_mse']:.4f}, normal mse {row['normal_err']:.4f}, "
              f"E1 AUROC {row['auroc_e1']:.3f}, smoothed d=2 {row['smoothed_d2']:.3f}, "
              f"d=4 {row['smoothed_d4']:.3f}", flush=True)

    # 1. The KL-on models set the target error
    for seed in seeds:
        record("beta1", seed, train(beta=1.0, seed=seed, **WIDE_MLP), beta=1.0, **WIDE_MLP)
    target = float(np.mean([r["val_mse"] for r in rows]))
    print(f"target validation MSE: {target:.4f}", flush=True)

    # 2. Pick the small network's latent size on validation MSE, using the first seed
    candidates = {z: train(beta=0.0, seed=seeds[0], latent_dim=z, hidden=SMALL_MLP_HIDDEN) for z in SMALL_LATENTS}
    for z, result in candidates.items():
        print(f"  small network, latent {z:>2}: val mse {result.best['val_recon']:.4f}", flush=True)
    best_z = min(candidates, key=lambda z: abs(candidates[z].best["val_recon"] - target))
    small = {"latent_dim": best_z, "hidden": SMALL_MLP_HIDDEN}

    # 3. Train the KL-off arms
    for seed in seeds:
        record("beta0_full", seed, train(beta=0.0, seed=seed, **WIDE_MLP), beta=0.0, **WIDE_MLP)
        record("beta0_earlystop", seed, train(beta=0.0, seed=seed, stop_at_mse=target, **WIDE_MLP),
               beta=0.0, **WIDE_MLP)
        result = candidates[best_z] if seed == seeds[0] else train(beta=0.0, seed=seed, **small)
        record("beta0_small", seed, result, beta=0.0, **small)

    print_matched_summary(rows)


def print_matched_summary(rows):
    columns = ["val_mse", "normal_err", "auroc_e1", "smoothed_d2", "smoothed_d4",
               "flatline_d2", "clipped_d2", "noise_d2"]
    print("\nmean±std over seeds")
    print(f"{'arm':<16} " + " ".join(f"{c:>15}" for c in columns))
    for arm in ["beta1", "beta0_earlystop", "beta0_small", "beta0_full"]:
        pts = [r for r in rows if r["arm"] == arm]
        print(f"{arm:<16} " + " ".join(
            f"{np.mean([r[c] for r in pts]):.4f}±{np.std([r[c] for r in pts]):.4f}".rjust(15)
            for c in columns))


# ---------- E3: synthetic anomalies on saved runs ----------

def run_e3(names, figure):
    splits = data.load_splits()
    e3 = load_e3_data(splits)
    x_test = to_tensor(splits["X_test"])

    all_rows = {}
    for name in names:
        model, config = load_run(name)
        rows = e3_rows(model, config["seed"], e3)
        e1 = summarize(model.score(x_test)["recon"].numpy(), splits["y_test"], data.NORMAL_CLASS)
        write_csv(rows, METRICS_DIR / f"E3_{name}.csv")
        all_rows[name] = rows

        print(f"\n{name}: E1 recon AUROC {e1['auroc']:.3f}, normal mse {e1['normal_err']:.4f}")
        for s in ["recon", "neg_elbo"]:
            print(f"  {s:<9}" + " ".join(f"{'d=' + format(d, 'g'):>6}" for d in anomalies.DISTANCES))
            for t in anomalies.TYPES:
                cells = [r[f"auroc_{s}"] for r in rows if r["anomaly_type"] == t]
                print(f"  {t:<9}" + " ".join(f"{a:>6.3f}" for a in cells))

    vae_plots.plot_e3(all_rows, read_metrics("pca_ecg_E3.csv"), FIGURES_DIR / f"{figure}.png")


# ---------- command line ----------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="experiment", required=True)

    e1 = commands.add_parser("e1", help="AUROC vs. latent size")
    e1.add_argument("--latent", type=int, nargs="+", default=[2, 8, 32, 128])
    e1.add_argument("--beta", type=float, default=1.0)
    e1.add_argument("--seed", type=int, default=0)
    e1.add_argument("--device", default="cpu", help="cpu, cuda or mps")

    sweep = commands.add_parser("beta-sweep", help="lower beta on one no-bottleneck network")
    sweep.add_argument("--betas", type=float, nargs="+", default=[1, 0.1, 0.01, 0.001, 0])
    sweep.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    sweep.add_argument("--channels", type=int, default=None, help="use a ConvVAE with this base width")
    sweep.add_argument("--device", default="cpu", help="cpu, cuda or mps")

    matched = commands.add_parser("matched-error", help="KL on vs. off at equal reconstruction error")
    matched.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    matched.add_argument("--device", default="cpu", help="cpu, cuda or mps")

    e3 = commands.add_parser("e3", help="synthetic anomalies, on saved runs")
    e3.add_argument("runs", nargs="+", help="run names in results/runs/")
    e3.add_argument("--figure", default="vae_ecg_E3", help="figure file name, without .png")

    args = parser.parse_args()
    if args.experiment == "e1":
        run_e1(args.latent, args.beta, args.seed, args.device)
    elif args.experiment == "beta-sweep":
        run_beta_sweep(args.betas, args.seeds, args.channels, args.device)
    elif args.experiment == "matched-error":
        run_matched_error(args.seeds, args.device)
    else:
        run_e3(args.runs, args.figure)


if __name__ == "__main__":
    main()
