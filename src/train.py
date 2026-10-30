"""Train VAEs on normal ECG5000 beats and run the E1 capacity sweep over latent size.

Each model is trained on normal training beats only, with early stopping on
the validation loss (negative ELBO on normal validation beats). Each run is
saved to results/runs/<name>/ (weights, config, training history).

    python src/train.py                              # latent 2 8 32 128, seed 0
    python src/train.py --latent 4 16 --seed 1 --beta 0.5
    python src/train.py --beta-sweep                 # beta 1 to 0, no bottleneck, 3 seeds
    python src/train.py --beta-sweep --channels 32 --betas 0 --seeds 0   # conv VAE
    python src/train.py --e3 RUN [RUN ...]           # E3 on saved runs
    python src/train.py --matched-error              # KL on vs. off at equal reconstruction error
Add --device cuda to train on a GPU. This speeds up the conv VAE; small MLPs are faster on the CPU.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import anomalies
import data
from evaluate import summarize
from vae import ConvVAE, MLPVAE

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "results" / "runs"
METRICS_DIR = ROOT / "results" / "metrics"
FIGURES_DIR = ROOT / "results" / "figures"

SCORES = ["recon", "kl", "neg_elbo"]
# Default categorical palette, fixed order: one color per score
SCORE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
# Default categorical palette, fixed order; line style also marks the direction
TYPE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


def train_vae(X_train, X_val, latent_dim, hidden=(128, 64), beta=1.0, seed=0,
              lr=1e-3, batch_size=128, max_epochs=3000, patience=200, select="loss", channels=None,
              stop_at_mse=None, device="cpu"):
    """Train one VAE; return the model at its best validation metric and the history.

    select: "loss" stops on the validation negative ELBO, "mse" on the validation
    MSE of decode(mu), which is the recon score itself.
    channels: if given, train a ConvVAE with this base width instead of an MLPVAE.
    stop_at_mse: if given, stop at the first epoch whose validation MSE reaches it.
    The returned model is always on the CPU.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    X_val = torch.tensor(X_val, dtype=torch.float32, device=device)

    if channels is None:
        model = MLPVAE(X_train.shape[1], latent_dim, hidden=hidden, beta=beta)
    else:
        model = ConvVAE(X_train.shape[1], latent_dim, channels=channels, beta=beta)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss, best_state, best_epoch = float("inf"), None, 0
    history = []

    for epoch in range(max_epochs):
        model.train()
        for idx in torch.randperm(len(X_train)).split(batch_size):
            loss, _ = model.loss(X_train[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss, parts = model.loss(X_val)
            val_recon = (X_val - model.decode(model.encode(X_val)[0])).pow(2).mean().item()
        history.append({"epoch": epoch, "val_loss": val_loss.item(), "val_recon": val_recon, **parts})

        if stop_at_mse is not None and val_recon <= stop_at_mse:
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            break

        metric = val_recon if select == "mse" else val_loss.item()
        if metric < best_loss:
            best_loss, best_epoch = metric, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            break

    model.load_state_dict(best_state)
    model.cpu()
    model.eval()
    return model, history, best_epoch


def save_run(name, model, history, config):
    run_dir = RUNS_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_dir / "model.pt")
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    write_csv(history, run_dir / "history.csv")


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def run_e1(latent_dims, beta, seed, device="cpu"):
    splits = data.load_splits()
    X_test = torch.tensor(splits["X_test"], dtype=torch.float32)
    y_test = splits["y_test"]

    rows = []
    for latent_dim in latent_dims:
        name = f"vae_ecg5000_z{latent_dim}_b{beta:g}_seed{seed}"
        start = time.time()
        model, history, best_epoch = train_vae(splits["X_train"], splits["X_val"],
                                               latent_dim, beta=beta, seed=seed, device=device)
        best = history[best_epoch]
        save_run(name, model, history, {"latent_dim": latent_dim, "hidden": [128, 64],
                                        "beta": beta, "seed": seed, "best_epoch": best_epoch})

        torch.manual_seed(seed)  # neg_elbo samples z
        scores = {k: v.numpy() for k, v in model.score(X_test).items()}
        write_csv([{"sample_id": i, "class": int(c), **{s: float(scores[s][i]) for s in SCORES}}
                   for i, c in enumerate(y_test)],
                  METRICS_DIR / f"ecg5000_E1_vae_z{latent_dim}_b{beta:g}_seed{seed}.csv")

        active = model.active_units(torch.tensor(splits["X_val"], dtype=torch.float32))
        for s in SCORES:
            rows.append({"latent_dim": latent_dim, "score": s, "best_epoch": best_epoch,
                         "val_mse": best["mse"], "sigma_x": best["sigma_x"],
                         "active_units": active,
                         **summarize(scores[s], y_test, data.NORMAL_CLASS)})
        print(f"z={latent_dim:>4}: best epoch {best_epoch:>4}, val mse {best['mse']:.4f}, "
              f"sigma_x {best['sigma_x']:.3f}, active {active}/{latent_dim}, "
              + ", ".join(f"AUROC {r['score']} {r['auroc']:.3f}" for r in rows[-3:])
              + f"  ({time.time() - start:.0f}s)")

    tag = f"b{beta:g}_seed{seed}"
    write_csv(rows, METRICS_DIR / f"vae_ecg_E1_{tag}.csv")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_e1(rows, latent_dims, FIGURES_DIR / f"vae_ecg_E1_{tag}.png",
            title=f"VAE capacity sweep on ECG5000 (β = {beta:g}, seed {seed})")


def run_beta_sweep(betas, seeds, latent_dim=256, hidden=(512, 256), channels=None, device="cpu"):
    """Can the VAE reach the error range where PCA collapsed? Lower beta on one architecture.

    The latent size is wider than the 140-point input, and so are the MLP's hidden
    layers, so the network itself has no bottleneck. `channels` switches to a
    ConvVAE. Checkpoints are selected by validation MSE.
    """
    arch = "" if channels is None else f"_conv{channels}"
    splits = data.load_splits()
    X_test = torch.tensor(splits["X_test"], dtype=torch.float32)
    y_test = splits["y_test"]

    rows = []
    for beta in betas:
        for seed in seeds:
            start = time.time()
            model, history, best_epoch = train_vae(splits["X_train"], splits["X_val"], latent_dim,
                                                   hidden=hidden, beta=beta, seed=seed, select="mse",
                                                   channels=channels, device=device)
            save_run(f"vae_ecg5000_sweep{arch}_b{beta:g}_seed{seed}", model, history,
                     {"latent_dim": latent_dim, "hidden": list(hidden), "channels": channels,
                      "beta": beta, "seed": seed, "best_epoch": best_epoch, "select": "mse"})
            torch.manual_seed(seed)  # neg_elbo samples z
            scores = {k: v.numpy() for k, v in model.score(X_test).items()}
            active = model.active_units(torch.tensor(splits["X_val"], dtype=torch.float32))
            row = {"beta": beta, "seed": seed, "best_epoch": best_epoch,
                   "sigma_x": history[best_epoch]["sigma_x"], "active_units": active}
            for s in SCORES:
                summary = summarize(scores[s], y_test, data.NORMAL_CLASS)
                row[f"auroc_{s}"] = summary["auroc"]
                if s == "recon":
                    row["normal_err"], row["anomaly_err"] = summary["normal_err"], summary["anomaly_err"]
            rows.append(row)
            print(f"beta={beta:<6g} seed={seed}: best epoch {best_epoch:>4}, "
                  f"normal mse {row['normal_err']:.4f}, anomaly mse {row['anomaly_err']:.4f}, "
                  f"AUROC recon {row['auroc_recon']:.3f} kl {row['auroc_kl']:.3f} "
                  f"neg_elbo {row['auroc_neg_elbo']:.3f}  ({time.time() - start:.0f}s)", flush=True)
            write_csv(rows, METRICS_DIR / f"vae_ecg_beta_sweep{arch}.csv")  # after each run, in case of a stop

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_beta_sweep(rows, FIGURES_DIR / f"vae_ecg_beta_sweep{arch}.png")


def plot_beta_sweep(rows, path):
    """AUROC (recon) vs. normal test error, VAE seeds against the PCA curve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.2))
    with open(METRICS_DIR / "pca_ecg.csv") as f:
        pca = list(csv.DictReader(f))
    ax.plot([float(r["normal_err"]) for r in pca], [float(r["auroc"]) for r in pca], "s--",
            color="0.45", lw=2, ms=6, label="PCA (k = 1 to 120)")
    ax.plot([r["normal_err"] for r in rows], [r["auroc_recon"] for r in rows], "o",
            color=SCORE_COLORS[0], ms=7, label="VAE (one point per β and seed)")
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_xscale("log")
    ax.invert_xaxis()  # better reconstruction to the right
    ax.set_xlabel("mean reconstruction error on normal test beats (MSE, log)")
    ax.set_ylabel("AUROC (recon score)")
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", color="0.9", lw=0.8)
    ax.legend(loc="lower left", frameon=False)
    ax.set_title("VAE β sweep vs. PCA on ECG5000", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def load_run(name):
    """Rebuild a saved model from results/runs/<name>/."""
    run_dir = RUNS_DIR / name
    config = json.loads((run_dir / "config.json").read_text())
    if config.get("channels"):
        model = ConvVAE(140, config["latent_dim"], channels=config["channels"], beta=config["beta"])
    else:
        model = MLPVAE(140, config["latent_dim"], hidden=tuple(config["hidden"]), beta=config["beta"])
    model.load_state_dict(torch.load(run_dir / "model.pt"))
    model.eval()
    return model, config


def e3_rows(model, seed, x_normal, x_anom, anoms):
    """AUROC of test normals vs. each synthetic type at each distance, for every score."""
    torch.manual_seed(seed)  # neg_elbo samples z
    s_normal = {k: v.numpy() for k, v in model.score(x_normal).items()}
    s_anom = {k: v.numpy() for k, v in model.score(x_anom).items()}
    rows = []
    for t, (_, direction, _, _) in anomalies.TYPES.items():
        for d in anomalies.DISTANCES:
            m = (anoms["anomaly_type"] == t) & (anoms["distance"] == d)
            labels = np.r_[np.zeros(len(x_normal)), np.ones(m.sum())]
            row = {"anomaly_type": t, "direction": direction, "distance": d, "n": int(m.sum())}
            for s in SCORES:
                row[f"auroc_{s}"] = roc_auc_score(labels, np.r_[s_normal[s], s_anom[s][m]])
            rows.append(row)
    return rows


def run_e3(names, figure="vae_ecg_E3"):
    """E3 on saved runs: AUROC of test normals vs. each synthetic type at each distance."""
    splits = data.load_splits()
    X_normal = splits["X_test"][splits["y_test"] == data.NORMAL_CLASS]
    anoms = anomalies.generate(X_normal)  # same anomalies as the PCA E3 run
    x_normal = torch.tensor(X_normal, dtype=torch.float32)
    x_anom = torch.tensor(anoms["X"], dtype=torch.float32)
    x_test = torch.tensor(splits["X_test"], dtype=torch.float32)

    all_rows = {}
    for name in names:
        model, config = load_run(name)
        rows = e3_rows(model, config["seed"], x_normal, x_anom, anoms)
        e1 = summarize(model.score(x_test)["recon"].numpy(), splits["y_test"], data.NORMAL_CLASS)
        write_csv(rows, METRICS_DIR / f"E3_{name}.csv")
        all_rows[name] = rows

        print(f"\n{name}: E1 recon AUROC {e1['auroc']:.3f}, normal mse {e1['normal_err']:.4f}")
        for s in ["recon", "neg_elbo"]:
            print(f"  {s:<9}" + " ".join(f"{'d=' + format(d, 'g'):>6}" for d in anomalies.DISTANCES))
            for t in anomalies.TYPES:
                cells = [r[f"auroc_{s}"] for r in rows if r["anomaly_type"] == t]
                print(f"  {t:<9}" + " ".join(f"{a:>6.3f}" for a in cells))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_e3(all_rows, FIGURES_DIR / f"{figure}.png")


def run_matched_error(seeds, device="cpu", latent_grid=(2, 3, 4, 6, 8, 12, 16)):
    """Does the KL help beyond limiting reconstruction? Compare models at equal error.

    Arms, all selected on validation data only:
        beta1           : no-bottleneck MLP with the KL on; its validation MSE is the target
        beta0_full      : same network with the KL off, trained as far as it improves
        beta0_earlystop : same network with the KL off, stopped when it reaches the target
        beta0_small     : smaller network with the KL off, latent size closest to the target
    """
    splits = data.load_splits()
    X_normal = splits["X_test"][splits["y_test"] == data.NORMAL_CLASS]
    anoms = anomalies.generate(X_normal)  # same anomalies as the PCA E3 run
    x_normal = torch.tensor(X_normal, dtype=torch.float32)
    x_anom = torch.tensor(anoms["X"], dtype=torch.float32)
    x_test = torch.tensor(splits["X_test"], dtype=torch.float32)
    wide = {"latent_dim": 256, "hidden": (512, 256)}
    small = {"hidden": (128, 64)}

    def train(**kwargs):
        return train_vae(splits["X_train"], splits["X_val"], select="mse", device=device, **kwargs)

    rows = []

    def record(arm, seed, trained, config):
        model, history, best_epoch = trained
        save_run(f"vae_ecg5000_matched_{arm}_seed{seed}", model, history,
                 {**config, "hidden": list(config["hidden"]), "seed": seed,
                  "best_epoch": best_epoch, "select": "mse"})
        e1 = summarize(model.score(x_test)["recon"].numpy(), splits["y_test"], data.NORMAL_CLASS)
        row = {"arm": arm, "seed": seed, "latent_dim": config["latent_dim"],
               "hidden": "x".join(map(str, config["hidden"])), "beta": config["beta"],
               "best_epoch": best_epoch, "val_mse": history[best_epoch]["val_recon"],
               "normal_err": e1["normal_err"], "anomaly_err": e1["anomaly_err"], "auroc_e1": e1["auroc"]}
        for r in e3_rows(model, seed, x_normal, x_anom, anoms):
            row[f"{r['anomaly_type']}_d{r['distance']:g}"] = r["auroc_recon"]
        rows.append(row)
        write_csv(rows, METRICS_DIR / "vae_ecg_matched_error.csv")  # after each run, in case of a stop
        print(f"{arm:<16} seed={seed}: val mse {row['val_mse']:.4f}, normal mse {row['normal_err']:.4f}, "
              f"E1 AUROC {row['auroc_e1']:.3f}, smoothed d=2 {row['smoothed_d2']:.3f}, "
              f"d=4 {row['smoothed_d4']:.3f}", flush=True)

    for seed in seeds:
        record("beta1", seed, train(beta=1.0, seed=seed, **wide), {"beta": 1.0, **wide})
    target = float(np.mean([r["val_mse"] for r in rows]))
    print(f"target validation MSE: {target:.4f}", flush=True)

    # Pick the small network's latent size on validation MSE, using the first seed
    candidates = {z: train(beta=0.0, seed=seeds[0], latent_dim=z, **small) for z in latent_grid}
    for z, (_, history, best_epoch) in candidates.items():
        print(f"  small network, latent {z:>2}: val mse {history[best_epoch]['val_recon']:.4f}", flush=True)
    best_z = min(candidates, key=lambda z: abs(candidates[z][1][candidates[z][2]]["val_recon"] - target))

    for seed in seeds:
        record("beta0_full", seed, train(beta=0.0, seed=seed, **wide), {"beta": 0.0, **wide})
        record("beta0_earlystop", seed, train(beta=0.0, seed=seed, stop_at_mse=target, **wide),
               {"beta": 0.0, **wide})
        trained = candidates[best_z] if seed == seeds[0] else train(beta=0.0, seed=seed, latent_dim=best_z, **small)
        record("beta0_small", seed, trained, {"beta": 0.0, "latent_dim": best_z, **small})

    columns = ["val_mse", "normal_err", "auroc_e1", "smoothed_d2", "smoothed_d4", "flatline_d2", "clipped_d2", "noise_d2"]
    print("\nmean±std over seeds")
    print(f"{'arm':<16} " + " ".join(f"{c:>15}" for c in columns))
    for arm in ["beta1", "beta0_earlystop", "beta0_small", "beta0_full"]:
        pts = [r for r in rows if r["arm"] == arm]
        print(f"{arm:<16} " + " ".join(
            f"{np.mean([r[c] for r in pts]):.4f}±{np.std([r[c] for r in pts]):.4f}".rjust(15)
            for c in columns))


def plot_e3(all_rows, path):
    """AUROC (recon) vs. distance, one line per type: PCA k=50 next to each saved run."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(METRICS_DIR / "pca_ecg_E3.csv") as f:
        pca = [{**r, "auroc_recon": float(r["auroc"]), "distance": float(r["distance"])}
               for r in csv.DictReader(f) if r["k"] == "50"]
    panels = {"PCA k=50": pca, **all_rows}
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.4), sharey=True, squeeze=False)
    for ax, (name, rows) in zip(axes[0], panels.items()):
        for color, (t, (_, direction, _, _)) in zip(TYPE_COLORS, anomalies.TYPES.items()):
            pts = [r for r in rows if r["anomaly_type"] == t]
            ax.plot([r["distance"] for r in pts], [r["auroc_recon"] for r in pts],
                    "o-" if direction == "simpler" else "s--", color=color, lw=2, ms=6,
                    label=f"{t} ({direction})")
        ax.axhline(0.5, color="gray", ls=":", lw=1)
        ax.set_title(name.replace("vae_ecg5000_sweep_", ""), fontsize=10)
        ax.set_xlabel("distance d")
        ax.set_xticks(anomalies.DISTANCES)
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", color="0.9", lw=0.8)
    axes[0][0].set_ylabel("AUROC (test normals vs. synthetic)")
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="lower center",
               ncol=len(anomalies.TYPES), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_e1(rows, latent_dims, path, title):
    """Left: AUROC vs. latent size per score. Right: AUROC vs. normal test error, VAE vs. PCA."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    # Dashed neg_elbo so it stays visible when it lies on top of recon
    for s, color, style in zip(SCORES, SCORE_COLORS, ["o-", "o-", "s--"]):
        pts = [r for r in rows if r["score"] == s]
        ax.plot([r["latent_dim"] for r in pts], [r["auroc"] for r in pts], style,
                color=color, lw=2, ms=6, label=s)
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xticks(latent_dims, [str(d) for d in latent_dims])
    ax.set_xlabel("latent dimension")
    ax.set_ylabel("AUROC (test normals vs. all anomalies)")
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", color="0.9", lw=0.8)
    ax.legend(title="score", loc="lower left", frameon=False)
    ax.set_title(title, fontsize=11)

    # Compare models at equal reconstruction quality (recon score only)
    vae_pts = [r for r in rows if r["score"] == "recon"]
    ax2.plot([r["normal_err"] for r in vae_pts], [r["auroc"] for r in vae_pts], "o-",
             color=SCORE_COLORS[0], lw=2, ms=6, label="VAE (latent dim)")
    # The larger latent sizes sit almost on one spot, so label them as a group
    labels = [(vae_pts[0], f"z={vae_pts[0]['latent_dim']}"),
              (vae_pts[-1], f"z={vae_pts[1]['latent_dim']}–{vae_pts[-1]['latent_dim']}")]
    for r, text in labels[:len(vae_pts)]:
        ax2.annotate(text, (r["normal_err"], r["auroc"]), textcoords="offset points",
                     xytext=(0, -14), ha="center", fontsize=8, color=SCORE_COLORS[0])
    pca_csv = METRICS_DIR / "pca_ecg.csv"
    if pca_csv.exists():
        with open(pca_csv) as f:
            pca = list(csv.DictReader(f))
        ax2.plot([float(r["normal_err"]) for r in pca], [float(r["auroc"]) for r in pca], "s--",
                 color="0.45", lw=2, ms=6, label="PCA (k)")
        for r in pca:
            if r["k"] in ("1", "20", "50", "80", "120"):
                ax2.annotate(r["k"], (float(r["normal_err"]), float(r["auroc"])),
                             textcoords="offset points", xytext=(6, -10), fontsize=8, color="0.35")
    ax2.axhline(0.5, color="gray", ls=":", lw=1)
    ax2.set_xscale("log")
    ax2.invert_xaxis()  # better reconstruction to the right
    ax2.set_xlabel("mean reconstruction error on normal test beats (MSE, log)")
    ax2.set_ylim(0, 1.02)
    ax2.grid(axis="y", color="0.9", lw=0.8)
    ax2.legend(loc="lower left", frameon=False)
    ax2.set_title("AUROC (recon score) at matched reconstruction quality", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent", type=int, nargs="+", default=[2, 8, 32, 128])
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--beta-sweep", action="store_true",
                        help="run the beta sweep (no-bottleneck VAE, 3 seeds) instead of E1")
    parser.add_argument("--betas", type=float, nargs="+", default=[1, 0.1, 0.01, 0.001, 0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--channels", type=int, default=None,
                        help="beta sweep only: use a ConvVAE with this base width")
    parser.add_argument("--e3", nargs="+", metavar="RUN", help="run E3 on these saved runs")
    parser.add_argument("--e3-figure", default="vae_ecg_E3", help="E3 figure file name, without .png")
    parser.add_argument("--matched-error", action="store_true",
                        help="compare KL on vs. off at equal reconstruction error")
    parser.add_argument("--device", default="cpu", help="cpu, cuda or mps")
    args = parser.parse_args()
    if args.e3:
        run_e3(args.e3, args.e3_figure)
    elif args.matched_error:
        run_matched_error(args.seeds, args.device)
    elif args.beta_sweep:
        run_beta_sweep(args.betas, args.seeds, channels=args.channels, device=args.device)
    else:
        run_e1(args.latent, args.beta, args.seed, args.device)


if __name__ == "__main__":
    main()
