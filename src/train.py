"""Train VAEs on normal ECG5000 beats and run the E1 capacity sweep over latent size.

Each model is trained on normal training beats only, with early stopping on
the validation loss (negative ELBO on normal validation beats). Each run is
saved to results/runs/<name>/ (weights, config, training history).

    python src/train.py                              # latent 2 8 32 128, seed 0
    python src/train.py --latent 4 16 --seed 1 --beta 0.5
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

import data
from evaluate import summarize
from vae import MLPVAE

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "results" / "runs"
METRICS_DIR = ROOT / "results" / "metrics"
FIGURES_DIR = ROOT / "results" / "figures"

SCORES = ["recon", "kl", "neg_elbo"]
# Default categorical palette, fixed order: one color per score
SCORE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]


def train_vae(X_train, X_val, latent_dim, hidden=(128, 64), beta=1.0, seed=0,
              lr=1e-3, batch_size=128, max_epochs=3000, patience=200):
    """Train one VAE; return the model at its best validation loss and the history."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)

    model = MLPVAE(X_train.shape[1], latent_dim, hidden=hidden, beta=beta)
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
        history.append({"epoch": epoch, "val_loss": val_loss.item(), **parts})

        if val_loss.item() < best_loss:
            best_loss, best_epoch = val_loss.item(), epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            break

    model.load_state_dict(best_state)
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


def run_e1(latent_dims, beta, seed):
    splits = data.load_splits()
    X_test = torch.tensor(splits["X_test"], dtype=torch.float32)
    y_test = splits["y_test"]

    rows = []
    for latent_dim in latent_dims:
        name = f"vae_ecg5000_z{latent_dim}_b{beta:g}_seed{seed}"
        start = time.time()
        model, history, best_epoch = train_vae(splits["X_train"], splits["X_val"],
                                               latent_dim, beta=beta, seed=seed)
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
    args = parser.parse_args()
    run_e1(args.latent, args.beta, args.seed)


if __name__ == "__main__":
    main()
