"""Train one VAE and save or load it. The experiments that use this are in vae_experiments.py.

A run is saved to results/runs/<name>/ as model.pt, config.json and history.csv.
"""
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

from evaluate import write_csv
from vae import ConvVAE, MLPVAE

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "results" / "runs"

INPUT_DIM = 140  # points per ECG5000 beat


class TrainResult(NamedTuple):
    model: MLPVAE | ConvVAE  # at its best epoch, on the CPU
    history: list            # one dict of validation metrics per epoch
    best_epoch: int

    @property
    def best(self):
        """Validation metrics at the best epoch."""
        return self.history[self.best_epoch]


def build_model(latent_dim, hidden=(128, 64), channels=None, beta=1.0):
    """A ConvVAE if `channels` is given, otherwise an MLPVAE."""
    if channels is not None:
        return ConvVAE(INPUT_DIM, latent_dim, channels=channels, beta=beta)
    return MLPVAE(INPUT_DIM, latent_dim, hidden=hidden, beta=beta)


def train_vae(X_train, X_val, latent_dim, hidden=(128, 64), channels=None, beta=1.0, seed=0,
              select="loss", stop_at_mse=None, device="cpu",
              lr=1e-3, batch_size=128, max_epochs=3000, patience=200):
    """Train on normal beats and keep the weights from the best validation epoch.

    select      : what "best" means. "loss" is the validation negative ELBO, "mse" is
                  the validation MSE of decode(mu), which is the recon score itself
    stop_at_mse : if given, stop at the first epoch whose validation MSE reaches it
    patience    : stop after this many epochs without a new best
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    X_val = torch.tensor(X_val, dtype=torch.float32, device=device)

    model = build_model(latent_dim, hidden, channels, beta).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_metric, best_state, best_epoch = float("inf"), None, 0
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
        if metric < best_metric:
            best_metric, best_epoch = metric, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            break

    model.load_state_dict(best_state)
    model.cpu()
    model.eval()
    return TrainResult(model, history, best_epoch)


def save_run(name, result, config):
    run_dir = RUNS_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(result.model.state_dict(), run_dir / "model.pt")
    (run_dir / "config.json").write_text(json.dumps({**config, "best_epoch": result.best_epoch}, indent=2))
    write_csv(result.history, run_dir / "history.csv")


def load_run(name):
    """Return (model, config) for a saved run."""
    run_dir = RUNS_DIR / name
    config = json.loads((run_dir / "config.json").read_text())
    model = build_model(config["latent_dim"], tuple(config["hidden"]), config.get("channels"), config["beta"])
    model.load_state_dict(torch.load(run_dir / "model.pt"))
    model.eval()
    return model, config
