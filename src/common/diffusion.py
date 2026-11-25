"""A small denoising diffusion model (DDPM) for images, used as a reconstruction model.

Training (on normal images only): take an image, pick a random noise step t in 1..T,
mix the image with Gaussian noise to that level, and train the U-Net to predict the
noise that was added. Loss = MSE between predicted and true noise.

Reconstruction (for anomaly scoring): noise a test image to a chosen step t_start,
then walk back to a clean image with deterministic DDIM steps. The model only knows
normal images, so whatever the noise destroyed is filled back in as normal. A small
t_start keeps most of the image (defects included); a large one repaints more.

    noise schedule : linear betas 1e-4 .. 0.02 over T = 1000 steps (standard DDPM)
    alpha_bar[t]   : how much of the original image is left at step t (alpha_bar[0] = 1)
    x_t            = sqrt(alpha_bar[t]) x + sqrt(1 - alpha_bar[t]) noise

Inputs and outputs are flat (height, width, RGB) vectors in [0, 1], like the VAEs, so
src/mvtec/scoring.py works unchanged. Internally images are (N, 3, H, W) in [-1, 1].
"""
import copy
import math
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

T = 1000


def timestep_embedding(t, dim):
    """Sinusoidal embedding of the noise step, as in transformers' position encoding."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=1)


class ResBlock(nn.Module):
    """Two 3x3 convs with GroupNorm + SiLU; the step embedding is added in between."""

    def __init__(self, c_in, c_out, t_dim):
        super().__init__()
        self.norm1, self.conv1 = nn.GroupNorm(8, c_in), nn.Conv2d(c_in, c_out, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, c_out)
        self.norm2, self.conv2 = nn.GroupNorm(8, c_out), nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(t_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    """Predicts the added noise from a noisy image and its step.

    Resolutions 128 -> 64 -> 32 -> 16 (at 128x128 input) with widths c, 2c, 4c, 4c.
    Unlike the VAE there is no bottleneck: skip connections carry each resolution
    straight to the decoder, so the network can keep fine detail.
    """

    def __init__(self, channels=32, t_dim=128):
        super().__init__()
        c = channels
        self.t_dim = t_dim
        self.t_mlp = nn.Sequential(nn.Linear(t_dim, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim))
        self.conv_in = nn.Conv2d(3, c, 3, padding=1)
        self.enc = nn.ModuleList([ResBlock(c, c, t_dim), ResBlock(c, 2 * c, t_dim), ResBlock(2 * c, 4 * c, t_dim)])
        self.down = nn.ModuleList([nn.Conv2d(w, w, 3, stride=2, padding=1) for w in [c, 2 * c, 4 * c]])
        self.mid = nn.ModuleList([ResBlock(4 * c, 4 * c, t_dim), ResBlock(4 * c, 4 * c, t_dim)])
        # decoder: upsample, concatenate the encoder output at that resolution, ResBlock
        self.up = nn.ModuleList([nn.Conv2d(w, w, 3, padding=1) for w in [4 * c, 4 * c, 2 * c]])
        self.dec = nn.ModuleList([ResBlock(8 * c, 4 * c, t_dim), ResBlock(6 * c, 2 * c, t_dim),
                                  ResBlock(3 * c, c, t_dim)])
        self.norm_out, self.conv_out = nn.GroupNorm(8, c), nn.Conv2d(c, 3, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.t_mlp(timestep_embedding(t, self.t_dim))
        h = self.conv_in(x)
        skips = []
        for block, down in zip(self.enc, self.down):
            h = block(h, t_emb)
            skips.append(h)
            h = down(h)
        for block in self.mid:
            h = block(h, t_emb)
        for block, up in zip(self.dec, self.up):
            h = up(F.interpolate(h, scale_factor=2, mode="nearest"))
            h = block(torch.cat([h, skips.pop()], dim=1), t_emb)
        return self.conv_out(F.silu(self.norm_out(h)))


class Diffusion(nn.Module):
    """U-Net plus the noise schedule; loss for training, reconstruct for scoring."""

    def __init__(self, image_size, channels=32):
        super().__init__()
        self.image_size = image_size
        self.unet = UNet(channels)
        betas = torch.linspace(1e-4, 0.02, T)
        self.register_buffer("alpha_bar", torch.cat([torch.ones(1), torch.cumprod(1 - betas, 0)]))

    def to_image(self, x):
        """Flat (N, H*W*3) in [0, 1] -> (N, 3, H, W) in [-1, 1]."""
        s = self.image_size
        return x.view(-1, s, s, 3).permute(0, 3, 1, 2) * 2 - 1

    def from_image(self, x):
        return ((x.clamp(-1, 1) + 1) / 2).permute(0, 2, 3, 1).reshape(len(x), -1)

    def add_noise(self, x, t, noise):
        a = self.alpha_bar[t][:, None, None, None]
        return a.sqrt() * x + (1 - a).sqrt() * noise

    def loss(self, x_flat, t=None, noise=None):
        """Noise-prediction MSE. t and noise are random unless given (validation uses fixed ones)."""
        x = self.to_image(x_flat)
        if t is None:
            t = torch.randint(1, T + 1, (len(x),), device=x.device)
        if noise is None:
            noise = torch.randn_like(x)
        return F.mse_loss(self.unet(self.add_noise(x, t, noise), t), noise)

    @torch.no_grad()
    def reconstruct(self, x_flat, t_start, noise):
        """Noise to step t_start with the given noise, then deterministic DDIM steps back to 0."""
        x = self.to_image(x_flat)
        if t_start == 0:
            return x_flat.clone()
        n_steps = min(25, max(5, t_start // 10))
        steps = sorted({int(round(s)) for s in np.linspace(t_start, 0, n_steps + 1)}, reverse=True)
        x_t = self.add_noise(x, torch.full((len(x),), t_start, device=x.device), noise)
        for t, t_prev in zip(steps[:-1], steps[1:]):
            a, a_prev = self.alpha_bar[t], self.alpha_bar[t_prev]
            eps = self.unet(x_t, torch.full((len(x),), t, device=x.device))
            x0 = ((x_t - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-1, 1)
            x_t = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
        return self.from_image(x_t)


def reconstruct_set(model, X, t_start, device="cpu", seed=0, batch_size=32):
    """Reconstruct a whole (n, H*W*3) numpy array. The noise for image i depends only on
    `seed` and i, so the same image always gets the same reconstruction."""
    noise = torch.randn(len(X), 3, model.image_size, model.image_size, generator=torch.Generator().manual_seed(seed))
    model = model.to(device).eval()
    out = []
    for i in range(0, len(X), batch_size):
        x = torch.tensor(X[i:i + batch_size], device=device)
        out.append(model.reconstruct(x, t_start, noise[i:i + batch_size].to(device)).cpu().numpy())
    return np.concatenate(out)


class TrainResult(NamedTuple):
    model: Diffusion        # EMA weights from the best validation check, on the CPU
    history: list           # one dict per validation check
    best_step: int


VAL_STEPS = [50, 100, 250, 500]  # the validation loss averages these, with fixed noise


def validation_loss(model, X_val, noise, device):
    model.eval()
    x = torch.tensor(X_val, device=device)
    with torch.no_grad():
        return float(np.mean([model.loss(x, torch.full((len(x),), t, device=device), noise.to(device)).item()
                              for t in VAL_STEPS]))


def train_diffusion(X_train, X_val, image_size, seed=0, device="cpu", channels=32, lr=2e-4, batch_size=32,
                    max_steps=40000, eval_every=500, patience=10, ema_decay=0.999):
    """Train on normal images; keep the EMA weights from the check with the lowest validation loss.

    EMA = an exponential moving average of the weights, standard for diffusion models
    because it gives much smoother samples than the raw weights. Validation loss is
    measured on the EMA weights every `eval_every` steps; training stops after
    `patience` checks without improvement, or at `max_steps`.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    X_train = torch.tensor(X_train, device=device)
    val_noise = torch.randn(len(X_val), 3, image_size, image_size, generator=torch.Generator().manual_seed(1))

    model = Diffusion(image_size, channels).to(device)
    ema = copy.deepcopy(model).eval()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state, best_step, history, step = float("inf"), None, 0, [], 0

    while step < max_steps:
        model.train()
        for idx in torch.randperm(len(X_train), device=device).split(batch_size):
            loss = model.loss(X_train[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                for p_ema, p in zip(ema.parameters(), model.parameters()):
                    p_ema.lerp_(p, 1 - ema_decay)
            step += 1
            if step % eval_every == 0:
                val = validation_loss(ema, X_val, val_noise, device)
                history.append({"step": step, "train_loss": loss.item(), "val_loss": val})
                if val < best:
                    best, best_step = val, step
                    best_state = {k: v.detach().clone() for k, v in ema.state_dict().items()}
                elif (step - best_step) // eval_every >= patience:
                    step = max_steps
            if step >= max_steps:
                break

    if best_state is None:  # max_steps smaller than eval_every (smoke tests): one check at the end
        history.append({"step": step, "train_loss": loss.item(),
                        "val_loss": validation_loss(ema, X_val, val_noise, device)})
        best_state, best_step = ema.state_dict(), step
    ema.load_state_dict(best_state)
    return TrainResult(ema.cpu().eval(), history, best_step)
