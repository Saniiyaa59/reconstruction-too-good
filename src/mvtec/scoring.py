"""Anomaly maps and scores for MVTec test images, shared by every model.

An anomaly map gives every pixel a score. Scoring an image by the largest value
in its map, instead of its mean error, keeps a small defect from being averaged
away by the normal pixels around it.

    error map : squared reconstruction error per pixel (mean over RGB), Gaussian-
                smoothed with sigma = size / 32 pixels (4 at 128x128) so a single
                noisy pixel can't set the score
    KL map    : KL of each latent grid cell (VAE only), each cell's value spread
                over the image patch it covers

Scores per image:
    mean   : mean squared error over the whole image (the only score of the first runs)
    max    : max of the smoothed error map
    kl     : total KL, summed over cells (the ECG kl score)
    kl_max : max of the KL map

Metrics per score: image AUROC (normal vs. defective test images, overall and
per defect type) and, for map scores, two localization AUROCs against the
ground-truth masks:

    pixel AUROC : every test pixel, scored by its map value
    cell AUROC  : the 8 x 8 grid of the latent code, each cell scored by the
                  mean of its map values and defective if any of its pixels is.
                  The KL map only has one value per cell, so this grades the
                  error and KL maps at the same resolution
"""
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score

from src.common.evaluate import summarize


def error_maps(X, X_hat, size):
    """(n, size, size) squared error per pixel, averaged over RGB, from flat (height, width, RGB) images."""
    return ((X - X_hat) ** 2).reshape(len(X), size, size, 3).mean(axis=3)


def reconstruction_scores(X, X_hat, size):
    """(scores, maps): the mean and max scores, and the smoothed error maps behind max."""
    raw = error_maps(X, X_hat, size)
    smoothed = np.stack([gaussian_filter(m, sigma=size / 32) for m in raw])
    return {"mean": raw.mean(axis=(1, 2)), "max": smoothed.max(axis=(1, 2))}, {"max": smoothed}


def model_scores(model, X, size, batch_size=64):
    """Scores and maps for a ConvVAE2D; KL scores only if it was trained with the KL on (beta > 0)."""
    x = torch.tensor(X)
    with torch.no_grad():
        X_hat = np.concatenate([model.decode(model.encode(b)[0]).numpy() for b in x.split(batch_size)])
    scores, maps = reconstruction_scores(X, X_hat, size)
    if model.beta > 0:
        kl = np.concatenate([model.kl_map(b).numpy() for b in x.split(batch_size)])
        scores["kl"], scores["kl_max"] = kl.sum(axis=(1, 2)), kl.max(axis=(1, 2))
        cell = size // kl.shape[1]
        maps["kl_max"] = kl.repeat(cell, axis=1).repeat(cell, axis=2)
    return scores, maps


def to_cells(maps, grid=8):
    """(n, size, size) -> (n, grid, grid), each cell the mean of its block."""
    n, size, _ = maps.shape
    b = size // grid
    return maps.reshape(n, grid, b, grid, b).mean(axis=(2, 4))


def evaluate(scores, maps, splits):
    """One row per score. normal_mse (the mean score on normal test images) is the
    reconstruction quality, the same in every row, for plotting against."""
    normal = splits["defect"] == "good"
    rows = []
    cell_masks = to_cells(splits["masks"].astype(float)) > 0
    for name, s in scores.items():
        pixel = cell = float("nan")
        if name in maps:
            pixel = roc_auc_score(splits["masks"].ravel(), maps[name].ravel())
            cell = roc_auc_score(cell_masks.ravel(), to_cells(maps[name]).ravel())
        rows.append({"score": name, "normal_mse": float(scores["mean"][normal].mean()), "pixel_auroc": pixel,
                     "cell_auroc": cell, **summarize(s, splits["defect"], "good")})
    return rows
