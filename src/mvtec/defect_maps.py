"""One defect type, one saved VAE: maps picture and recon-vs-KL scatter.

Picture: 2 normal and the first 4 defective test images (file order, not hand-picked),
each as input (mask in red), reconstruction, smoothed error map and KL map.
Scatter: recon score (max) vs. KL score (kl_max), one dot per normal or defective test
image, with thresholds at the 95th percentile of the normal test images' scores.

    python -m src.mvtec.defect_maps                      # tile / gray_stroke, VAE latent 1024, seed 0
    python -m src.mvtec.defect_maps --category cable --defect missing_cable --run mvtec_cable_128_b1_z1024_seed0
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from src.common import paths
from src.common.train import load_run
from src.mvtec import data
from src.mvtec.scoring import model_scores

parser = argparse.ArgumentParser()
parser.add_argument("--category", default="tile")
parser.add_argument("--defect", default="gray_stroke")
parser.add_argument("--run", default="mvtec_tile_128_b1_z1024_seed0", help="a saved VAE (beta > 0) in results/runs/")
args = parser.parse_args()
DEFECT = args.defect
POOLED = args.run.startswith("mvtec_all_")  # trained on all categories' normals (run-pooled)
MODEL = "pooled VAE" if POOLED else "VAE"
out = paths.RESULTS_DIR / "mvtec" / ("vae_pooled" if POOLED else "vae")
out.mkdir(parents=True, exist_ok=True)
name = f"defect_{args.category}_{DEFECT}"

# ---------- Part 1: load and sanity-check ----------
model, config = load_run(args.run)
SIZE = config["image_size"]
splits = data.load_splits(args.category, SIZE)
scores, maps = model_scores(model, splits["X_test"], SIZE)

defect = splits["defect"]
normal, stroke = defect == "good", defect == DEFECT
both = normal | stroke
for s in ["max", "kl_max"]:
    auroc = roc_auc_score(stroke[both], scores[s][both])
    print(f"AUROC {DEFECT} vs normal, {s}: {auroc:.3f}")
print(f"{normal.sum()} normal, {stroke.sum()} {DEFECT} test images")

# ---------- Part 2: picture ----------
rows = list(np.flatnonzero(normal)[:2]) + list(np.flatnonzero(stroke)[:4])  # first in file order
x = torch.tensor(splits["X_test"][rows])
with torch.no_grad():
    recon = model.decode(model.encode(x)[0]).numpy()
err_vmax = maps["max"][rows].max()
kl_vmax = maps["kl_max"][rows].max()

fig, axes = plt.subplots(len(rows), 4, figsize=(8.5, 2.2 * len(rows) + 0.8), layout="constrained")
for i, idx in enumerate(rows):
    img = splits["X_test"][idx].reshape(SIZE, SIZE, 3)
    axes[i, 0].imshow(img)
    if splits["masks"][idx].any():
        axes[i, 0].contour(splits["masks"][idx], levels=[0.5], colors="red", linewidths=0.8)
    axes[i, 1].imshow(np.clip(recon[i].reshape(SIZE, SIZE, 3), 0, 1))
    e = axes[i, 2].imshow(maps["max"][idx], cmap="magma", vmin=0, vmax=err_vmax)
    k = axes[i, 3].imshow(maps["kl_max"][idx], cmap="viridis", vmin=0, vmax=kl_vmax)
    axes[i, 0].set_ylabel(f"{defect[idx]}\nrecon {scores['max'][idx]:.3f}\nKL {scores['kl_max'][idx]:.0f}", fontsize=8)
    for ax in axes[i]:
        ax.set_xticks([]); ax.set_yticks([])
for ax, title in zip(axes[0], ["input (mask in red)", f"{MODEL} reconstruction", "error map (smoothed)", "KL map (8x8 cells)"]):
    ax.set_title(title, fontsize=9)
fig.colorbar(e, ax=axes[-1, 2], location="bottom", shrink=0.9, label="squared error")
fig.colorbar(k, ax=axes[-1, 3], location="bottom", shrink=0.9, label="KL per cell")
fig.suptitle(f"{args.category}, {MODEL} (latent {config['latent_dim']}, seed {config['seed']}): "
             f"2 normal and the first 4 {DEFECT} test images", fontsize=10)
fig.savefig(out / f"{name}_maps.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# ---------- Part 3: scatter ----------
r_thr = np.percentile(scores["max"][normal], 95)
k_thr = np.percentile(scores["kl_max"][normal], 95)
r_hit = scores["max"][stroke] > r_thr
k_hit = scores["kl_max"][stroke] > k_thr
print(f"\nthresholds = 95th percentile of normal test images")
print(f"{DEFECT} above recon threshold: {r_hit.sum()}/{stroke.sum()}")
print(f"{DEFECT} above KL threshold:    {k_hit.sum()}/{stroke.sum()}")
print(f"KL catches, recon misses:     {(k_hit & ~r_hit).sum()}/{stroke.sum()}")
print(f"recon catches, KL misses:     {(r_hit & ~k_hit).sum()}/{stroke.sum()}")

fig, ax = plt.subplots(figsize=(5.5, 4.5))
ax.scatter(scores["max"][normal], scores["kl_max"][normal], c="0.55", s=22, label=f"normal ({normal.sum()})")
ax.scatter(scores["max"][stroke], scores["kl_max"][stroke], c="#eb6834", s=26, label=f"{DEFECT} ({stroke.sum()})")
ax.axvline(r_thr, color="0.4", ls="--", lw=0.8)
ax.axhline(k_thr, color="0.4", ls="--", lw=0.8)
ax.set_xlabel("recon score (max of smoothed error map)")
ax.set_ylabel("KL score (max over latent cells)")
ax.set_title(f"{args.category}, {MODEL} (latent {config['latent_dim']}, seed {config['seed']}); "
             "dashed = 95th pct of normals", fontsize=9)
ax.legend(fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(out / f"{name}_scatter.png", dpi=130)
