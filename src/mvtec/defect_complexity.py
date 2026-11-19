"""Is each defect simpler or more complex than the texture around it? (tile first)

For every defective test image:
    detail map : |gray - blurred gray|, the fine detail at each pixel (blur sigma = 2 px)
    inside     : mean detail over the mask pixels
    ring       : mean detail over a band 8 px wide around the mask (normal texture nearby)
    log2 ratio : log2(inside / ring); < 0 = defect simpler than its surroundings, > 0 = more complex
Controls:
    area       : fraction of the image covered by the mask
    contrast   : distance between mean RGB colour inside the mask and in the ring

Then, per defect type, puts the medians next to AE recon vs VAE kl_max AUROC
(averaged over 5 latents and 3 seeds, from results/mvtec/vae/<category>_128.csv).

    python -m src.mvtec.defect_complexity --category tile
"""
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter

from src.common import paths
from src.mvtec import data as mvtec

SIZE, BLUR, RING = 128, 2, 8


def measures(img, mask):
    gray = img.mean(axis=2)
    detail = np.abs(gray - gaussian_filter(gray, BLUR))
    ring = binary_dilation(mask, iterations=RING) & ~mask
    inside, around = detail[mask].mean(), detail[ring].mean()
    return {"log2_ratio": np.log2(inside / around), "area": mask.mean(),
            "contrast": np.linalg.norm(img[mask].mean(axis=0) - img[ring].mean(axis=0))}, ring


def aurocs(category):
    """Per defect type: (AE recon, VAE kl_max), each averaged over latents and seeds; recon gets its better score."""
    rows = list(csv.DictReader(open(paths.RESULTS_DIR / "mvtec" / "vae" / f"{category}_{SIZE}.csv")))
    def mean(beta, score, t):
        return np.mean([float(r["auroc_class" + t]) for r in rows if float(r["beta"]) == beta and r["score"] == score])
    types = [k.removeprefix("auroc_class") for k in rows[0] if k.startswith("auroc_class")]
    return {t: (max(mean(0, "mean", t), mean(0, "max", t)), mean(1, "kl_max", t)) for t in types}


def main(category):
    s = mvtec.load_splits(category, SIZE)
    images = s["X_test"].reshape(-1, SIZE, SIZE, 3)
    types = sorted(set(s["defect"]) - {"good"})
    per_type, examples = {}, {}
    for t in types:
        idx = np.flatnonzero(s["defect"] == t)
        per_type[t] = [measures(images[i], s["masks"][i])[0] for i in idx]
        examples[t] = (images[idx[0]], s["masks"][idx[0]], measures(images[idx[0]], s["masks"][idx[0]])[1])

    auc = aurocs(category)
    print(f"{category}: medians per defect type (log2 ratio < 0 = simpler than surroundings)\n")
    print(f"{'defect':<14}{'n':>3}{'log2 ratio':>12}{'area':>8}{'contrast':>10}{'AE recon':>10}{'VAE kl_max':>12}{'gain':>7}")
    for t in sorted(types, key=lambda t: np.median([m["log2_ratio"] for m in per_type[t]])):
        med = {k: np.median([m[k] for m in per_type[t]]) for k in ["log2_ratio", "area", "contrast"]}
        recon, kl = auc[t]
        print(f"{t:<14}{len(per_type[t]):>3}{med['log2_ratio']:>12.2f}{med['area']:>8.3f}{med['contrast']:>10.3f}"
              f"{recon:>10.2f}{kl:>12.2f}{kl - recon:>+7.2f}")

    # Figure: one example per type (mask in red, ring in yellow), then log2 ratio per image by type
    fig = plt.figure(figsize=(2.2 * len(types), 5.2), layout="constrained")
    grid = fig.add_gridspec(2, len(types), height_ratios=[1, 1.1])
    for j, t in enumerate(types):
        img, mask, ring = examples[t]
        ax = fig.add_subplot(grid[0, j])
        overlay = img.copy()
        overlay[ring] = 0.5 * overlay[ring] + 0.5 * np.array([1, 0.85, 0])
        overlay[mask] = 0.5 * overlay[mask] + 0.5 * np.array([1, 0, 0])
        ax.imshow(overlay)
        ax.set_title(t, fontsize=8)
        ax.axis("off")
    ax = fig.add_subplot(grid[1, :])
    for j, t in enumerate(types):
        r = [m["log2_ratio"] for m in per_type[t]]
        ax.scatter(np.full(len(r), j) + np.random.default_rng(0).uniform(-.15, .15, len(r)), r, s=12, alpha=.7)
        ax.plot([j - .3, j + .3], [np.median(r)] * 2, color="k", lw=2)
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.set_xticks(range(len(types)), [f"{t}\nKL gain {auc[t][1] - auc[t][0]:+.2f}" for t in types], fontsize=8)
    ax.set_ylabel("log2(detail inside / ring)\n< 0 = simpler than surroundings")
    fig.suptitle(f"{category}: defect detail vs. nearby texture (red = mask, yellow = ring)", fontsize=10)
    out = paths.RESULTS_DIR / "mvtec" / "data" / f"defect_complexity_{category}.png"
    fig.savefig(out, dpi=150)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="tile")
    main(parser.parse_args().category)
