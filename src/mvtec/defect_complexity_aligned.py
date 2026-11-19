"""Aligned version for object categories (cable first): compare the defect with the SAME SPOT in normal images.

Object images are taken from a fixed camera, so a given pixel shows the same part of
the object in every normal image. For a "missing" defect the question is not "is it
smoother than its neighbours" but "is it smoother than what is normally there".

    normal detail map : the detail map (|gray - blurred gray|, as in defect_complexity.py)
                        averaged over the normal training images
    log2 ratio        : log2(mean detail inside the mask in this image /
                             mean of the normal detail map inside the same mask)
                        < 0 = this spot has less detail than it normally does (structure removed)
                        > 0 = more detail than normal (structure added)
    colour change     : distance between this image's mean RGB inside the mask and the
                        normal mean RGB there (a control: how visible the change is)
    area              : fraction of the image the mask covers

Null check: every defect mask is also laid on every normal TEST image. Those ratios
should sit around 0; how wide they spread shows how much normal cables differ from
each other (if they are badly aligned, the spread will be large).

    python -m src.mvtec.defect_complexity_aligned --category cable
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from src.mvtec import defect_complexity as dc
from src.common import paths
from src.mvtec import data as mvtec


def detail(img):
    gray = img.mean(axis=2)
    return np.abs(gray - gaussian_filter(gray, dc.BLUR))


def main(category):
    s = mvtec.load_splits(category, dc.SIZE)
    train = s["X_train"].reshape(-1, dc.SIZE, dc.SIZE, 3)
    test = s["X_test"].reshape(-1, dc.SIZE, dc.SIZE, 3)
    normal_detail = np.mean([detail(im) for im in train], axis=0)
    normal_rgb = train.mean(axis=0)

    def ratio(img, mask):
        return np.log2(detail(img)[mask].mean() / normal_detail[mask].mean())

    types = sorted(set(s["defect"]) - {"good"})
    per_type = {}
    for t in types:
        per_type[t] = [{"log2_ratio": ratio(test[i], s["masks"][i]), "area": s["masks"][i].mean(),
                        "colour": np.linalg.norm(test[i][s["masks"][i]].mean(axis=0) -
                                                 normal_rgb[s["masks"][i]].mean(axis=0))}
                       for i in np.flatnonzero(s["defect"] == t)]

    # Null: defect masks on normal test images
    normal_idx = np.flatnonzero(s["defect"] == "good")
    defect_idx = np.flatnonzero(s["defect"] != "good")
    null = np.array([ratio(test[i], s["masks"][j]) for i in normal_idx for j in defect_idx])
    lo, hi = np.percentile(null, [5, 95])

    auc = dc.aurocs(category)
    print(f"{category}: compared with the same spot in normal images (log2 ratio < 0 = detail removed)")
    print(f"null (defect masks on {len(normal_idx)} normal test images): median {np.median(null):+.2f}, "
          f"5-95% range {lo:+.2f} to {hi:+.2f}\n")
    print(f"{'defect':<22}{'n':>3}{'log2 ratio':>12}{'outside null':>14}{'area':>8}{'colour':>8}"
          f"{'AE recon':>10}{'VAE kl_max':>12}{'gain':>7}")
    for t in sorted(types, key=lambda t: np.median([m["log2_ratio"] for m in per_type[t]])):
        r = np.array([m["log2_ratio"] for m in per_type[t]])
        recon, kl = auc[t]
        print(f"{t:<22}{len(r):>3}{np.median(r):>12.2f}{f'{(r < lo).sum()} low {(r > hi).sum()} high':>14}"
              f"{np.median([m['area'] for m in per_type[t]]):>8.3f}{np.median([m['colour'] for m in per_type[t]]):>8.3f}"
              f"{recon:>10.2f}{kl:>12.2f}{kl - recon:>+7.2f}")

    # Figure: one example per type (mask outlined) above the per-image ratios, with the null band
    fig = plt.figure(figsize=(2.0 * len(types), 5.4), layout="constrained")
    grid = fig.add_gridspec(2, len(types), height_ratios=[1, 1.1])
    for j, t in enumerate(types):
        i = np.flatnonzero(s["defect"] == t)[0]
        ax = fig.add_subplot(grid[0, j])
        ax.imshow(test[i])
        ax.contour(s["masks"][i], levels=[0.5], colors="red", linewidths=1)
        ax.set_title(t, fontsize=7)
        ax.axis("off")
    ax = fig.add_subplot(grid[1, :])
    ax.axhspan(lo, hi, color="gray", alpha=.2, label="normal images, 5-95%")
    for j, t in enumerate(types):
        r = [m["log2_ratio"] for m in per_type[t]]
        ax.scatter(np.full(len(r), j) + np.random.default_rng(0).uniform(-.15, .15, len(r)), r, s=12, alpha=.7)
        ax.plot([j - .3, j + .3], [np.median(r)] * 2, color="k", lw=2)
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.set_xticks(range(len(types)), [f"{t}\nKL gain {auc[t][1] - auc[t][0]:+.2f}" for t in types], fontsize=7)
    ax.set_ylabel("log2(detail here / normally here)\n< 0 = detail removed")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle(f"{category}: detail inside the defect vs. the same spot in normal images", fontsize=10)
    out = paths.RESULTS_DIR / "mvtec" / "data" / f"defect_complexity_{category}.png"
    fig.savefig(out, dpi=150)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="cable")
    main(parser.parse_args().category)
