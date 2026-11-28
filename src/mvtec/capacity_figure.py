"""Capacity figure: AE reconstruction vs. VAE KL AUROC across latent sizes, for the four
detail-removing defect types where KL wins, plus the AE's error on normal test images.

As the bottleneck widens the AE reconstructs normal images better (last panel). On cable
its AUROC on the missing parts falls steeply (.98 -> .58, .61 -> .38): it starts copying
them. On tile it stays low (.54-.73) except gray_stroke at latent 4096. The VAE's KL beats
the AE on all four types from latent 256 up; at latent 64 neither is clearly better.
No title: the claim goes in the paper's caption.

    AE  : the better of the mean and max scores at each point (as in the paper table),
          which favours the AE
    KL  : the VAE's kl_max score
    dots: mean of 3 seeds; bars: the lowest and highest seed

    python -m src.mvtec.capacity_figure
"""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.common import paths

PANELS = [("tile", "gray_stroke"), ("tile", "glue_strip"), ("cable", "missing_cable"), ("cable", "missing_wire")]
AE_COLOR, KL_COLOR, INK, MUTED = "#eb6834", "#2a78d6", "#0b0b0b", "#52514e"


def load(category):
    return list(csv.DictReader(open(paths.RESULTS_DIR / "mvtec" / "vae" / f"{category}_128.csv")))


def curve(rows, beta, scores, column):
    """(latents, mean, low, high) over seeds; with several scores, the one with the higher mean per latent."""
    latents = sorted({int(r["latent_dim"]) for r in rows})
    out = []
    for z in latents:
        per_score = [np.array([float(r[column]) for r in rows if int(r["latent_dim"]) == z
                               and r["beta"] == beta and r["score"] == s]) for s in scores]
        v = max(per_score, key=np.mean)
        out.append((v.mean(), v.min(), v.max()))
    mean, low, high = map(np.array, zip(*out))
    return np.array(latents), mean, low, high


def main():
    data = {c: load(c) for c in ["tile", "cable"]}
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.1), layout="constrained",
                             gridspec_kw={"width_ratios": [1, 1, 1, 1, 0.9]})
    for ax, (category, defect) in zip(axes, PANELS):
        for beta, scores, color, label in [("0", ["mean", "max"], AE_COLOR, "AE reconstruction"),
                                           ("1", ["kl_max"], KL_COLOR, "VAE KL")]:
            z, m, lo, hi = curve(data[category], beta, scores, f"auroc_class{defect}")
            ax.errorbar(z, m, yerr=[m - lo, hi - m], color=color, lw=2, marker="o", ms=5,
                        capsize=0, elinewidth=1, label=label)
        ax.axhline(0.5, color=MUTED, lw=0.8, ls=":")
        ax.set_title(f"{category}: {defect.replace('_', ' ')}", fontsize=10, color=INK)
        ax.set_ylim(0.3, 1.03)
    axes[0].set_ylabel("image AUROC (defect type vs. normal)")
    axes[0].text(64, 0.51, "chance", fontsize=7, color=MUTED, va="bottom")

    ax = axes[4]  # reconstruction quality: the AE's error on normal test images
    for category, ls in [("tile", "-"), ("cable", "--")]:
        z, m, lo, hi = curve(data[category], "0", ["mean"], "normal_mse")
        ax.plot(z, m, ls, color=AE_COLOR, lw=2, marker="o", ms=5)
        ax.annotate(category, (z[0], m[0]), xytext=(6, 0), textcoords="offset points", fontsize=8,
                    color=INK, va="center")
    ax.set_ylim(0, None)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v * 1000:g}")
    ax.set_title("AE error on normal images", fontsize=10, color=INK)
    ax.set_ylabel("mean squared error (×10⁻³)")

    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks([64, 256, 1024, 4096, 16384], ["64", "256", "1024", "4096", "16k"])
        ax.minorticks_off()
        ax.set_xlabel("latent size")
        ax.grid(axis="y", color="#e6e5e0", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False, fontsize=9)
    out = paths.RESULTS_DIR / "mvtec" / "vae" / "capacity_detail_removed"
    for ext in ["png", "pdf"]:
        fig.savefig(f"{out}.{ext}", dpi=160, bbox_inches="tight")
    print(f"saved {out}.png and .pdf")


if __name__ == "__main__":
    main()
