"""All texture defect types: median log2(detail inside / ring) vs. KL gain (VAE kl_max - AE recon).
Also: does the answer depend on the blur and ring settings?

    python -m src.mvtec.textures_scatter
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from src.mvtec import defect_complexity as dc
from src.common import paths
from src.mvtec import data as mvtec

TEXTURES = ["carpet", "grid", "leather", "tile", "wood"]


def medians(blur, ring):
    dc.BLUR, dc.RING = blur, ring
    out = []
    for c in TEXTURES:
        s = mvtec.load_splits(c, dc.SIZE)
        images = s["X_test"].reshape(-1, dc.SIZE, dc.SIZE, 3)
        auc = dc.aurocs(c)
        for t in sorted(set(s["defect"]) - {"good"}):
            r = [dc.measures(images[i], s["masks"][i])[0]["log2_ratio"] for i in np.flatnonzero(s["defect"] == t)]
            out.append((c, t, np.median(r), auc[t][1] - auc[t][0], auc[t][0]))
    return out


print("Spearman correlation, median log2 ratio vs. KL gain, 25 texture types (negative = smoother -> bigger gain)")
for blur in [1, 2, 4]:
    for ring in [4, 8, 16]:
        rows = medians(blur, ring)
        rho, p = spearmanr([r[2] for r in rows], [r[3] for r in rows])
        smooth = [r for r in rows if r[2] < 0]
        print(f"  blur {blur} ring {ring:>2}: rho {rho:+.2f} (p {p:.3f}) | smoother types {len(smooth):>2}, "
              f"KL wins >.05 among them {sum(r[3] > .05 for r in smooth)}; "
              f"busier types {len(rows) - len(smooth):>2}, KL wins among them {sum(r[3] > .05 for r in rows if r[2] >= 0)}")

rows = medians(2, 8)
fig, ax = plt.subplots(figsize=(7, 5))
colors = dict(zip(TEXTURES, plt.cm.tab10.colors))
for c, t, ratio, gain, recon in rows:
    ax.scatter(ratio, gain, color=colors[c], s=40)
    ax.annotate(t, (ratio, gain), fontsize=6, xytext=(3, 3), textcoords="offset points")
for c in TEXTURES:
    ax.scatter([], [], color=colors[c], label=c)
ax.axhline(0, color="gray", ls="--", lw=1)
ax.axvline(0, color="gray", ls="--", lw=1)
ax.set_xlabel("median log2(detail inside mask / ring)   (< 0 = smoother than surroundings)")
ax.set_ylabel("KL gain: VAE kl_max - AE recon AUROC")
ax.set_title("Texture defect types: smoothness of the defect vs. what KL adds", fontsize=10)
ax.legend(fontsize=8, frameon=False)
fig.tight_layout()
out = paths.RESULTS_DIR / "mvtec" / "data" / "defect_complexity_textures.png"
fig.savefig(out, dpi=150)
print(f"saved {out}")
