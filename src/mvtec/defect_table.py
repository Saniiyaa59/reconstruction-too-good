"""One row per defect type (73): detail removed or not, and the KL gain (VAE kl_max - AE recon)
averaged over all 5 latents vs. at latent 1024 only (both averaged over 3 seeds).

detail removed:
    textures : median log2(detail inside mask / ring) < 0           (defect_complexity.py)
    objects  : median log2(detail here / normally here) below the 5th
               percentile of the null (defect masks on normal test images)  (defect_complexity_aligned.py)
AE recon gets its better score (mean or max) in each column. Writes results/mvtec/data/defect_types.csv.

    python -m src.mvtec.defect_table
"""
import csv

import numpy as np

from src.mvtec import defect_complexity as dc
from src.mvtec.defect_complexity_aligned import detail
from src.common import paths
from src.mvtec import data as mvtec

TEXTURES = ["carpet", "grid", "leather", "tile", "wood"]
OBJECTS = ["bottle", "cable", "capsule", "hazelnut", "metal_nut", "pill", "screw", "toothbrush", "transistor", "zipper"]


def gains(category, latents):
    rows = list(csv.DictReader(open(paths.RESULTS_DIR / "mvtec" / "vae" / f"{category}_{dc.SIZE}.csv")))
    rows = [r for r in rows if r["latent_dim"] in latents]
    def mean(beta, score, t):
        return np.mean([float(r["auroc_class" + t]) for r in rows if float(r["beta"]) == beta and r["score"] == score])
    types = [k.removeprefix("auroc_class") for k in rows[0] if k.startswith("auroc_class")]
    out = {}
    for t in types:
        recon = max(mean(0, "mean", t), mean(0, "max", t))
        out[t] = (recon, mean(1, "kl_max", t))
    return out


table = []
for c in TEXTURES + OBJECTS:
    s = mvtec.load_splits(c, dc.SIZE)
    test = s["X_test"].reshape(-1, dc.SIZE, dc.SIZE, 3)
    types = sorted(set(s["defect"]) - {"good"})
    if c in TEXTURES:
        med = {t: np.median([dc.measures(test[i], s["masks"][i])[0]["log2_ratio"]
                             for i in np.flatnonzero(s["defect"] == t)]) for t in types}
        removed = {t: med[t] < 0 for t in types}
    else:
        train = s["X_train"].reshape(-1, dc.SIZE, dc.SIZE, 3)
        normal = np.mean([detail(im) for im in train], axis=0)
        d = {i: detail(test[i]) for i in range(len(test))}
        ratio = lambda i, m: np.log2(d[i][m].mean() / normal[m].mean())
        null = [ratio(i, s["masks"][j]) for i in np.flatnonzero(s["defect"] == "good")
                for j in np.flatnonzero(s["defect"] != "good")]
        lo = np.percentile(null, 5)
        med = {t: np.median([ratio(i, s["masks"][i]) for i in np.flatnonzero(s["defect"] == t)]) for t in types}
        removed = {t: med[t] < lo for t in types}
    g_all, g_1024 = gains(c, {"64", "256", "1024", "4096", "16384"}), gains(c, {"1024"})
    for t in types:
        table.append({"category": c, "defect": t, "kind": "texture" if c in TEXTURES else "object",
                      "log2_ratio": med[t], "removed": removed[t],
                      "recon_all": g_all[t][0], "kl_all": g_all[t][1], "gain_all": g_all[t][1] - g_all[t][0],
                      "recon_1024": g_1024[t][0], "kl_1024": g_1024[t][1], "gain_1024": g_1024[t][1] - g_1024[t][0]})

out = paths.RESULTS_DIR / "mvtec" / "data" / "defect_types.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(table[0]))
    w.writeheader()
    w.writerows(table)

print(f"{'category':<11}{'defect':<22}{'removed':>8}{'ratio':>7} | {'AE':>5}{'KL':>6}{'gain':>7}  (all latents) | "
      f"{'AE':>5}{'KL':>6}{'gain':>7}  (latent 1024)")
for r in sorted(table, key=lambda r: (not r["removed"], -r["gain_1024"])):
    print(f"{r['category']:<11}{r['defect']:<22}{'yes' if r['removed'] else '':>8}{r['log2_ratio']:>7.2f} | "
          f"{r['recon_all']:>5.2f}{r['kl_all']:>6.2f}{r['gain_all']:>+7.2f}                | "
          f"{r['recon_1024']:>5.2f}{r['kl_1024']:>6.2f}{r['gain_1024']:>+7.2f}")

print("\nKL wins (gain > .05):")
for col in ["gain_all", "gain_1024"]:
    rem = [r for r in table if r["removed"]]
    rest = [r for r in table if not r["removed"]]
    print(f"  {col:<10} removed: {sum(r[col] > .05 for r in rem)} of {len(rem)} | "
          f"rest: {sum(r[col] > .05 for r in rest)} of {len(rest)}")
for kind in ["texture", "object"]:
    k = [r for r in table if r["kind"] == kind]
    print(f"  mean VAE kl_max on {kind}s: all latents {np.mean([r['kl_all'] for r in k]):.2f}, "
          f"latent 1024 {np.mean([r['kl_1024'] for r in k]):.2f} | AE recon: {np.mean([r['recon_all'] for r in k]):.2f}, "
          f"{np.mean([r['recon_1024'] for r in k]):.2f}")
print(f"\nsaved {out}")
