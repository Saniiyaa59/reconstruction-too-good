"""Is the defect area the part that is reconstructed badly? Per defect type, per model.

Image AUROC only asks whether a defect image scores higher than a normal one, not
why. This checks, on each defective test image, whether its anomaly map is high
*inside* the ground-truth mask:

    ratio    : mean map value inside the mask / mean outside it. Above 1 = the
               defect area is worse than the rest of the image
    peak hit : is the map's largest 8x8 cell (each cell = mean of its block, as in
               cell AUROC) one that contains defect pixels? This is the spot the
               max / kl_max scores read
    chance   : the hit rate a randomly placed peak would get = the share of cells
               that contain defect pixels, averaged over the images

Maps compared: the smoothed error map of the AE, the VAE and the diffusion model
(each noise level), the VAE's KL map, and the diffusion model's smoothed
noise-prediction error map (each noise level, and the approximate bound). Reported per defect type: median ratio
over the images, and hit rate.

    python -m src.mvtec.localization --category tile --device mps
"""
import argparse

import numpy as np

from src.common import paths
from src.common.diffusion import noise_error_set, reconstruct_set
from src.common.evaluate import write_csv
from src.common.train import load_run
from src.mvtec import data
from src.mvtec.diffusion_experiments import NOISE_LEVELS, load_run as load_diffusion, run_name
from src.mvtec.scoring import map_scores, model_scores, reconstruction_scores, to_cells


def localization(maps, masks):
    """(ratio, hit, chance) per image, for defective images only (masks with any pixel set)."""
    masks = masks.astype(bool)
    cells_map, cells_mask = to_cells(maps), to_cells(masks.astype(float)) > 0
    ratio, hit, chance = [], [], []
    for m, mask, cm, cmask in zip(maps, masks, cells_map, cells_mask):
        ratio.append(m[mask].mean() / m[~mask].mean())
        hit.append(cmask.flat[cm.argmax()])
        chance.append(cmask.mean())
    return np.array(ratio), np.array(hit), np.array(chance)


def model_maps(category, size, seeds, device):
    """{label: (n, size, size) map} for every model with a saved run."""
    splits = data.load_splits(category, size)
    out = {}
    for beta, name in [(0, "AE"), (1, "VAE")]:
        for seed in seeds:
            run = f"mvtec_{category}_{size}_b{beta}_z1024_seed{seed}"
            if not (paths.RUNS_DIR / run / "model.pt").exists():
                continue
            model, _ = load_run(run)
            _, maps = model_scores(model, splits["X_test"], size)
            out[(f"{name} error", seed)] = maps["max"]
            if beta:
                out[("VAE KL", seed)] = maps["kl_max"]
    for seed in seeds:
        if not (paths.RUNS_DIR / run_name(category, size, seed) / "model.pt").exists():
            continue
        model, _ = load_diffusion(run_name(category, size, seed))
        for t in NOISE_LEVELS:  # whole test set, so the noise matches the CSV scores
            X_hat = reconstruct_set(model, splits["X_test"], t, device)
            out[(f"diffusion t={t}", seed)] = reconstruction_scores(splits["X_test"], X_hat, size)[1]["max"]
        for t, raw in noise_error_set(model, splits["X_test"], NOISE_LEVELS, device).items():
            out[(f"noise error {t if t == 'bound' else f't={t}'}", seed)] = map_scores(raw, size)[1]["max"]
    return splits, out


def main(category, size, seeds, device):
    splits, maps = model_maps(category, size, seeds, device)
    types = [t for t in dict.fromkeys(splits["defect"]) if t != "good"]
    rows = []
    for (label, seed), m in maps.items():
        for t in types:
            idx = splits["defect"] == t
            ratio, hit, chance = localization(m[idx], splits["masks"][idx])
            rows.append({"category": category, "map": label, "seed": seed, "defect": t, "n": int(idx.sum()),
                         "median_ratio": float(np.median(ratio)), "hit_rate": float(hit.mean()),
                         "chance": float(chance.mean())})
    write_csv(rows, paths.RESULTS_DIR / "mvtec" / "data" / f"localization_{category}.csv")

    labels = list(dict.fromkeys(label for label, _ in maps))
    for key, title in [("median_ratio", "median inside/outside ratio (>1 = defect area worse)"),
                       ("hit_rate", "peak hit rate (chance in last row)")]:
        print(f"\n{category}: {title}, averaged over seeds")
        print(f"{'map (seeds)':22}" + " ".join(f"{t[:11]:>11}" for t in types))
        for label in labels:
            vals = [np.mean([r[key] for r in rows if r["map"] == label and r["defect"] == t]) for t in types]
            n_seeds = len({r["seed"] for r in rows if r["map"] == label})
            print(f"{f'{label} ({n_seeds})':22}" + " ".join(f"{v:>11.2f}" for v in vals))
        if key == "hit_rate":
            print(f"{'chance':22}" + " ".join(
                f"{np.mean([r['chance'] for r in rows if r['defect'] == t]):>11.2f}" for t in types))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--category", default="tile")
    parser.add_argument("--size", type=int, default=data.SIZE)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--device", default="cpu", help="cpu, cuda or mps (diffusion only)")
    args = parser.parse_args()
    main(args.category, args.size, args.seeds, args.device)
