"""Paper tables: image AUROC with bootstrap 95% CIs, AP and F1-max, per category and defect type.

Step 1 (cached): per-image scores of every saved model on the test set, written to
results/mvtec/scores/<category>_<size>.npz. Models: AE and VAE at latent 1024
(mean and max scores, VAE kl_max) and the diffusion model at each noise level
(mean and max of the denoising error, and mean and max of the noise-prediction error map,
plus the approximate bound; see src/common/diffusion.py), each for every seed that has a
saved model.pt. Delete the .npz to recompute after adding a score.

Step 2: for each method (model + score) and each defect type ("all" = every defect
vs. normal), averaged over seeds:

    auroc, auroc_lo, auroc_hi : image AUROC and its bootstrap 95% CI
    ap                        : average precision (area under precision-recall)
    f1_max                    : best F1 over all thresholds. The threshold is picked on
                                the test set, so it is optimistic; it is the
                                convention of recent MVTec papers (UniAD, DiAD)
    vs_ae, vs_ae_lo, vs_ae_hi : AUROC minus the AE's, with a 95% CI. The AE score
                                (mean or max) is whichever is better on that defect
                                type, which favours the AE; CI above 0 = a clear win

Bootstrap: normal and defective test images are resampled separately (n_boot times,
with replacement). The same resamples are used for every method and seed, so
differences between methods are paired.

    python -m src.mvtec.paper_tables --category tile --device mps
    python -m src.mvtec.paper_tables --combine tile cable     # LaTeX tables from the CSVs
"""
import argparse

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

from src.common import paths
from src.common.diffusion import noise_error_set, reconstruct_set
from src.common.evaluate import write_csv
from src.common.train import load_run
from src.mvtec import data
from src.mvtec.diffusion_experiments import NOISE_LEVELS, load_run as load_diffusion, run_name
from src.mvtec.scoring import map_scores, model_scores, reconstruction_scores

SCORES_DIR = paths.RESULTS_DIR / "mvtec" / "scores"
SEEDS = [0, 1, 2]


def compute_scores(category, size, device):
    """{"<method>|seed<k>": (n_test,) scores} for every saved model, plus the defect labels."""
    splits = data.load_splits(category, size)
    out = {}
    for beta, name in [(0, "AE"), (1, "VAE")]:
        for seed in SEEDS:
            run = f"mvtec_{category}_{size}_b{beta}_z1024_seed{seed}"
            if (paths.RUNS_DIR / run / "model.pt").exists():
                scores, _ = model_scores(load_run(run)[0], splits["X_test"], size)
                for s in ["mean", "max"] + (["kl_max"] if beta else []):
                    out[f"{name} {s}|seed{seed}"] = scores[s]
    for seed in SEEDS:
        if (paths.RUNS_DIR / run_name(category, size, seed) / "model.pt").exists():
            model, _ = load_diffusion(run_name(category, size, seed))
            for t in NOISE_LEVELS:
                X_hat = reconstruct_set(model, splits["X_test"], t, device)
                scores, _ = reconstruction_scores(splits["X_test"], X_hat, size)
                for s in ["mean", "max"]:
                    out[f"diffusion t={t} {s}|seed{seed}"] = scores[s]
            for t, raw in noise_error_set(model, splits["X_test"], NOISE_LEVELS, device).items():
                scores, _ = map_scores(raw, size)
                for s in ["mean", "max"]:
                    out[f"noise error {t if t == 'bound' else f't={t}'} {s}|seed{seed}"] = scores[s]
    return out, splits["defect"]


def load_scores(category, size, device):
    path = SCORES_DIR / f"{category}_{size}.npz"
    if not path.exists():
        scores, defect = compute_scores(category, size, device)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, defect=defect, **scores)
    f = np.load(path)
    return {k: f[k] for k in f.files if k != "defect"}, f["defect"]


def auroc(neg, pos):
    """AUROC from scores of normal (..., n_neg) and defective (..., n_pos) images; ties count half."""
    diff = pos[..., :, None] - neg[..., None, :]
    return (diff > 0).mean(axis=(-2, -1)) + 0.5 * (diff == 0).mean(axis=(-2, -1))


def f1_max(labels, scores):
    precision, recall, _ = precision_recall_curve(labels, scores)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(f1.max())


def table(category, size, device, n_boot=1000, seed=0):
    scores, defect = load_scores(category, size, device)
    methods = list(dict.fromkeys(k.split("|")[0] for k in scores))
    seeds_of = {m: [k for k in scores if k.split("|")[0] == m] for m in methods}
    normal = np.flatnonzero(defect == "good")
    rng = np.random.default_rng(seed)
    rows = []
    for t in ["all"] + [d for d in dict.fromkeys(defect) if d != "good"]:
        bad = np.flatnonzero(defect != "good") if t == "all" else np.flatnonzero(defect == t)
        neg_idx = rng.choice(normal, (n_boot, len(normal)))  # one set of resamples per defect type,
        pos_idx = rng.choice(bad, (n_boot, len(bad)))        # shared by every method and seed
        labels = np.r_[np.zeros(len(normal)), np.ones(len(bad))]
        point, boot = {}, {}
        for m in methods:
            ks = seeds_of[m]
            point[m] = np.mean([auroc(scores[k][normal], scores[k][bad]) for k in ks])
            boot[m] = np.mean([auroc(scores[k][neg_idx], scores[k][pos_idx]) for k in ks], axis=0)
        ae = max(["AE mean", "AE max"], key=lambda m: point[m]) if "AE mean" in point else None
        for m in methods:
            ks = seeds_of[m]
            both = [np.r_[scores[k][normal], scores[k][bad]] for k in ks]
            row = {"category": category, "defect": t, "n_defect": len(bad), "method": m, "seeds": len(ks),
                   "auroc": point[m], "auroc_lo": np.percentile(boot[m], 2.5),
                   "auroc_hi": np.percentile(boot[m], 97.5),
                   "ap": np.mean([average_precision_score(labels, s) for s in both]),
                   "f1_max": np.mean([f1_max(labels, s) for s in both])}
            if ae:
                d = boot[m] - boot[ae]
                row.update({"ae_ref": ae, "vs_ae": point[m] - point[ae], "vs_ae_lo": np.percentile(d, 2.5),
                            "vs_ae_hi": np.percentile(d, 97.5)})
            rows.append(row)
    return rows


def main(category, size, device, n_boot):
    rows = table(category, size, device, n_boot)
    write_csv(rows, paths.RESULTS_DIR / "mvtec" / "data" / f"paper_table_{category}.csv")
    for t in dict.fromkeys(r["defect"] for r in rows):
        sub = [r for r in rows if r["defect"] == t]
        print(f"\n{category} / {t} ({sub[0]['n_defect']} defect images); vs AE = vs {sub[0].get('ae_ref')}")
        print(f"{'method':22}{'AUROC [95% CI]':>22}{'AP':>7}{'F1max':>7}{'vs AE [95% CI]':>26}")
        for r in sub:
            vs = f"{r['vs_ae']:+.2f} [{r['vs_ae_lo']:+.2f}, {r['vs_ae_hi']:+.2f}]" if "vs_ae" in r else ""
            mark = " *" if "vs_ae" in r and r["vs_ae_lo"] > 0 else ""
            label = f"{r['method']} ({r['seeds']})"
            ci = f"{r['auroc']:.2f} [{r['auroc_lo']:.2f}, {r['auroc_hi']:.2f}]"
            print(f"{label:22}{ci:>22}{r['ap']:>7.2f}{r['f1_max']:>7.2f}{vs:>26}{mark}")


# (header, methods (the better one is shown per row), localisation map for the hit rate)
COLUMNS = [("AE", ["AE mean", "AE max"], "AE error"),
           ("VAE KL", ["VAE kl_max"], "VAE KL"),
           ("Diffusion\\\\recon. $t{=}50$", ["diffusion t=50 mean", "diffusion t=50 max"], "diffusion t=50"),
           ("Diffusion\\\\bound", ["noise error bound mean", "noise error bound max"], "noise error bound")]
MAIN_ROWS = [("tile", "all"), ("tile", "gray_stroke"), ("tile", "glue_strip"),
             ("cable", "all"), ("cable", "missing_cable"), ("cable", "missing_wire")]


def _hit_rates(category):
    """{(defect, map): (hit rate, chance)}, each the mean over seeds, from localization_<category>.csv."""
    import csv
    rows = list(csv.DictReader(open(paths.RESULTS_DIR / "mvtec" / "data" / f"localization_{category}.csv")))
    out = {}
    for key in {(r["defect"], r["map"]) for r in rows}:
        sub = [r for r in rows if (r["defect"], r["map"]) == key]
        out[key] = (np.mean([float(r["hit_rate"]) for r in sub]), np.mean([float(r["chance"]) for r in sub]))
    return out


def combined_table(categories):
    """Three LaTeX tables, written to results/mvtec/paper_tables.tex:

    main     : the MAIN_ROWS. (a) image AUROC [95% CI] per method plus KL - AE [95% CI];
               (b) peak-cell hit rate per method (localization.py, mean of 3 seeds) and chance
    appendix : (a) for every defect type, and AP / F1-max for every defect type
    Methods with a mean and a max score show the better one on each row, which favours
    them over the KL score. A dagger marks the detail-removed types (defect_types.csv)."""
    import csv
    removed = {(r["category"], r["defect"]) for r in csv.DictReader(open(
        paths.RESULTS_DIR / "mvtec" / "data" / "defect_types.csv")) if r["removed"] == "True"}
    auroc_rows, app_rows, main_rows, hit_rows = {}, [], [], {}
    for category in categories:
        rows = list(csv.DictReader(open(paths.RESULTS_DIR / "mvtec" / "data" / f"paper_table_{category}.csv")))
        hits = _hit_rates(category)
        for t in dict.fromkeys(r["defect"] for r in rows):
            by = {r["method"]: r for r in rows if r["defect"] == t}
            picked = [max((by[m] for m in ms), key=lambda r: float(r["auroc"])) for _, ms, _ in COLUMNS]
            best = max(float(r["auroc"]) for r in picked)
            name = "all defects" if t == "all" else t.replace("_", " ") + ("$^\\dagger$" if (category, t) in removed else "")
            label = f"{category} & {name} ({by['VAE kl_max']['n_defect']})"
            cells = [(f"\\textbf{{{float(r['auroc']):.2f}}}" if float(r["auroc"]) == best else f"{float(r['auroc']):.2f}")
                     + f" {{\\scriptsize[{float(r['auroc_lo']):.2f}, {float(r['auroc_hi']):.2f}]}}" for r in picked]
            kl = by["VAE kl_max"]
            diff = f"{float(kl['vs_ae']):+.2f} {{\\scriptsize[{float(kl['vs_ae_lo']):+.2f}, {float(kl['vs_ae_hi']):+.2f}]}}"
            if float(kl["vs_ae_lo"]) > 0:
                diff = f"\\textbf{{{diff}}}"
            auroc_rows[(category, t)] = " & ".join([label] + cells + [diff]) + " \\\\"
            app_rows.append(" & ".join([label] + [f"{float(r['ap']):.2f} / {float(r['f1_max']):.2f}" for r in picked])
                            + " \\\\")
            if (category, t) in MAIN_ROWS and t != "all":
                h = [hits[(t, m)][0] for _, _, m in COLUMNS]
                hit_rows[(category, t)] = (" & ".join([label] + [f"\\textbf{{{v:.2f}}}" if v == max(h) else f"{v:.2f}" for v in h]
                                           + [f"{hits[(t, COLUMNS[0][2])][1]:.2f}"]) + " \\\\")
        app_rows.append("\\midrule")
    for key in MAIN_ROWS:
        if key[0] in categories:
            if main_rows and key[1] == "all":
                main_rows.append("\\midrule")
            main_rows.append(auroc_rows[key])
    full_rows = []
    for category in categories:
        full_rows += [v for (c, _), v in auroc_rows.items() if c == category] + ["\\midrule"]
    heads = " & ".join(["Category & Defect type ($n$)"] + [f"\\makecell{{{c}}}" for c, _, _ in COLUMNS])
    cols = "ll" + "c" * (len(COLUMNS) + 1)
    ci_note = ("Image AUROC with bootstrap 95\\% CI (1000 resamples, 3 seeds averaged), MVTec AD at "
               "$128\\times128$, latent 1024 for AE and VAE. AE and diffusion show the better of their mean "
               "and max scores on each row, which favours them. KL $-$ AE: paired bootstrap CI; bold = CI above 0. "
               "$^\\dagger$ = defect removes fine detail. Bold AUROC = best in row.")
    tex = ("% Made by: python -m src.mvtec.paper_tables --combine " + " ".join(categories) + "\n"
           "% Needs \\usepackage{booktabs,makecell}\n"
           "\\begin{table}[t]\\centering\\small\n"
           "\\caption{Detail-removing defects. (a) " + ci_note + " (b) Share of defect images whose highest-scoring "
           "$8\\times8$ cell of the anomaly map contains defect pixels (mean of 3 seeds); chance = share of defect "
           "cells. Bold = best in row.}\n\\label{tab:main}\n"
           "\\begin{tabular}{" + cols + "}\\toprule\n"
           + heads + " & KL $-$ AE \\\\\\midrule\n"
           "\\multicolumn{" + str(len(COLUMNS) + 3) + "}{l}{\\textit{(a) Image AUROC [95\\% CI]}} \\\\\n"
           + "\n".join(main_rows) + "\n\\midrule\n"
           "\\multicolumn{" + str(len(COLUMNS) + 2) + "}{l}{\\textit{(b) Peak-cell hit rate}} & \\textit{chance} \\\\\n"
           + "\n".join(hit_rows[k] for k in MAIN_ROWS if k in hit_rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n\n"
           "\\begin{table}[t]\\centering\\small\n"
           "\\caption{All defect types. " + ci_note + "}\n\\label{tab:all-types}\n"
           "\\begin{tabular}{" + cols + "}\\toprule\n"
           + heads + " & KL $-$ AE \\\\\\midrule\n" + "\n".join(full_rows[:-1]) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n\n"
           "\\begin{table}[t]\\centering\\small\n"
           "\\caption{AP / F1-max for the same methods and rows. F1-max picks its threshold on the test "
           "set, so it is optimistic; it is reported for comparison with prior MVTec work.}\n\\label{tab:ap-f1}\n"
           "\\begin{tabular}{" + cols[:-1] + "}\\toprule\n"
           + heads + " \\\\\\midrule\n" + "\n".join(app_rows[:-1]) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    out = paths.RESULTS_DIR / "mvtec" / "paper_tables.tex"
    out.write_text(tex)
    print(f"saved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--category", default="tile")
    parser.add_argument("--size", type=int, default=data.SIZE)
    parser.add_argument("--device", default="cpu", help="cpu, cuda or mps (diffusion scores only)")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--combine", nargs="+", metavar="CATEGORY",
                        help="write the LaTeX tables from these categories' CSVs instead")
    args = parser.parse_args()
    if args.combine:
        combined_table(args.combine)
    else:
        main(args.category, args.size, args.device, args.n_boot)
