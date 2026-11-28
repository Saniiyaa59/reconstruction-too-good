# When Reconstruction Gets Too Good

**When and why does reconstruction-based anomaly detection fail?**

A model trained only on normal images is supposed to reconstruct defects badly,
so a high reconstruction error flags a defect. But a model that reconstructs
well can also reproduce the defect, and then the defect leaves no error. This
project studies when that happens on MVTec AD, comparing an autoencoder (AE),
a VAE and a diffusion model, and asks which defects each score misses and why.

## Main findings

All numbers are image AUROC on MVTec AD at 128×128, averaged over 3 seeds,
with AE and VAE at latent size 1024 unless stated otherwise.

1. **Better reconstruction does not mean worse detection overall.** The
   diffusion model reconstructs normal images better than the AE in all 15
   categories, and it is also the best detector: .92 on average vs. .81 for the
   AE and .55 for the VAE's KL score. It beats or ties the AE in every category.

   | | AE recon | VAE KL | Diffusion (t = 50) |
   |---|---|---|---|
   | Textures (5) | .80 | .66 | **.89** |
   | Objects (10) | .82 | .49 | **.94** |
   | All 15 | .81 | .55 | **.92** |

2. **The failure is specific to defects that remove detail.** We measure
   whether a defect has less fine detail than what should be there (log2
   ratio, see below). This flags 11 of the 73 defect types. On 4 of them, the
   AE and the diffusion model copy the defect and only the VAE's KL score
   catches it (bootstrap 95% CI of KL − AE above 0):

   | Defect type | AE recon | Diffusion recon | Diffusion bound | VAE KL |
   |---|---|---|---|---|
   | tile gray_stroke | .61 | .61 | .69 | **1.00** |
   | tile glue_strip | .54 | .69 | **.93** | .83 |
   | cable missing_cable | .68 | .49 | .46 | **.97** |
   | cable missing_wire | .40 | .57 | .56 | **.76** |

   Diffusion starts from the image plus a little noise and repairs what looks
   abnormal. A crack gets repaired, which leaves error. A smooth stroke or an
   empty cable slot already looks like plain normal surface, so it is kept and
   leaves no error.
   ([denoising: tile](results/mvtec/diffusion/defect_tile_gray_stroke_denoise.png),
   [cable](results/mvtec/diffusion/defect_cable_missing_cable_denoise.png))

3. **The KL score points at the defect.** The highest-scoring cell of the KL
   map lands on the gray stroke in 100% of images and on the missing cable in
   94% (chance .12 and .22). The AE's error peak lands there in 33% and 36%.
   ([error vs. KL maps](results/mvtec/vae/defect_tile_gray_stroke_maps.png))

4. **Capacity makes the AE copy the missing parts.** On cable, as the AE's
   latent grows and its reconstructions improve, its AUROC on missing_cable
   falls from .98 to .58 and on missing_wire from .61 to .38. The KL beats the
   AE on all 4 types from latent 256 up.
   ([capacity figure](results/mvtec/vae/capacity_detail_removed.png))

5. **One model for all categories erases the KL wins.** A pooled AE/VAE
   trained on all 15 categories lowers reconstruction AUROC (.81 → .75) and
   KL on gray_stroke falls from 1.00 to .38 (missing_cable .97 → .53).

### Limits

- Removing detail is necessary but not sufficient: every clear KL win is a
  detail-removing type, but 7 of the 11 flagged types are not KL wins (tile
  rough and oil lean towards KL but their CIs include 0).
- All 4 wins are in tile and cable, so that part of the story rests on 4 types
  in 2 categories. Per-type CIs are wide (10–18 images per type).
- The detail measure differs for textures and objects, and the "removed" rule
  was set after looking at the data.
- Small models at 128×128; results may change at full resolution.

## Setup

**Data.** MVTec AD: 15 industrial categories (5 textures, 10 objects), 73
defect types with pixel-level masks. Images resized to 128×128 RGB. Normal
training images are split 85/15 (train / val); the official test set (normal +
defective) is used for evaluation.

**Models.**

| Model | Details |
|---|---|
| AE / VAE | ConvVAE2D with an 8×8 latent grid; β = 0 is the AE, β = 1 the VAE. Latent sizes 64–16384, 3 seeds, one model per category (450 models) |
| Pooled AE / VAE | one model on all 15 categories, latent 1024, 3 seeds |
| Diffusion | small DDPM U-Net per category, 3 seeds (45 models); EMA weights, early stopping on validation loss |

**Scores.**

| Score | What it is |
|---|---|
| `mean` / `max` | mean of the reconstruction error map, or max of the map after Gaussian smoothing (σ = 4 px) |
| `kl_max` | largest per-cell KL of the VAE's 8×8 latent grid |
| Diffusion recon | noise the image to step t ∈ {50, 100, 250, 500}, denoise with deterministic DDIM, then `mean` / `max` of the error |
| Diffusion bound | noise-prediction error summed over 20 steps with the ELBO weights (the diffusion counterpart of the KL term) |

For the AE and diffusion, tables show the better of `mean` and `max` on each
row, which favours them over the KL.

**Detail measure.** Detail map = |gray − blurred gray|. Textures: detail
inside the defect mask vs. a ring of surface around it. Objects: detail at the
defect spot vs. the same spot in normal images. A type counts as "detail
removed" if its median log2 ratio is below 0 (textures) or below the 5th
percentile of a null built from normal images (objects).

**Evaluation.** Image AUROC overall and per defect type; bootstrap 95% CIs
(1000 resamples, normal and defective images resampled separately, paired
across methods); AP and F1-max; peak-cell hit rate on the 8×8 grid for
localisation. Details in [results/README.md](results/README.md).

## Status

- [x] AE and VAE on all 15 categories (5 latent sizes, 3 seeds), error and KL maps
- [x] Detail measure for all 73 defect types, against KL vs. reconstruction
- [x] Pooled training on all categories
- [x] Diffusion on all 15 categories (3 seeds)
- [x] Localisation check and bootstrap CIs (tile, cable)
- [x] Capacity figure and main LaTeX table
- [ ] Scale figure: diffusion noise error vs. noise level
- [ ] Bootstrap CIs for the other 7 detail-removing types
- [ ] Blog post, then paper

## Repo structure

```
.
├── src/                    # run as modules from the repo root: python -m src.mvtec.data
│   ├── common/
│   │   ├── paths.py        # data and results locations
│   │   ├── evaluate.py     # AUROC and summaries
│   │   ├── vae.py          # conv VAE (β = 0 gives the AE)
│   │   ├── diffusion.py    # small DDPM U-Net; DDIM reconstruction; noise error and bound
│   │   ├── train.py        # train one VAE; save and load runs
│   │   └── plots.py        # shared figure styling
│   ├── mvtec/
│   │   ├── data.py         # MVTec AD loader + 85/15 train/val split
│   │   ├── scoring.py      # error and KL maps; image, pixel and cell AUROC
│   │   ├── pca.py          # PCA capacity sweep (baseline)
│   │   ├── vae_experiments.py            # AE and VAE, per category and pooled: train and score
│   │   ├── defect_maps.py                # one defect type: error vs. KL maps
│   │   ├── defect_complexity.py          # textures: defect detail vs. surrounding ring
│   │   ├── defect_complexity_aligned.py  # objects: defect detail vs. same spot in normal images
│   │   ├── textures_scatter.py           # all texture types: detail vs. KL gain
│   │   ├── defect_table.py               # all 73 types: detail removed?, KL gain
│   │   ├── diffusion_experiments.py      # diffusion: train and score at noise levels 50-500, denoising picture
│   │   ├── localization.py               # is the map high inside the defect mask?
│   │   ├── paper_tables.py               # bootstrap CIs, AP, F1-max; LaTeX tables
│   │   └── capacity_figure.py            # AE vs. KL AUROC across latent sizes
│   └── ecg/                # earlier ECG5000 pilot, not part of the current write-up
├── scripts/
│   ├── scc_mvtec.sh               # SCC GPU array job: AE + VAE, one task per category
│   ├── scc_mvtec_pooled.sh        # SCC GPU array job: pooled AE + VAE, train only
│   ├── scc_mvtec_pooled_score.sh  # SCC GPU job: score the pooled models per category
│   └── scc_mvtec_diffusion.sh     # SCC GPU job: one diffusion model (category, seed as arguments)
├── results/                # see results/README.md for what each file shows
│   ├── mvtec/{data,pca,vae,vae_pooled,diffusion}/
│   ├── mvtec/paper_tables.tex     # LaTeX tables for the paper
│   └── runs/               # saved models, one folder per run (gitignored)
└── data/                   # downloaded and cached splits (gitignored)
```

## Installation

Requires Python 3.13. On Apple Silicon, use a native arm64 Python (e.g. from
python.org) so pip installs a torch build with MPS support.

```bash
git clone https://github.com/Saniiyaa59/reconstruction-too-good.git
cd reconstruction-too-good
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Usage

```bash
python -m src.mvtec.data                               # build splits for the categories in data/mvtec/
python -m src.mvtec.vae_experiments run --device mps   # AE + VAE per category: train (or load) and score
python -m src.mvtec.vae_experiments run-pooled --device mps   # one AE + VAE on all categories
python -m src.mvtec.diffusion_experiments run --categories tile --device mps   # diffusion: train (or load) and score
python -m src.mvtec.diffusion_experiments maps --category tile --defect gray_stroke   # denoising picture
python -m src.mvtec.defect_maps                        # tile/gray_stroke error vs. KL maps
python -m src.mvtec.defect_complexity --category tile          # defect detail per type (textures)
python -m src.mvtec.defect_complexity_aligned --category cable # defect detail per type (objects)
python -m src.mvtec.defect_table                       # 73-type table (CSV)
python -m src.mvtec.localization --category tile --device mps  # localisation per defect type
python -m src.mvtec.paper_tables --category tile --device mps  # AUROC with bootstrap CIs, AP, F1-max
python -m src.mvtec.paper_tables --combine tile cable          # LaTeX tables for the paper
python -m src.mvtec.capacity_figure                    # capacity figure
```

On BU's SCC, diffusion runs one job per category and seed:
`qsub scripts/scc_mvtec_diffusion.sh cable 1`.

## Background

Final project for CS599 "Deep Visual Generative Models" at Boston University.
The course deliverable is a blog post; the study will then be extended into a
paper.

## References

- Bergmann et al., "MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection" (CVPR 2019)
- Nalisnick et al., "Do Deep Generative Models Know What They Don't Know?" (ICLR 2019)
- Bauer, Nakajima & Müller, "Self-Supervised Training with Autoencoders for Visual Anomaly Detection" (arXiv:2206.11723)
- Yong & Brintrup, "Do Autoencoders Need a Bottleneck for Anomaly Detection?" (arXiv:2202.12637)
- Zhang et al., "DiffusionAD: Norm-guided One-step Denoising Diffusion for Anomaly Detection" (arXiv:2303.08730)
- Serrà et al., "Input Complexity and Out-of-Distribution Detection with Likelihood-based Generative Models" (ICLR 2020)

## License

MIT
