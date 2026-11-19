# Results

Grouped by dataset, then model: `results/<dataset>/<model>/`. Each figure sits
next to the CSV it was made from (CSVs are gitignored; rerun the command to get
them). Saved models are in `runs/` (gitignored).

**Reading the names:** `E1`/`E2`/`E3` are the ECG experiments in the main
README. `b1`/`b0` = β = 1 (VAE) / β = 0 (plain autoencoder). `mlp`/`conv32` =
the network (conv32 = 1D conv VAE with base width 32). `_64`/`_128` = MVTec
image size in pixels.

## ECG5000

| File | What it shows | Made by |
|---|---|---|
| `ecg/data/synthetic_examples.png` | One normal beat and each synthetic anomaly type at d = 2, 4, 6, 8 | `python -m src.ecg.anomalies` |
| `ecg/data/complexity_by_class.png` | Complexity (zlib, spectral, total variation) of test beats per class | `python -m src.ecg.pca --exp E2` |
| `ecg/pca/E1_capacity_sweep` | PCA: AUROC and normal/anomaly error vs. k on the real anomalies; AUROC collapses below chance at high k | `python -m src.ecg.pca` |
| `ecg/pca/E2_complexity` | PCA: AUROC per complexity quartile of the real anomalies vs. k; the simplest quartile is missed first | `python -m src.ecg.pca --exp E2` |
| `ecg/pca/E3_synthetic` | PCA: AUROC per synthetic type vs. distance, at five k; smoothed and speedup are missed at high k | `python -m src.ecg.pca --exp E3` |
| `ecg/vae/E1_latent_sweep_b1_seed0` | Small MLP VAE (β = 1): AUROC vs. latent size per score, and vs. normal error next to PCA | `python -m src.ecg.vae_experiments e1` |
| `ecg/vae/beta_sweep_mlp` | Wide MLP VAE, β from 1 to 0, 3 seeds: AUROC vs. normal error next to the PCA curve | `python -m src.ecg.vae_experiments beta-sweep` |
| `ecg/vae/beta_sweep_conv32` | The same with the 1D conv VAE; β = 0 reaches PCA's collapse range without collapsing | `... beta-sweep --channels 32` |
| `ecg/vae/E2_complexity` | VAE runs (points) vs. PCA (lines) per complexity quartile, against normal error | `... e2 RUN ... --figure E2_complexity` |
| `ecg/vae/E3_types.png` | AUROC per synthetic type vs. distance: PCA k = 50 next to each VAE run | `... e3 RUN ... --figure E3_types` |
| `ecg/vae/matched_error_{mlp,conv32}.csv` | KL on vs. off at equal validation error (no figure) | `... matched-error [--channels 32]` |
| `ecg/vae/scores/` | Per-run CSVs: E1 per-beat scores, E3 AUROC per type and distance | the commands above |

## MVTec AD

| File | What it shows | Made by |
|---|---|---|
| `mvtec/data/examples_<category>_<size>.png` | One normal test image and one per defect type, at the size the models see | `python -m src.mvtec.data [--size 64]` |
| `mvtec/data/defect_complexity_<texture>.png` | Textures: fine detail inside each defect vs. a ring of nearby texture (log2 ratio < 0 = defect smoother), per image, with each type's KL gain | `python -m src.mvtec.defect_complexity --category tile` |
| `mvtec/data/defect_complexity_<object>.png` | Objects: detail inside each defect vs. the same spot in normal training images (< 0 = detail removed), with the normal-image range shaded | `python -m src.mvtec.defect_complexity_aligned --category cable` |
| `mvtec/data/defect_complexity_textures.png` | All 25 texture types: median log2 ratio vs. KL gain (VAE kl_max − AE recon); prints the check over blur and ring settings | `python -m src.mvtec.textures_scatter` |
| `mvtec/data/defect_types.csv` | All 73 types: detail removed or not, AE recon, VAE kl_max and KL gain, averaged over all latents and at latent 1024 | `python -m src.mvtec.defect_table` |
| `mvtec/pca/sweep_<size>.png` | PCA: image AUROC vs. k per category; black = max score, gray = mean score, colors = defect types | `python -m src.mvtec.pca [--size 64]` |
| `mvtec/pca/<category>_<size>.csv` | Per k and score: normal MSE, image AUROC (overall, per defect type), pixel and cell AUROC | same |
| `mvtec/vae/sweep_<size>.png` | Image AUROC (max score) vs. normal MSE: PCA curve, AE and VAE per latent size, VAE kl_max | `python -m src.mvtec.vae_experiments plot` |
| `mvtec/vae/<category>_<size>.csv` | Per model (β, latent size, seed) and score: the same metrics as the PCA CSVs | `python -m src.mvtec.vae_experiments run` |
| `mvtec/vae/defect_<category>_<defect>_maps.png` | One VAE, one defect type: 2 normal and 4 defective test images as input, reconstruction, error map, KL map | `python -m src.mvtec.defect_maps [--category --defect --run]` |
| `mvtec/vae/defect_<category>_<defect>_scatter.png` | Recon vs. KL score per test image (normal vs. that defect), with 95th-percentile-of-normal thresholds | same |

MVTec scores (see `src/mvtec/scoring.py`): `mean` = mean squared error over
the image; `max` = max of the smoothed error map; `kl` / `kl_max` = total / max
KL over the latent grid (VAE only). Pixel and cell AUROC grade the maps against
the ground-truth defect masks.

The 64×64 MVTec results are the first look (bottle, carpet, tile; AE seed 0
only); the 128×128 results are the main ones.
