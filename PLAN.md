# PLAN.md — reconstruction-anomaly-paradox (CS599 Final Project)

*Updated Oct 23, 2026.*
- *The Fashion-MNIST/MNIST experiments are dropped (approved by Prof. Gong). The "simpler anomalies" question is tested on ECG instead, using measured and synthetic anomaly complexity.*
- *Prof. Gong suggested aiming for a conference submission. This version adds the review fixes (normalization, distance-matched anomalies, σ_x, diffusion width), a second dataset (MIT-BIH) and RQ4 (choosing capacity without labels).*

## Goal

This is a controlled empirical study of **when and why reconstruction-based anomaly detection fails** in generative models. It asks whether what makes a model *generative* (a KL prior, a noise schedule, a likelihood score) protects detection as capacity grows.

Models: **PCA** (linear baseline), **VAE**, **diffusion model**. All are trained on **normal data only**.

Deliverables:
1. **Course blog,** mid-December 2026.
2. **Paper** for a conference or workshop, about 4–6 weeks after the blog. The venue still needs choosing; check the real deadlines.

## Research questions

- **RQ1 – Capacity:** Does detection degrade as capacity increases, in a VAE (latent dim, β) and in a diffusion model (denoiser width)? How does the safe capacity range compare with PCA's?
- **RQ2 – Scores:** Do reconstruction error, KL (or the per-step denoising terms) and likelihood (ELBO) agree on which samples are anomalous? When they disagree, which term of the score explains it?
- **RQ3 – Why it fails:** Is failure driven by **capacity**, or by anomalies being **simpler** than normal data?
  - A **capacity failure** depends on capacity.
  - A **simplicity failure** means simple anomalies score AUROC **< 0.5** compared with complex anomalies *at the same distance from normal*, at *every* capacity.
- **RQ4 – Label-free selection:** Without labeled anomalies, can we choose a capacity that detects well? Test whether AUROC on synthetic anomalies built from validation normals tracks real AUROC, compared against naive label-free criteria.

## Data

### ECG5000 (UCR), primary dataset
- 5000 beats of 140 points each. Class 1 = normal, classes 2–5 = anomalous. Counts: {1: 2919, 2: 1767, 3: 96, 4: 194, 5: 24}.
- **Already z-normalized per beat by UCR.** Checked: every beat has mean 0 and std 0.996. Amplitude and variance information can't be recovered. We keep this preprocessing and say so in the paper.
  - So every synthetic anomaly built from ECG5000 is **also z-normalized per beat**, which stops models from detecting it through scale alone.
  - Variance is constant here, so it's not used as a complexity measure for ECG5000.
- Split: normal beats into 70% train / 15% val / 15% test (fixed seed). All anomalies go to test only.

### MIT-BIH Arrhythmia (PhysioNet `mitdb`), second dataset
- 48 half-hour two-lead recordings with beat-level annotations. Load with `wfdb`.
- Beats: a window around each annotated R-peak (−0.25 s to +0.45 s), resampled to 140 points with `data.resample()`. Use lead MLII.
- Labels: N = normal. Anomaly classes: V (PVC), A (atrial premature), L/R (bundle branch blocks), and others with enough samples.
- **Split by patient, not by beat** (DS1/DS2 split of de Chazal et al., 2004). Normal beats from the DS1 patients are split into train and val; the test set is the DS2 patients' beats.
- **Global normalization:** one mean and std computed from the normal training beats. Amplitude is meaningful here, so variance **is** used as a complexity measure.
- Purpose: tests whether the findings generalize, on raw-amplitude data.

## Complexity measures (`src/complexity.py`)
Computed for every test beat, real or synthetic:
- `zlib`: quantize to 8 bits, then `len(zlib.compress(...))` (Serrà et al., 2020)
- `spectral`: fraction of signal energy above a cutoff (e.g. above FFT bin 10)
- `tv`: total variation Σ|x[i+1] − x[i]|
- `var`: variance (MIT-BIH only)

A complexity claim is reported only where the measures agree.

## Experiments

| # | Normal | Anomalies | Purpose |
|---|---|---|---|
| **E1** | class 1 / N | real anomaly classes | Capacity sweeps for RQ1 and RQ2. Core result. |
| **E2** | class 1 / N | real classes, ranked by measured complexity | RQ3 on real data: are the *simpler* real anomalies the ones that get missed? |
| **E3** | class 1 / N | **synthetic**, distance-matched (below) | RQ3, controlled |
| **E4** | class 1 / N | synthetic anomalies made from **validation** normals | RQ4: label-free capacity selection |

All experiments run on ECG5000, and then on MIT-BIH once its loader exists.

### E2 – Complexity of real anomalies
- Show the complexity distribution for each class next to normal beats.
- Compute per-class AUROC at each capacity, and check whether the simpler classes are the ones that fail (AUROC < 0.5).

### E3 – Synthetic anomalies, matched on distance (`src/anomalies.py`)
Made from **held-out normal test beats only**, never training beats. Each anomaly is z-normalized per beat after it's generated (ECG5000).

| Type | Direction | How |
|---|---|---|
| Smoothed | simpler | Gaussian low-pass filter, width σ |
| Flatline | simpler | contiguous segment replaced by its mean, segment fraction f |
| Peak-clipped | simpler | main peak clipped to a percentile p |
| Noise | more complex | additive Gaussian noise, scale s |
| Spikes | more complex | a few narrow pulses of amplitude h |

- **Distance matching:** for each type, pick its dial value per beat so that `d = ‖x_anom − x_orig‖` (after normalization) hits target levels d ∈ {2, 4, 6, 8}. Find the dial value by bisection on a monotone parameter. Every type then covers the same distances.
- Damping is **dropped**, because per-beat normalization turns it into a no-op.
- **Key plot:** AUROC against distance d, one line per anomaly type, one panel per capacity (separately for each score).
  - Simple and complex lines overlap → only distance matters.
  - Simple lines sit lower, below 0.5, at every capacity → **simplicity failure**.
  - The gap opens only at high capacity → **capacity failure** that particularly affects simple anomalies.

### E4 – Label-free capacity selection (RQ4)
At every capacity setting, compute:
- **real AUROC:** test normals vs. real anomalies (for evaluation only)
- **proxy AUROC:** validation normals vs. E3-style synthetic anomalies generated from *other* validation normals (no labels needed)

Evaluate how well each selection criterion picks a setting with high real AUROC:
- Spearman correlation between the criterion and real AUROC across settings
- **regret:** real AUROC(best setting) − real AUROC(setting picked by the criterion)

Criteria to compare: proxy AUROC (ours); best validation reconstruction error (the naive choice); train/val gap; active latent units (VAE); random.

Target headline: *"picking the best reconstructor picks a bad detector; synthetic-anomaly AUROC recovers most of the best."*

## Models and sweeps

### PCA (baseline)
- k ∈ {1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120}
- Score: reconstruction MSE.
- Result on E1 (70/15/15 split): AUROC ≈ 0.97–0.99 at k ≤ 20, falling to 0.55 at k = 80 and 0.21 at k = 120. Next: run E2–E4.
- Resolution sweep (done, appendix): with k fixed, AUROC is flat from L = 20 to 560. It falls only when k grows with L, so resolution acts as a stand-in for capacity.

### VAE (1D)
- Encoder and decoder: small MLP (1D-conv if reconstructions are poor). Standard N(0, I) prior.
- **Decoder variance:** a single learned scalar `log σ_x`, trained jointly.
  - Loss: `SSE/(2σ_x²) + D·log σ_x + β·KL`, where D = input length.
  - Robustness check: fixed σ_x, with σ_x² set to the validation MSE after training.
- **Sweep latent dim:** {2, 4, 8, 16, 32, 64, 128} at β = 1
- **Sweep β:** {0.1, 0.5, 1, 2, 4} at one mid latent dim. Run the full grid if time allows.
- Early stopping on validation loss (normal data only). Log active units.

**Scores per test sample** (kept per sample; never averaged over the batch):
- `recon`: MSE between x and decode(μ)
- `kl`: 0.5 · Σ(μ² + σ² − log σ² − 1)
- `neg_elbo`: Gaussian NLL with the learned σ_x, averaged over K = 10 samples of z, plus KL. **Always with β = 1**, whatever β was used in training.

### Diffusion model (1D)
- Small 1D U-Net or MLP denoiser with timestep embedding, ε-prediction, DDPM schedule (T ≤ 1000), simplified loss ‖ε − ε̂‖².
- **Capacity sweep (required): denoiser width.** Base channels (or hidden size) ∈ {8, 16, 32, 64, 128}.
- **Test-time axis: noise level t**, about 8 values across [0, T]. Report a width × t grid. When comparing with the PCA and VAE capacity curves, use one t chosen on validation data.

**Scores per test sample:**
- `recon_t`: noise x to level t, denoise back to x̂₀, then MSE(x, x̂₀), averaged over several noise draws
- `denoise_terms`: the vector of per-timestep weighted noise-prediction errors (the diffusion counterpart of the VAE's KL)
- `neg_elbo`: the full variational bound, summing weighted terms over all t. Use a Gaussian decoder at t = 0; the prior term ≈ 0.

## Evaluation
- **AUROC** for every score × setting × experiment × dataset, with **5 seeds** (mean ± std).
- **Bootstrap 95% CIs** (1000 resamples of the test set) on every AUROC, especially per-class AUROC (ECG5000 class 5 has only 24 beats).
- **Paired bootstrap** for "score A beats score B" claims.
- **Spearman correlation** between scores (RQ2).
- **Disagreement analysis:** for samples ranked very differently by recon and ELBO, find which term (reconstruction vs. KL, or which timesteps) drives the gap.
- **Sanity checks:**
  - held-out normal error ≈ training error
  - normal-vs-normal AUROC (val vs. test) ≈ 0.5
  - shuffled labels → AUROC ≈ 0.5

## Figures
1. AUROC vs. capacity: PCA, VAE and diffusion on one axis, one panel per score (both datasets).
2. Normal vs. anomaly error vs. capacity, for each model.
3. Reconstruction examples at low vs. high capacity, plus diffusion noise-to-norm at low vs. high t.
4. Complexity distributions: normal vs. each real anomaly class (E2).
5. **AUROC vs. distance d, one line per synthetic type, one panel per capacity (E3), the key RQ3 plot.**
6. Recon score vs. ELBO score scatter, colored by normal/anomaly (RQ2).
7. **Proxy vs. real AUROC across settings, and regret per selection criterion (E4).**

## Logging / outputs
- Per-sample scores: `results/metrics/{dataset}_{exp}_{model}_{setting}_seed{s}.csv`, with columns `sample_id, label, class, anomaly_type, distance, complexity_zlib, complexity_spectral, complexity_tv, <score columns>`.
- Summary table: `results/metrics/summary.csv`.
- Figures: `results/figures/`. Checkpoints: `results/runs/` (gitignored).

## Repo structure
```
data/raw/                 ECG5000_*.txt, mitdb/
data/processed/
src/data.py               ECG5000 loading, splits, resample
src/mitbih.py             beat extraction, patient split, global normalization
src/anomalies.py          synthetic generators + distance matching
src/complexity.py         zlib, spectral, tv, var
src/pca_model.py
src/vae.py
src/diffusion.py
src/train.py              shared training loop
src/evaluate.py           scores, AUROC, bootstrap CIs, Spearman, per-class AUROC, selection regret
experiments/run_sweep.py  --model {pca,vae,diffusion} --dataset {ecg5000,mitbih} --exp {E1,E2,E3,E4}
results/figures/, results/metrics/
blog/draft.md
```

## Status
- [x] ECG5000 download + 70/15/15 split: 2043 train / 437 val / 439 normal + 2081 anomalous test
- [x] Shared evaluation + paradox-curve plot
- [x] PCA capacity sweep on ECG5000: AUROC 0.99 (k=2) → 0.21 (k=120)
- [x] PCA resolution sweep on ECG5000
- [x] `.venv` rebuilt on python.org 3.13 arm64 (torch 2.15, MPS available); versions pinned in `requirements.txt`
- [x] MNIST code removed (archived at git tag `mnist-pca-archive`)
- [x] `src/vae.py` written (MLP VAE, fixed σ_x; **update to learned σ_x**), not run yet
- [ ] `src/complexity.py`, `src/anomalies.py`; PCA on E2–E4
- [ ] VAE training script + sweeps
- [ ] Everything else below

## Timeline
| Dates | Work |
|---|---|
| Oct 23–26 | Rebuild venv; 70/15/15 split; rerun PCA; `complexity.py` + `anomalies.py` (distance matching); PCA on E2–E4 |
| Oct 27–Nov 7 | VAE (learned σ_x) + training script; E1 sweeps with 5 seeds; VAE on E2–E4 |
| Nov 1–10 | MIT-BIH loader + patient split (in parallel with the VAE sweeps); PCA + VAE on MIT-BIH |
| Nov 8–21 | Diffusion: implementation, width × t sweeps, E1–E4 on both datasets |
| Nov 22–30 | Buffer (Thanksgiving); disagreement analysis (RQ2); bootstrap CIs; final figures |
| Dec 1–mid | **Course blog** |
| mid-Dec–Jan | **Paper:** related work, write-up, extra seeds and ablations; submit to the chosen venue |

Fallback if behind: MIT-BIH and RQ4 move to the paper phase, and the blog covers ECG5000 E1–E3 only.

## Related work
- Bauer, Nakajima & Müller (2022, arXiv 2206.11723): capacity limits (bottleneck, sparsity) don't stop autoencoders from reconstructing anomalies; self-supervised denoising fix
- Yong & Brintrup (2022, arXiv 2202.12637), "Do autoencoders need a bottleneck for anomaly detection?": non-bottlenecked AEs can do *better*, a counterpoint
- Gong et al. (ICCV 2019), MemAE: autoencoders over-generalize to anomalies
- Ding, Zhao & Akoglu (NeurIPS 2022): hyperparameter sensitivity in unsupervised outlier detection (motivates RQ4)
- Nalisnick et al. (ICLR 2019): higher likelihood for simpler out-of-distribution data
- Serrà et al. (ICLR 2020): input complexity and out-of-distribution detection
- Zhang et al. (2023), DiffusionAD; Wyatt et al. (CVPRW 2022), AnoDDPM; Graham et al. (CVPRW 2023): diffusion reconstruction across noise levels; Goodier & Campbell (BMVC 2023): diffusion likelihood for out-of-distribution detection
- de Chazal et al. (2004): patient-wise MIT-BIH split
