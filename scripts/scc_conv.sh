#!/bin/bash -l
# BU SCC batch job: finish the conv VAE questions on ECG5000, on one GPU.
#
#     qsub scripts/scc_conv.sh
#
# Run scc_ecg.sh once first: this job needs the data and PCA results it creates.
#
#$ -N conv_vae
#$ -P cs599bg
#$ -l h_rt=3:00:00
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -pe omp 8
#$ -j y
#$ -cwd

set -e
if [ "$CONDA_DEFAULT_ENV" != "fall-2026-pyt" ]; then
    module load miniconda/23.1.0 academic-ml/fall-2026
    conda activate fall-2026-pyt
fi
export OMP_NUM_THREADS=${NSLOTS:-8}

# Stop right away if the job can't see a GPU, instead of failing partway through
python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"

# 1. Does the KL help in the conv model? KL on vs. off at equal error
python src/vae_experiments.py matched-error --device cuda --channels 32

# 2. How low can a conv VAE with the KL on get? (includes beta 1 and 0 with 3 seeds)
python src/vae_experiments.py beta-sweep --device cuda --channels 32

# 3. Can wider models with the KL off reach PCA k=80's error of 0.0017?
python src/vae_experiments.py beta-sweep --device cuda --channels 64 --betas 0
python src/vae_experiments.py beta-sweep --device cuda --channels 128 --betas 0

# 4. Synthetic anomalies: every sweep model gets a CSV, seed 0 also gets a figure
all_runs=()
for beta in 1 0.1 0.01 0.001 0; do
    for seed in 0 1 2; do
        all_runs+=("vae_ecg5000_sweep_conv32_b${beta}_seed${seed}")
    done
done
for channels in 64 128; do
    for seed in 0 1 2; do
        all_runs+=("vae_ecg5000_sweep_conv${channels}_b0_seed${seed}")
    done
done
python src/vae_experiments.py e3 "${all_runs[@]}"
python src/vae_experiments.py e3 vae_ecg5000_sweep_conv32_b{1,0.1,0.01,0.001,0}_seed0 \
    --figure vae_ecg_E3_conv32_betas
python src/vae_experiments.py e3 vae_ecg5000_sweep_conv32_b0_seed0 vae_ecg5000_sweep_conv64_b0_seed0 \
    vae_ecg5000_sweep_conv128_b0_seed0 --figure vae_ecg_E3_conv_widths

