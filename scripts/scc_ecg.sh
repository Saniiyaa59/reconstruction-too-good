#!/bin/bash -l
# BU SCC job: the matched-error test and the conv VAE at beta 1 and 0, on one GPU.
#
# In an OnDemand code-server session with the course environment active:
#     nohup bash scripts/scc_ecg.sh > scc_ecg.log 2>&1 &
# As a batch job:
#     qsub scripts/scc_ecg.sh
#
#$ -N ecg_vae
#$ -P cs599bg
#$ -l h_rt=5:00:00
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

# Download ECG5000 if needed and build the splits
python src/data.py

# The training figures compare against these PCA results
python src/pca_model.py
python src/pca_model.py --exp E3

# The MLPs train faster on the CPU; only the conv VAE gains from the GPU
python src/train.py --matched-error
python src/train.py --device cuda --beta-sweep --channels 32 --betas 1 0 --seeds 0
python src/train.py --e3 vae_ecg5000_sweep_conv32_b1_seed0 vae_ecg5000_sweep_conv32_b0_seed0 \
    --e3-figure vae_ecg_E3_conv32

