#!/bin/bash -l
# BU SCC: one diffusion model on MVTec (one category, one seed), trained on normal images
# and scored at noise levels 50/100/250/500. Not an array job: category and seed are the
# script's arguments (default: tile 0).
#
#     qsub scripts/scc_mvtec_diffusion.sh              # tile, seed 0
#     qsub scripts/scc_mvtec_diffusion.sh cable 1      # cable, seed 1
#
# Writes results/runs/mvtec_<category>_128_diffusion_seed<seed>/ and
# results/mvtec/diffusion/<category>_128_seed<seed>.csv. A saved model is loaded and
# rescored, not retrained. Needs data/processed/mvtec_<category>_128.npz.
#
#$ -N mvtec_diffusion
#$ -P cs599bg
#$ -l h_rt=2:00:00
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -pe omp 4
#$ -j y
#$ -cwd

set -e
if [ "$CONDA_DEFAULT_ENV" != "fall-2026-pyt" ]; then
    module load miniconda/23.1.0 academic-ml/fall-2026
    conda activate fall-2026-pyt
fi
export OMP_NUM_THREADS=${NSLOTS:-4}
CATEGORY=${1:-tile}
SEED=${2:-0}

python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"
echo "diffusion: $CATEGORY, seed $SEED"

python -m src.mvtec.diffusion_experiments run --categories "$CATEGORY" --seeds "$SEED" --device cuda
