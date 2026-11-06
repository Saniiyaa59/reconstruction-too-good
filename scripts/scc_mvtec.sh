#!/bin/bash -l
# BU SCC array job: conv AE (beta 0) and VAE (beta 1) on MVTec AD at 128x128, one task per category.
#
#     qsub scripts/scc_mvtec.sh                 # all 15 categories in parallel
#     qsub -t 4 scripts/scc_mvtec.sh            # only task 4 (carpet)
#
# Needs data/processed/mvtec_<category>_128.npz (the cached splits with masks); the raw
# images are not needed. Runs that are already saved are loaded and rescored, not
# retrained, so a task that hits the time limit can just be submitted again. Keep the
# full seed list in SEEDS when adding seeds: each task rewrites its category's CSV
# from the runs it scores.
#
#$ -N mvtec_vae
#$ -P cs599bg
#$ -t 1-15
#$ -l h_rt=8:00:00
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

CATEGORIES=(bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper)
CATEGORY=${CATEGORIES[$((SGE_TASK_ID - 1))]}
SEEDS="0"   # first pass; then "0 1 2"

# Stop right away if the job can't see a GPU, instead of failing partway through
python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"
echo "task $SGE_TASK_ID: $CATEGORY, seeds $SEEDS"

python -m src.mvtec.vae_experiments run --categories "$CATEGORY" --seeds $SEEDS --device cuda
