#!/bin/bash -l
# BU SCC: conv AE and VAE trained on the normal images of ALL 15 MVTec categories together
# (latent 1024), then scored on each category's test set. Two steps:
#
#     qsub scripts/scc_mvtec_pooled.sh                                         # 1. train: 6 tasks, one model each
#     qsub -t 1 -v SCORE=1 -hold_jid mvtec_pooled scripts/scc_mvtec_pooled.sh  # 2. score all 6 once they finish
#
# Step 1 tasks only train and save (in parallel they would overwrite each other's CSVs).
# Step 2 loads the 6 saved models and writes results/mvtec/vae_pooled/<category>_128.csv.
# Needs data/processed/mvtec_<category>_128.npz for all 15 categories.
#
#$ -N mvtec_pooled
#$ -P cs599bg
#$ -t 1-6
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
CATEGORIES="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"

python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"

if [ "$SCORE" = "1" ]; then
    echo "scoring all pooled models"
    python -m src.mvtec.vae_experiments run-pooled --categories $CATEGORIES --device cuda
else
    BETAS=(0 0 0 1 1 1)
    SEEDS=(0 1 2 0 1 2)
    BETA=${BETAS[$((SGE_TASK_ID - 1))]}
    SEED=${SEEDS[$((SGE_TASK_ID - 1))]}
    echo "task $SGE_TASK_ID: beta $BETA, seed $SEED"
    python -m src.mvtec.vae_experiments run-pooled --categories $CATEGORIES --betas $BETA --seeds $SEED \
        --device cuda --train-only
fi
