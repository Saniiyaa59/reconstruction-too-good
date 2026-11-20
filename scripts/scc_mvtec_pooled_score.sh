#!/bin/bash -l
# BU SCC: step 2 of the pooled MVTec runs. Loads the 6 pooled models trained by
# scc_mvtec_pooled.sh (AE and VAE, latent 1024, seeds 0-2), scores each on every category's
# test set and writes results/mvtec/vae_pooled/<category>_128.csv.
#
#     qsub -hold_jid mvtec_pooled scripts/scc_mvtec_pooled_score.sh   # waits for the training tasks
#     qsub scripts/scc_mvtec_pooled_score.sh                          # or once they have finished
#
#$ -N mvtec_pooled_score
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
CATEGORIES="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"

python -m src.mvtec.vae_experiments run-pooled --categories $CATEGORIES --device cuda
