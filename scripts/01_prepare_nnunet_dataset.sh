#!/bin/bash
#SBATCH --job-name=nnunet_prep
#SBATCH --account=uoa04396
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/nesi/project/uoa04396/isin038/logs/nnunet_prep_%j.out
#SBATCH --error=/nesi/project/uoa04396/isin038/logs/nnunet_prep_%j.err

echo "Started: $(date)"

module purge
module load Python/3.11.6-foss-2023a

IXI_ROOT=/nesi/project/uoa04396/isin038/data/IXI
OUT_ROOT=/nesi/project/uoa04396/isin038/results/part2/nnUNet_raw
SCRIPTS=/nesi/project/uoa04396/isin038/scripts/scripts

mkdir -p ${OUT_ROOT}

# echo ">>> Preparing k2 dataset ..."
# python -u ${SCRIPTS}/01_prepare_nnunet_dataset.py \
#     --ixi_root    ${IXI_ROOT} \
#     --out_root    ${OUT_ROOT} \
#     --resolution  k2 \
#     --test_fraction 0.15 \
#     --seed 42

# echo ">>> Preparing k3 dataset ..."
# python -u ${SCRIPTS}/01_prepare_nnunet_dataset.py \
#     --ixi_root    ${IXI_ROOT} \
#     --out_root    ${OUT_ROOT} \
#     --resolution  k3 \
#     --test_fraction 0.15 \
#     --seed 42

echo ">>> Preparing k4 dataset ..."
python -u ${SCRIPTS}/01_prepare_nnunet_dataset.py \
    --ixi_root    ${IXI_ROOT} \
    --out_root    ${OUT_ROOT} \
    --resolution  k4 \
    --test_fraction 0.15 \
    --seed 42

echo "Done: $(date)"