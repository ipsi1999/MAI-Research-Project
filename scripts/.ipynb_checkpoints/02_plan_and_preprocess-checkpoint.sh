#!/bin/bash
#SBATCH --job-name=nnunet_plan
#SBATCH --account=uoa04396
#SBATCH --time=12:00:00
#SBATCH --mem=196G
#SBATCH --cpus-per-task=8
#SBATCH --output=/nesi/project/uoa04396/isin038/logs/nnunet_plan_%j.out
#SBATCH --error=/nesi/project/uoa04396/isin038/logs/nnunet_plan_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=isin038@aucklanduni.ac.nz
set -e

echo "Started: $(date)"

module purge
module load Python/3.11.6-foss-2023a

export nnUNet_raw=/nesi/project/uoa04396/isin038/results/part2/nnUNet_raw
export nnUNet_preprocessed=/nesi/project/uoa04396/isin038/results/part2/nnUNet_preprocessed
export nnUNet_results=/nesi/project/uoa04396/isin038/results/part2/nnUNet_results

mkdir -p ${nnUNet_preprocessed} ${nnUNet_results}

echo ">>> Running plan and preprocess for k4 (Dataset003)..." ###k2 (Dataset001), k3 (Dataset002)

##previously for both k2 and k3 -d 1 2 3
nnUNetv2_plan_and_preprocess -d 3 \
    -c 3d_fullres \
    --verify_dataset_integrity 

echo "Done: $(date)"