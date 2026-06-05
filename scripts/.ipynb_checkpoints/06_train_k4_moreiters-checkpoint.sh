#!/bin/bash -e
#SBATCH --job-name=k4_1000iters
#SBATCH --output=/nesi/project/uoa04396/isin038/logs/k4_1000iters_%j.out
#SBATCH --error=/nesi/project/uoa04396/isin038/logs/k4_1000iters_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --gpus-per-node=A100:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=milan,genoa
#SBATCH --exclude=mg15,mg16

module purge
module load Python/3.11.6-foss-2023a CUDA/12.1.1

export nnUNet_raw="/nesi/project/uoa04396/isin038/results/part2/nnUNet_raw"
export nnUNet_preprocessed="/nesi/project/uoa04396/isin038/results/part2/nnUNet_preprocessed"
export nnUNet_results="/nesi/project/uoa04396/isin038/results/part2/nnUNet_results"

FOLD=${SLURM_ARRAY_TASK_ID}

echo "Training Dataset003_PialK4, fold ${FOLD}, 1000 iters/epoch"
echo "Start: $(date)"

nnUNetv2_train 003 3d_fullres ${FOLD} -tr nnUNetTrainer_1000iters

echo "End: $(date)"
