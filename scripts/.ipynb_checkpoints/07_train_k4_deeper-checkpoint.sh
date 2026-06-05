#!/bin/bash -e
#SBATCH --job-name=k4_deeper
#SBATCH --output=/nesi/project/uoa04396/isin038/logs/k4_deeper_%A_%a.out
#SBATCH --error=/nesi/project/uoa04396/isin038/logs/k4_deeper_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --gpus-per-node=H100:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=genoa
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=isin038@aucklanduni.ac.nz

module purge
module load Python/3.11.6-foss-2023a CUDA/12.1.1

export nnUNet_raw="/nesi/project/uoa04396/isin038/results/part2/nnUNet_raw"
export nnUNet_preprocessed="/nesi/project/uoa04396/isin038/results/part2/nnUNet_preprocessed"
export nnUNet_results="/nesi/project/uoa04396/isin038/results/part2/nnUNet_results"

FOLD=${SLURM_ARRAY_TASK_ID}
echo "Training Dataset003_PialK4 | Deeper Encoder | Fold ${FOLD}"
echo "Start: $(date)"
nnUNetv2_train 003 3d_fullres ${FOLD} -tr nnUNetTrainer_DeeperEncoder
echo "End: $(date)"
