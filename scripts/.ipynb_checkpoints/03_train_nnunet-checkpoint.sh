#!/bin/bash
#SBATCH --job-name=nnunet_train
#SBATCH --account=uoa04396
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=genoa,milan
#SBATCH --gpus-per-node=a100:1
#SBATCH --output=/nesi/project/uoa04396/isin038/logs/nnunet_train_d%a_f%x_%j.out
#SBATCH --error=/nesi/project/uoa04396/isin038/logs/nnunet_train_d%a_f%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=isin038@aucklanduni.ac.nz
#SBATCH --array=1-2
#SBATCH --exclude=mg15
set -e

echo "======================================================"
echo "Job ID     : $SLURM_JOB_ID"
echo "Dataset ID : $SLURM_ARRAY_TASK_ID"
echo "Fold       : $FOLD"
echo "Started    : $(date)"
echo "======================================================"

module purge
module load Python/3.11.6-foss-2023a
module load CUDA/12.1.1

export nnUNet_raw=/nesi/project/uoa04396/isin038/results/part2/nnUNet_raw
export nnUNet_preprocessed=/nesi/project/uoa04396/isin038/results/part2/nnUNet_preprocessed
export nnUNet_results=/nesi/project/uoa04396/isin038/results/part2/nnUNet_results

echo ">>> Training Dataset${SLURM_ARRAY_TASK_ID}, 3d_fullres, fold ${FOLD} ..."

nnUNetv2_train ${SLURM_ARRAY_TASK_ID} 3d_fullres ${FOLD} 

echo "Done: $(date)"