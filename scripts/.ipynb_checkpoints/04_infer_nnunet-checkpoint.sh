#!/bin/bash
#SBATCH --job-name=infer_nnunet
#SBATCH --account=uoa04396
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=milan,genoa
#SBATCH --gpus-per-node=a100:1
#SBATCH --output=/nesi/project/uoa04396/isin038/logs/nnunet_infer_%a_%j.out
#SBATCH --error=/nesi/project/uoa04396/isin038/logs/nnunet_infer_%a_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=isin038@aucklanduni.ac.nz
#SBATCH --array=1-2
#SBATCH --exclude=mg16
set -e

echo "======================================================"
echo "Job ID     : $SLURM_JOB_ID"
echo "Dataset ID : $SLURM_ARRAY_TASK_ID"
echo "Started    : $(date)"
echo "======================================================"

module purge
module load Python/3.11.6-foss-2023a
module load CUDA/12.1.1

export nnUNet_raw=/nesi/project/uoa04396/isin038/results/part2/nnUNet_raw
export nnUNet_preprocessed=/nesi/project/uoa04396/isin038/results/part2/nnUNet_preprocessed
export nnUNet_results=/nesi/project/uoa04396/isin038/results/part2/nnUNet_results

case $SLURM_ARRAY_TASK_ID in
  1) DATASET_NAME="Dataset001_PialK2" ; RES="k2" ;;
  2) DATASET_NAME="Dataset002_PialK3" ; RES="k3" ;;
esac

mkdir -p /nesi/project/uoa04396/isin038/results/part2/predictions/${RES}

nnUNetv2_predict \
    -i ${nnUNet_raw}/${DATASET_NAME}/imagesTs \
    -o /nesi/project/uoa04396/isin038/results/part2/predictions/${RES} \
    -d $SLURM_ARRAY_TASK_ID \
    -c 3d_fullres \
    -f 0 1 2 3 4 \
    --disable_progress_bar \
    --continue_prediction
    

echo "Done: $(date)"