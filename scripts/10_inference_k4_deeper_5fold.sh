#!/bin/bash -e
#SBATCH --job-name=k4_inf_deeper
#SBATCH --output=/nesi/project/uoa04396/isin038/logs/k4_inf_deeper_%j.out
#SBATCH --error=/nesi/project/uoa04396/isin038/logs/k4_inf_deeper_%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=H100:1
#SBATCH --partition=genoa
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=isin038@aucklanduni.ac.nz

module purge
module load Python/3.11.6-foss-2023a CUDA/12.1.1

export nnUNet_raw="/nesi/project/uoa04396/isin038/results/part2/nnUNet_raw"
export nnUNet_preprocessed="/nesi/project/uoa04396/isin038/results/part2/nnUNet_preprocessed"
export nnUNet_results="/nesi/project/uoa04396/isin038/results/part2/nnUNet_results"

INPUT_DIR="${nnUNet_raw}/Dataset003_PialK4/imagesTs"
OUTPUT_DIR="/nesi/project/uoa04396/isin038/results/part2/predictions/k4_deeper"

mkdir -p ${OUTPUT_DIR}

echo "Inference: nnUNetTrainer_DeeperEncoder | K4 | 5-fold ensemble"
echo "Output: ${OUTPUT_DIR}"
echo "Start: $(date)"

nnUNetv2_predict \
    -i ${INPUT_DIR} \
    -o ${OUTPUT_DIR} \
    -d 003 \
    -c 3d_fullres \
    -tr nnUNetTrainer_DeeperEncoder \
    -f 0 1 2 3 4

echo "End: $(date)"
