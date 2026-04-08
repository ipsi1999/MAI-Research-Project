#!/bin/bash
#SBATCH --job-name=mc_evaluation
#SBATCH --account=uoa04396
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/home/isin038/00_nesi_projects/uoa04396/isin038/results/part1/mc_eval_%j.log
#SBATCH --error=/home/isin038/00_nesi_projects/uoa04396/isin038/results/part1/mc_eval_%j.err

# Load Python module
module load Python/3.11.3-gimkl-2022a

python -m pip install nibabel trimesh scikit-image rtree --user --quiet

# Install required packages if not already installed
python -c "import nibabel, trimesh, skimage; print('All packages available')"

# Create output directory
mkdir -p /home/isin038/00_nesi_projects/uoa04396/isin038/results/part1

# Run the evaluation script
python -u /home/isin038/00_nesi_projects/uoa04396/isin038/scripts/run_mc_evaluation.py

echo "Job complete"
