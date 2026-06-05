#!/bin/bash -e
#SBATCH --job-name=mc_eval_k4v
#SBATCH --output=/nesi/project/uoa04396/isin038/logs/mc_eval_%j.out
#SBATCH --error=/nesi/project/uoa04396/isin038/logs/mc_eval_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --partition=milan,genoa
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=isin038@aucklanduni.ac.nz

module purge
module load Python/3.11.6-foss-2023a

SCRIPTS=/nesi/project/uoa04396/isin038/scripts/scripts
RES=$1

echo "MC Evaluation: ${RES}"
echo "Start: $(date)"

python3 -m pip install --user --quiet nibabel trimesh scikit-image numpy scipy pandas
python3 -u ${SCRIPTS}/run_mc_evaluation_part2.py --resolution ${RES}

echo "End: $(date)"
