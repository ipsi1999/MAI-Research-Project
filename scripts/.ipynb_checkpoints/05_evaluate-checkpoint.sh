#!/bin/bash
#SBATCH --job-name=mc_eval_part2
#SBATCH --account=uoa04396
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --output=/nesi/project/uoa04396/isin038/logs/mc_eval_part2_%a_%j.out
#SBATCH --error=/nesi/project/uoa04396/isin038/logs/mc_eval_part2_%a_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=isin038@aucklanduni.ac.nz
#SBATCH --array=1-3


set -e

case $SLURM_ARRAY_TASK_ID in
  1) RES="k2" ;;
  2) RES="k3" ;;
  3) RES="k4" ;;
esac

SCRIPTS=/nesi/project/uoa04396/isin038/scripts/scripts

echo "======================================================"
echo "Job ID     : $SLURM_JOB_ID"
echo "Resolution : $RES"
echo "Started    : $(date)"
echo "======================================================"

module purge
echo ">>> Running MC evaluation for ${RES} ..."
module load Python/3.11.6-foss-2023a

python -m pip install --user --quiet nibabel trimesh scikit-image numpy scipy pandas

python -u ${SCRIPTS}/run_mc_evaluation_part2.py --resolution ${RES}

echo "Done: $(date)"
