#!/bin/bash
#SBATCH --job-name=stat_test
#SBATCH --account=uoa04396
#SBATCH --time=00:30:00             # 30 mins is plenty
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G                    # 8GB is more than enough for pandas merges
#SBATCH --output=/home/isin038/00_nesi_projects/uoa04396/isin038/results/part2/stats_%j.log

# Load the same Python version you used before
module load Python/3.11.3-gimkl-2022a

# Ensure scipy and pandas are available (standard in gimkl modules, but --user installs are safe)
python -m pip install pandas scipy numpy --user --quiet

# Run the stats script

python -u /home/isin038/00_nesi_projects/uoa04396/isin038/scripts/scripts/statistical_tests.py

echo "Stats Job complete"