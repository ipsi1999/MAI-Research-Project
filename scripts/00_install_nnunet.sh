#!/bin/bash
#SBATCH --job-name=install_nnunet
#SBATCH --account=uoa04396
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=/nesi/project/uoa04396/isin038/logs/install_nnunet_%j.out
#SBATCH --error=/nesi/project/uoa04396/isin038/logs/install_nnunet_%j.err

echo "Started: $(date)"

module purge
module load Python/3.11.6-foss-2023a
module load CUDA/12.1.1

python -m pip install --user --upgrade pip setuptools wheel
python -m pip install --user nnunetv2
python -m pip install --user nibabel scipy scikit-image trimesh numpy pandas matplotlib tqdm
python -m pip install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

python -u - <<'PYEOF'
checks = ["torch","nnunetv2","nibabel","scipy","skimage","trimesh","numpy"]
for mod in checks:
    try:
        m = __import__(mod)
        print(f"  OK  {mod:20} {getattr(m,'__version__','?')}")
    except ImportError as e:
        print(f"  FAIL {mod:20} {e}")

import torch
print(f"\n  CUDA available: {torch.cuda.is_available()}")
PYEOF

echo "Done: $(date)"
