# Fast Pial Cortical Surface Reconstruction from MRI

**Master of Artificial Intelligence — 30-point Research Project**  
Auckland Bioengineering Institute, University of Auckland  
*Supervisors: Dr. Jiantao Shen · Dr. Hamid Abbasi*

---

## Overview

This project develops a fast, automated pipeline for reconstructing pial cortical surfaces from T1-weighted MRI, targeting **sub-minute runtime** as a practical replacement for traditional neuroimaging tools such as FreeSurfer (which typically take ~1 hour per subject).

**Pipeline:** High-resolution MRI → nnU-Net v2 segmentation → Marching Cubes → Pial mesh

Performance is benchmarked against the current state-of-the-art, **RFENet** (Wang et al., 2026, IEEE TMRB; ASSD = 0.290 mm, HD90 = 0.628 ± 0.116 mm), using the IXI dataset with FastSurfer-generated pial meshes as pseudo-label ground truth.

---

## Repository Structure

```
MAI-Research-Project/
├── data/                        # Data paths and preprocessing scripts
├── scripts/
│   ├── preprocessing/           # MRI preprocessing utilities
│   ├── training/                # nnU-Net training configuration and custom trainers
│   ├── inference/               # SLURM inference scripts (NeSI HPC)
│   └── evaluation/
│       └── run_mc_evaluation_part2.py   # ASSD / HD90 evaluation (point-to-face)
├── nnunet_trainers/
│   └── variants/training_length/
│       └── nnUNetTrainer_ReceptiveField.py   # Custom trainers (DeeperEncoder, DilatedConv, 1000iters)
├── architectures/
│   ├── ruvsur.py                # RUVSUR: Swin Transformer + U-Net decoder + VAE bottleneck
│   ├── ruvsur_v2.py             # RUVSUR v2 with optional mirror CNN blocks
│   ├── unetr_block.py           # UNETR transformer block utilities
│   └── main.py                  # RUVSUR training entry point
├── results/                     # Evaluation outputs and comparison tables
└── README.md
```

---

## Dataset

- **IXI Dataset** — 185 subjects; split 157 train / 28 test (seed = 42)
- **Ground truth:** FastSurfer-generated pial meshes (pseudo-labels)
- **External validation (planned):** OASIS-1, ADNI

---

## Methods

### nnU-Net v2 — Multi-Resolution Experiments

Three voxel resolutions are explored, each auto-configured with identical patch sizes (160×128×112) and batch size 2:

| Config | Voxel Size | ASSD (mm) | HD90 (mm) | Notes |
|--------|-----------|-----------|-----------|-------|
| k2 | 0.469 mm | 0.1786 | 0.6122 | Baseline |
| k3 | 0.313 mm | **0.1695** | **0.5929** | Best overall |
| k4 default | 0.234 mm | 0.2024† | 0.6405 | Underperforms k3 |
| k4_1000iters | 0.234 mm | 0.1758 | 0.6176 | Extended training |
| k4_deeper | 0.234 mm | 0.1738 | 0.6064 | Best k4 variant |
| k4_dilated | 0.234 mm | 0.2091 | 0.7459 | Not recommended |

†Including outlier subject IXI072-HH-2324 (excluded from other comparisons due to data quality).  
All variants achieve ~39–41% improvement over RFENet on ASSD. All pairwise differences are statistically significant (Wilcoxon signed-rank test).

### Custom nnU-Net Trainers

Located in `nnunet_trainers/variants/training_length/nnUNetTrainer_ReceptiveField.py`:

- **nnUNetTrainer_1000iters** — Extends iterations per epoch for high-resolution configs
- **nnUNetTrainer_DeeperEncoder** — Additional encoder depth for k4 resolution
- **nnUNetTrainer_DilatedConv** — Dilated convolutions (dilation=2); not recommended for surface reconstruction

### RUVSUR Architecture (Parallel Direction)

A hybrid deep learning architecture combining:
- Swin Transformer encoder
- U-Net decoder
- Optional VAE bottleneck
- Optional mirror CNN blocks (RUVSUR v2)

Benchmarked against nnU-Net using ASSD/HD90 on IXI, with 3-fold cross-validation.

---

## Evaluation

Evaluation uses **point-to-face distances** (via `trimesh`) to compute ASSD and HD90, consistent with RFENet's reported methodology. Point-to-point distances (e.g. via cKDTree) are **not** used, as they are not directly comparable to the SOTA benchmark.

**Coordinate correction:** Marching Cubes vertices are converted from scanner RAS to tkReg RAS by subtracting the `cras` vector before comparison against FreeSurfer ground truth.

```bash
# Run evaluation for a given resolution config
python scripts/evaluation/run_mc_evaluation_part2.py --resolution k3
# Valid options: k2, k3, k4, k4_1000iters, k4_deeper, k4_dilated
```

---

## HPC (NeSI Mahuika)

- **Project account:** `uoa04396`
- **Modules:** `Python/3.11.6-foss-2023a`, `CUDA/12.1.1`
- **Key paths:**
  - Data: `/nesi/project/uoa04396/isin038/data/IXI/`
  - Scripts: `/nesi/project/uoa04396/isin038/scripts/scripts/`
  - Logs: `/nesi/project/uoa04396/isin038/logs/`

> **Note:** Inference SLURM jobs must **not** include `--gpus-per-node` (inference is CPU-scheduled). GPU type should be specified explicitly (e.g. `H100:1` or `A100:1`).

---

## Requirements

```
nnU-Net v2
trimesh
MONAI (install from source: pip install -e ".[all]")
PyTorch
nibabel
numpy
scipy
```

---

## Citation / Comparison Baseline

Wang et al. (2026). *RFENet: Rapid and Fine-grained Edge-aware Network for Cortical Surface Reconstruction*. IEEE Transactions on Medical Robotics and Bionics (TMRB).  
ASSD = 0.290 mm · HD90 = 0.628 ± 0.116 mm

---

## Status

- [x] nnU-Net v2 multi-resolution training (k2, k3, k4 variants)
- [x] Marching Cubes mesh extraction + coordinate correction
- [x] Point-to-face ASSD/HD90 evaluation pipeline
- [x] Statistical significance testing (Wilcoxon signed-rank)
- [ ] k4_deeper 5-fold ensemble evaluation
- [ ] External dataset validation (OASIS-1, ADNI)
- [ ] RUVSUR / RUVSUR v2 benchmarking
- [ ] Thesis submission (deadline: 29 June 2026)

---

*University of Auckland · Auckland Bioengineering Institute · 2026*
