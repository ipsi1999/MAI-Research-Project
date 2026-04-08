#!/bin/bash
#SBATCH --job-name=mc_test
#SBATCH --account=uoa04396
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/home/isin038/00_nesi_projects/uoa04396/isin038/results/part1/mc_test_%j.log
#SBATCH --error=/home/isin038/00_nesi_projects/uoa04396/isin038/results/part1/mc_test_%j.err

module load Python/3.11.3-gimkl-2022a
python -m pip install nibabel trimesh scikit-image rtree --user --quiet

python - << 'EOF'
import os, time, numpy as np, nibabel as nib, trimesh
from skimage import measure
from scipy.spatial import cKDTree

DATA_ROOT = "/home/isin038/00_nesi_projects/uoa04396/isin038/data/IXI"
subj = "IXI012-HH-1211"
subj_dir = os.path.join(DATA_ROOT, subj)
K_LEVELS = ["k2", "k3", "k4"]

def run_marching_cubes(nii_path):
    img = nib.load(nii_path)
    data = img.get_fdata()
    affine = img.affine
    verts, faces, normals, values = measure.marching_cubes(data, level=0.5)
    verts_physical = nib.affines.apply_affine(affine, verts)
    verts_physical[:, 0] *= -1
    verts_physical[:, 1] *= -1
    mesh = trimesh.Trimesh(vertices=verts_physical, faces=faces)
    mesh.fix_normals()
    return mesh

def load_freesurfer_surface(surf_path):
    vertices, faces, meta = nib.freesurfer.read_geometry(surf_path, read_metadata=True)
    vertices += meta['cras']
    vertices[:, 0] *= -1
    vertices[:, 1] *= -1
    return trimesh.Trimesh(vertices=vertices, faces=faces)

def compute_metrics(mesh_pred, mesh_gt):
    tree1 = cKDTree(mesh_gt.vertices)
    d1, _ = tree1.query(mesh_pred.vertices)
    tree2 = cKDTree(mesh_pred.vertices)
    d2, _ = tree2.query(mesh_gt.vertices)
    assd = (np.mean(d1) + np.mean(d2)) / 2
    hd95 = max(np.percentile(d1, 95), np.percentile(d2, 95))
    hd99 = max(np.percentile(d1, 99), np.percentile(d2, 99))
    return assd, hd95, hd99

print(f"Testing on: {subj}")

lh = load_freesurfer_surface(os.path.join(subj_dir, "surf", "lh.pial"))
rh = load_freesurfer_surface(os.path.join(subj_dir, "surf", "rh.pial"))
gt = trimesh.util.concatenate([lh, rh])
print(f"Ground truth loaded: {len(gt.vertices):,} vertices")

for k in K_LEVELS:
    pial_path = os.path.join(subj_dir, "proc", "upsampling", f"pial_resampled_{k}.nii.gz")
    print(f"\nRunning {k}...")
    t0 = time.time()
    mc = run_marching_cubes(pial_path)
    mc_time = time.time() - t0
    print(f"  MC done in {mc_time:.1f}s, vertices: {len(mc.vertices):,}")
    print(f"  Computing metrics...")
    assd, hd95, hd99 = compute_metrics(mc, gt)
    print(f"  ASSD={assd:.4f}mm  HD95={hd95:.4f}mm  HD99={hd99:.4f}mm")

print("\nTest complete!")
EOF
