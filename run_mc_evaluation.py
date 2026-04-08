"""
Part 1: Marching Cubes evaluation across all subjects and k levels.
Resumes from partial results if they exist.
Run via SLURM — do not run interactively for k4.
"""

import os
import time
import numpy as np
import nibabel as nib
from skimage import measure
import trimesh
import pandas as pd
from scipy.spatial import cKDTree
import gc

DATA_ROOT = "/home/isin038/00_nesi_projects/uoa04396/isin038/data/IXI"
OUTPUT_DIR = "/home/isin038/00_nesi_projects/uoa04396/isin038/results/part1"
K_LEVELS = ["k2", "k3", "k4"]
PARTIAL_CSV = os.path.join(OUTPUT_DIR, "mc_results_partial.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)


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
    cras_offset = meta['cras']
    vertices_ras = vertices + cras_offset
    vertices_ras[:, 0] *= -1
    vertices_ras[:, 1] *= -1
    mesh = trimesh.Trimesh(vertices=vertices_ras, faces=faces)
    return mesh


def get_surface_distances(src_vertices, tgt_vertices):
    tree = cKDTree(tgt_vertices)
    distances, _ = tree.query(src_vertices)
    return distances


def compute_metrics(mesh_pred, mesh_gt, n_samples=100000):
    # Sample points from surface (matching RFENet trimesh evaluation)
    pts_pred = mesh_pred.sample(n_samples)
    pts_gt = mesh_gt.sample(n_samples)

    # Point-to-face distance (matches RFENet eval_ad_hd_trimesh)
    _, P2G_dist, _ = trimesh.proximity.closest_point(mesh_pred, pts_gt)
    _, G2P_dist, _ = trimesh.proximity.closest_point(mesh_gt, pts_pred)

    assd = (P2G_dist.sum() + G2P_dist.sum()) / float(P2G_dist.size + G2P_dist.size)
    hd90 = max(np.percentile(P2G_dist, 90), np.percentile(G2P_dist, 90))
    hd95 = max(np.percentile(P2G_dist, 95), np.percentile(G2P_dist, 95))
    hd99 = max(np.percentile(P2G_dist, 99), np.percentile(G2P_dist, 99))

    return assd, hd90, hd95, hd99


def main():
    subjects = sorted([
        d for d in os.listdir(DATA_ROOT)
        if os.path.isdir(os.path.join(DATA_ROOT, d)) and d.startswith("IXI")
    ])
    print(f"Found {len(subjects)} subjects total")

    # Load existing partial results to resume from
    if os.path.exists(PARTIAL_CSV):
        existing_df = pd.read_csv(PARTIAL_CSV)
        # A subject is fully done if it has results for all 3 k levels
        done_subjects = set(
            existing_df.groupby("subject").filter(
                lambda x: len(x) == len(K_LEVELS)
            )["subject"].unique()
        )
        results = existing_df.to_dict("records")
        print(f"Resuming — {len(done_subjects)} subjects already completed, skipping them")
    else:
        done_subjects = set()
        results = []
        print("Starting fresh")

    remaining = [s for s in subjects if s not in done_subjects]
    print(f"Subjects remaining: {len(remaining)}")

    for i, subj in enumerate(remaining):
        print(f"\n[{i+1}/{len(remaining)}] {subj}")
        subj_dir = os.path.join(DATA_ROOT, subj)

        lh_path = os.path.join(subj_dir, "surf", "lh.pial")
        rh_path = os.path.join(subj_dir, "surf", "rh.pial")

        if not os.path.exists(lh_path) or not os.path.exists(rh_path):
            print(f"  Skipping — missing ground truth surfaces")
            continue

        mesh_lh = load_freesurfer_surface(lh_path)
        mesh_rh = load_freesurfer_surface(rh_path)
        gt_mesh = trimesh.util.concatenate([mesh_lh, mesh_rh])
        

        for k in K_LEVELS:
            pial_path = os.path.join(
                subj_dir, "proc", "upsampling", f"pial_resampled_{k}.nii.gz"
            )

            if not os.path.exists(pial_path):
                print(f"  Skipping {k} — missing mask")
                continue

            try:
                t0 = time.time()
                mc_mesh = run_marching_cubes(pial_path)
                mc_time = time.time() - t0
                n_verts = len(mc_mesh.vertices)

                assd, hd95, hd99 = compute_metrics(mc_mesh, gt_mesh)

                del mc_mesh #free RAM 

                gc.collect()

                print(f"  {k}: ASSD={assd:.4f}mm  HD95={hd95:.4f}mm  HD99={hd99:.4f}mm  time={mc_time:.1f}s")

                results.append({
                    "subject": subj,
                    "k_level": k,
                    "assd_mm": assd,
                    "hd90_mm": hd90,
                    "hd95_mm": hd95,
                    "hd99_mm": hd99,
                    "mc_time_s": mc_time,
                    "n_vertices": n_verts
                })

            except Exception as e:
                print(f"  Error {k}: {e}")
                continue

        # Save incrementally after each subject
        df = pd.DataFrame(results)
        df.to_csv(PARTIAL_CSV, index=False)

    # Final save
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR, "mc_results_final.csv"), index=False)

    # Summary
    summary = df.groupby("k_level").agg(
        assd_mean=("assd_mm", "mean"),
        assd_std=("assd_mm", "std"),
        hd90_mean=("hd90_mm", "mean"),
        hd90_std=("hd90_mm", "std"),
        hd95_mean=("hd95_mm", "mean"),
        hd95_std=("hd95_mm", "std"),
        hd99_mean=("hd99_mm", "mean"),
        hd99_std=("hd99_mm", "std"),
        mc_time_mean=("mc_time_s", "mean"),
        n_subjects=("subject", "count")
    ).reset_index()

    summary.to_csv(os.path.join(OUTPUT_DIR, "mc_summary.csv"), index=False)

    print("\n=== FINAL SUMMARY ===")
    print(summary.to_string(index=False))
    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
