"""
Part 2: Marching Cubes evaluation on nnU-Net predicted masks.
Compares predicted pial meshes against FastSurfer ground truth.

Usage:
    python run_mc_evaluation_part2.py --resolution k2
    python run_mc_evaluation_part2.py --resolution k3
    python run_mc_evaluation_part2.py --resolution k4

Run via SLURM (05_evaluate.sh).
"""

import os
import time
import argparse
import numpy as np
import nibabel as nib
from skimage import measure
import trimesh
import pandas as pd
import gc

DATA_ROOT    = "/nesi/project/uoa04396/isin038/data/IXI"
PRED_ROOT    = "/nesi/project/uoa04396/isin038/results/part2/predictions"
OUTPUT_DIR   = "/nesi/project/uoa04396/isin038/results/part2/evaluation"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_marching_cubes(nii_path):
    img  = nib.load(nii_path)
    data = img.get_fdata()
    affine = img.affine
    verts, faces, _, _ = measure.marching_cubes(data, level=0.5)
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


def compute_metrics(mesh_pred, mesh_gt, n_samples=100000):
    pts_pred, _ = trimesh.sample.sample_surface(mesh_pred, n_samples)
    pts_gt,   _ = trimesh.sample.sample_surface(mesh_gt,   n_samples)

    _, P2G_dist, _ = trimesh.proximity.closest_point(mesh_gt,   pts_pred)
    _, G2P_dist, _ = trimesh.proximity.closest_point(mesh_pred, pts_gt)

    assd = (P2G_dist.sum() + G2P_dist.sum()) / float(P2G_dist.size + G2P_dist.size)
    hd90 = max(np.percentile(P2G_dist, 90), np.percentile(G2P_dist, 90))
    hd95 = max(np.percentile(P2G_dist, 95), np.percentile(G2P_dist, 95))
    hd99 = max(np.percentile(P2G_dist, 99), np.percentile(G2P_dist, 99))

    return assd, hd90, hd95, hd99


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", required=True, choices=["k2", "k3", "k4"])
    args = parser.parse_args()

    res      = args.resolution
    pred_dir = os.path.join(PRED_ROOT, res)
    out_csv  = os.path.join(OUTPUT_DIR, f"results_{res}.csv")
    partial  = os.path.join(OUTPUT_DIR, f"results_{res}_partial.csv")

    # find predicted masks
    pred_files = sorted([
        f for f in os.listdir(pred_dir)
        if f.endswith(".nii.gz") and f.startswith("IXI")
    ])
    print(f"Found {len(pred_files)} predicted masks for {res}")

    # resume from partial results
    if os.path.exists(partial):
        existing = pd.read_csv(partial)
        done = set(existing["subject"].tolist())
        results = existing.to_dict("records")
        print(f"Resuming — {len(done)} subjects already done")
    else:
        done    = set()
        results = []

    remaining = [f for f in pred_files if f.replace(".nii.gz", "") not in done]
    print(f"Subjects remaining: {len(remaining)}")

    for i, fname in enumerate(remaining):
        subj_id  = fname.replace(".nii.gz", "")
        pred_path = os.path.join(pred_dir, fname)
        subj_dir  = os.path.join(DATA_ROOT, subj_id)

        lh_path = os.path.join(subj_dir, "surf", "lh.pial")
        rh_path = os.path.join(subj_dir, "surf", "rh.pial")

        if not os.path.exists(lh_path) or not os.path.exists(rh_path):
            print(f"[{i+1}/{len(remaining)}] {subj_id} — SKIP (no GT surface)")
            continue

        print(f"[{i+1}/{len(remaining)}] {subj_id} ...", flush=True)

        try:
            t0 = time.time()

            # load GT mesh
            lh_mesh = load_freesurfer_surface(lh_path)
            rh_mesh = load_freesurfer_surface(rh_path)
            gt_mesh = trimesh.util.concatenate([lh_mesh, rh_mesh])

            # run MC on prediction
            pred_mesh = run_marching_cubes(pred_path)

            # compute metrics
            assd, hd90, hd95, hd99 = compute_metrics(pred_mesh, gt_mesh)

            elapsed = time.time() - t0
            print(f"  ASSD={assd:.4f}mm  HD95={hd95:.4f}mm  time={elapsed:.1f}s")

            results.append({
                "subject":    subj_id,
                "resolution": res,
                "assd_mm":    assd,
                "hd90_mm":    hd90,
                "hd95_mm":    hd95,
                "hd99_mm":    hd99,
                "time_s":     elapsed,
                "n_verts":    len(pred_mesh.vertices),
            })

            del pred_mesh, gt_mesh, lh_mesh, rh_mesh
            gc.collect()

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        # save incrementally
        pd.DataFrame(results).to_csv(partial, index=False)

    # final save
    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)

    # summary
    print(f"\n{'='*50}")
    print(f"Resolution: {res}  |  N={len(df)} subjects")
    print(f"  ASSD  mean={df.assd_mm.mean():.4f}  std={df.assd_mm.std():.4f}  median={df.assd_mm.median():.4f}")
    print(f"  HD95  mean={df.hd95_mm.mean():.4f}  std={df.hd95_mm.std():.4f}  median={df.hd95_mm.median():.4f}")
    print(f"\nResults saved to {out_csv}")


if __name__ == "__main__":
    main()
