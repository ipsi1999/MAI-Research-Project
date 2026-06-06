#!/usr/bin/env python3
"""
Post-processing pipeline for pial surface meshes:
  1. Keep largest connected component
  2. Taubin smoothing
  3. Save as .ply for visualisation and .nii.gz for re-evaluation
"""

import trimesh
import numpy as np
import nibabel as nib
from skimage import measure
import os
import argparse

def load_mesh_from_mask(pred_path):
    img = nib.load(pred_path)
    mask = img.get_fdata().astype(np.uint8)
    spacing = img.header.get_zooms()
    affine = img.affine
    verts, faces, _, _ = measure.marching_cubes(mask, level=0.5, spacing=spacing)
    return verts, faces, affine, spacing, mask.shape

def postprocess_mesh(verts, faces, smooth_iters=10, smooth_lambda=0.5, smooth_nu=0.53):
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)

    # Step 1: Keep largest connected component
    components = mesh.split()
    print(f'  Components before: {len(components)}')
    mesh = max(components, key=lambda m: len(m.vertices))
    print(f'  Vertices after largest-only: {len(mesh.vertices):,}')

    # Step 2: Taubin smoothing
    # Alternates shrink (lambda) and inflate (nu) steps to preserve volume
    for i in range(smooth_iters):
        trimesh.smoothing.filter_laplacian(mesh, lamb=smooth_lambda, iterations=1)
        trimesh.smoothing.filter_laplacian(mesh, lamb=-smooth_nu, iterations=1)
    print(f'  Smoothing done ({smooth_iters} Taubin iterations)')

    genus = 1 - mesh.euler_number // 2
    print(f'  Genus after post-processing: {genus}')
    print(f'  Watertight: {mesh.is_watertight}')

    return mesh

def save_mesh(mesh, out_path_ply):
    os.makedirs(os.path.dirname(out_path_ply), exist_ok=True)
    mesh.export(out_path_ply)
    print(f'  Saved: {out_path_ply}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', required=True, help='e.g. k3, k4_deeper')
    parser.add_argument('--subjects', nargs='+', default=None, help='Optional: specific subjects')
    parser.add_argument('--smooth-iters', type=int, default=10)
    args = parser.parse_args()

    PRED_DIR = f'/nesi/project/uoa04396/isin038/results/part2/predictions/{args.resolution}'
    OUT_DIR = f'/nesi/project/uoa04396/isin038/results/part2/meshes_postprocessed/{args.resolution}'

    subjects = args.subjects or sorted([
        f.replace('.nii.gz', '')
        for f in os.listdir(PRED_DIR)
        if f.endswith('.nii.gz') and not f.endswith('_0000.nii.gz')
    ])

    for subj in subjects:
        print(f'\n=== {subj} ===')
        pred_path = os.path.join(PRED_DIR, f'{subj}.nii.gz')
        if not os.path.exists(pred_path):
            print(f'  SKIPPING: file not found')
            continue

        verts, faces, affine, spacing, shape = load_mesh_from_mask(pred_path)
        mesh = postprocess_mesh(verts, faces, smooth_iters=args.smooth_iters)

        out_ply = os.path.join(OUT_DIR, f'{subj}.ply')
        save_mesh(mesh, out_ply)

    print('\nDone.')
