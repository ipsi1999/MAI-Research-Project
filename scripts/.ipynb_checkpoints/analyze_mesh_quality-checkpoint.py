#!/usr/bin/env python3
"""
Analyze geometric quality of pial surface meshes from Marching Cubes.

Checks:
- Watertightness
- Connected components (should be 1)
- Genus (should be 0 for topologically correct surface)
- Face area statistics
"""

import trimesh
import numpy as np
import nibabel as nib
from skimage import measure
import os
import sys
from argparse import ArgumentParser

def analyze_mesh(pred_dir, subject, resolution):
    """Generate mesh and compute quality metrics."""
    path = os.path.join(pred_dir, f'{subject}.nii.gz')
    if not os.path.exists(path):
        return None
    
    img = nib.load(path)
    mask = img.get_fdata().astype(np.uint8)
    spacing = img.header.get_zooms()
    
    verts, faces, _, _ = measure.marching_cubes(mask, level=0.5, spacing=spacing)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    
    largest = max(mesh.split(), key=lambda m: len(m.vertices))
    
    return {
        'vertices': len(mesh.vertices),
        'faces': len(mesh.faces),
        'components': len(mesh.split()),
        'genus': 1 - mesh.euler_number // 2,
        'watertight': mesh.is_watertight,
        'mean_area': mesh.area_faces.mean(),
        'largest_vertices': len(largest.vertices),
        'largest_genus': 1 - largest.euler_number // 2,
    }

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--pred-dir', required=True)
    parser.add_argument('--subjects', nargs='+', required=True)
    args = parser.parse_args()
    
    for subj in args.subjects:
        stats = analyze_mesh(args.pred_dir, subj, 'k4_deeper')
        if stats:
            print(f'{subj}: {stats["components"]} components, genus={stats["genus"]}, watertight={stats["watertight"]}')
