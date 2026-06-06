#!/usr/bin/env python3
"""
Preprocess OASIS-1 FreeSurfer brain.mgz to match IXI input format
for k3 (0.3125mm) and k4 (0.234375mm) resolutions.
"""
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
import os
import argparse

OASIS_TEST = "/nesi/project/uoa04396/isin038/data/OASIS-1/test"
OUTPUT_BASE = "/nesi/project/uoa04396/isin038/data/OASIS-1/preprocessed"

RESOLUTION_MAP = {
    'k3': 0.3125,
    'k4': 0.234375,
}

parser = argparse.ArgumentParser()
parser.add_argument('--resolution', required=True, choices=['k3', 'k4'])
args = parser.parse_args()

TARGET_VOXEL = RESOLUTION_MAP[args.resolution]
OUTPUT_DIR = os.path.join(OUTPUT_BASE, args.resolution)
os.makedirs(OUTPUT_DIR, exist_ok=True)

subjects = sorted([s for s in os.listdir(OASIS_TEST) if s.startswith('OAS1')])
print(f"Found {len(subjects)} subjects | Resolution: {args.resolution} ({TARGET_VOXEL}mm)")

for subj in subjects:
    brain_path = os.path.join(OASIS_TEST, subj, 'mri', 'brain.mgz')
    out_path = os.path.join(OUTPUT_DIR, f'{subj}_0000.nii.gz')

    if not os.path.exists(brain_path):
        print(f'SKIP {subj}: brain.mgz not found')
        continue

    if os.path.exists(out_path):
        print(f'SKIP {subj}: already preprocessed')
        continue

    print(f'Processing {subj}...', flush=True)

    img = nib.load(brain_path)
    data = img.get_fdata()
    current_voxel = img.header.get_zooms()[:3]

    zoom_factors = tuple(c / TARGET_VOXEL for c in current_voxel)
    print(f'  Current voxel: {current_voxel}, zoom: {[round(z,3) for z in zoom_factors]}')

    resampled = zoom(data, zoom_factors, order=3, prefilter=True)
    print(f'  Resampled shape: {resampled.shape}')

    new_affine = img.affine.copy()
    for i in range(3):
        new_affine[i, i] = np.sign(img.affine[i, i]) * TARGET_VOXEL

    nib.save(nib.Nifti1Image(resampled, new_affine), out_path)
    print(f'  Saved: {out_path}')

print('\nDone!')
