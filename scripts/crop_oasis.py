#!/usr/bin/env python3
import nibabel as nib
import numpy as np
import os
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--resolution', required=True, choices=['k3', 'k4'])
parser.add_argument('--threshold', type=float, default=10.0)
args = parser.parse_args()

IN_DIR = f'/nesi/project/uoa04396/isin038/data/OASIS-1/preprocessed/{args.resolution}'
OUT_DIR = f'/nesi/project/uoa04396/isin038/data/OASIS-1/preprocessed_cropped/{args.resolution}'
os.makedirs(OUT_DIR, exist_ok=True)

files = sorted([f for f in os.listdir(IN_DIR) if f.endswith('.nii.gz')])
print(f'Cropping {len(files)} files at {args.resolution}', flush=True)

bad_files = []
for fname in files:
    out_path = os.path.join(OUT_DIR, fname)
    if os.path.exists(out_path):
        continue
    
    t0 = time.time()
    try:
        img = nib.load(os.path.join(IN_DIR, fname))
        data = img.get_fdata()
    except Exception as e:
        print(f'  {fname}: CORRUPTED - {e}', flush=True)
        bad_files.append(fname)
        continue
    
    mask = data > args.threshold
    x = np.any(mask, axis=(1, 2))
    y = np.any(mask, axis=(0, 2))
    z = np.any(mask, axis=(0, 1))
    
    if not x.any():
        print(f'  {fname}: empty', flush=True)
        continue
    
    xmin, xmax = np.where(x)[0][[0, -1]]
    ymin, ymax = np.where(y)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]
    
    xmin = max(xmin - 10, 0); xmax = min(xmax + 11, data.shape[0])
    ymin = max(ymin - 10, 0); ymax = min(ymax + 11, data.shape[1])
    zmin = max(zmin - 10, 0); zmax = min(zmax + 11, data.shape[2])
    
    cropped = data[xmin:xmax, ymin:ymax, zmin:zmax]
    
    new_affine = img.affine.copy()
    new_affine[:3, 3] = img.affine[:3, 3] + img.affine[:3, :3] @ np.array([xmin, ymin, zmin])
    
    nib.save(nib.Nifti1Image(cropped, new_affine), out_path)
    print(f'  {fname}: {img.shape} → {cropped.shape}  ({time.time()-t0:.1f}s)', flush=True)

if bad_files:
    print(f'\nCORRUPTED FILES ({len(bad_files)}):')
    for f in bad_files:
        print(f'  {f}')

print('Done')
