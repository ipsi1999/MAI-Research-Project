"""
Prepares nnU-Net v2 dataset for pial segmentation at k2 and k3 resolutions.

Input  : brainmask_resampled_k*.nii.gz (already at correct resolution)
Label  : pial_resampled_k*.nii.gz      (already at correct resolution)

No resampling needed - just organise into nnU-Net folder structure.
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np

RESOLUTION_MAP = {
    "k2": {"dataset_id": "001", "voxel_mm": 0.46875},
    "k3": {"dataset_id": "002", "voxel_mm": 0.3125},
    "k4": {"dataset_id":"003", "voxel_mm": 0.234375},
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ixi_root",       required=True)
    parser.add_argument("--out_root",       required=True)
    parser.add_argument("--resolution",     required=True, choices=["k2","k3","k4"])
    parser.add_argument("--test_fraction",  type=float, default=0.15)
    parser.add_argument("--seed",           type=int,   default=42)
    args = parser.parse_args()

    ixi_root = Path(args.ixi_root)
    out_root = Path(args.out_root)
    res      = args.resolution
    cfg      = RESOLUTION_MAP[res]

    dataset_name = f"Dataset{cfg['dataset_id']}_PialK{res[1]}"
    dataset_dir  = out_root / dataset_name
    images_tr    = dataset_dir / "imagesTr"
    labels_tr    = dataset_dir / "labelsTr"
    images_ts    = dataset_dir / "imagesTs"
    labels_ts    = dataset_dir / "labelsTs"

    for d in [images_tr, labels_tr, images_ts, labels_ts]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"Preparing {dataset_name}")
    print(f"Output: {dataset_dir}")

    # collect subjects
    subjects = sorted([s for s in ixi_root.iterdir() if s.is_dir()])
    print(f"Total subjects: {len(subjects)}")

    # train/test split
    rng = random.Random(args.seed)
    rng.shuffle(subjects)
    n_test    = max(1, round(len(subjects) * args.test_fraction))
    test_set  = subjects[:n_test]
    train_set = subjects[n_test:]
    print(f"Train: {len(train_set)}   Test: {len(test_set)}")

    train_case_ids = []
    test_case_ids  = []

    for split_name, split_data, img_dir, lbl_dir, case_id_list in [
        ("train", train_set, images_tr, labels_tr, train_case_ids),
        ("test",  test_set,  images_ts, labels_ts, test_case_ids),
    ]:
        for i, subj_dir in enumerate(split_data):
            case_id   = subj_dir.name
            brain_path = subj_dir / "proc" / "upsampling" / f"brainmask_resampled_{res}.nii.gz"
            pial_path  = subj_dir / "proc" / "upsampling" / f"pial_resampled_{res}.nii.gz"

            if not brain_path.exists() or not pial_path.exists():
                print(f"  [SKIP] {case_id}: file missing")
                continue

            print(f"  [{split_name} {i+1}] {case_id}", flush=True)

            # input image: brainmask (_0000 = modality 0 in nnUNet)
            shutil.copy(brain_path, img_dir / f"{case_id}_0000.nii.gz")

            # label: pial mask (must be uint8)
            pial_img  = nib.load(pial_path)
            pial_data = pial_img.get_fdata(dtype=np.float32).astype(np.uint8)
            nib.save(
                nib.Nifti1Image(pial_data, pial_img.affine),
                lbl_dir / f"{case_id}.nii.gz"
            )

            case_id_list.append(case_id)

    # dataset.json (nnU-Net v2 format)
    dataset_json = {
        "channel_names": {"0": "brainmask"},
        "labels":        {"background": 0, "pial": 1},
        "numTraining":   len(train_case_ids),
        "file_ending":   ".nii.gz",
    }
    with open(dataset_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

    # save test case list for inference later
    (dataset_dir / "test_cases.txt").write_text("\n".join(test_case_ids))

    print(f"\nDone! Dataset saved to: {dataset_dir}")
    print(f"dataset.json written.")
    print(f"Test cases saved to: {dataset_dir}/test_cases.txt")

if __name__ == "__main__":
    main()