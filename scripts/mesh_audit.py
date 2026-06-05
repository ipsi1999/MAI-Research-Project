"""
Mesh Topology Audit Script v2 — with Largest Connected Component (LCC) extraction

Runs MC on predicted masks, extracts the largest connected component,
compares Original vs LCC topology, and exports cleaned meshes.

Usage:
    python mesh_audit_v2.py --masks /path/to/pred1.nii.gz /path/to/pred2.nii.gz
    python mesh_audit_v2.py --masks /path/to/pred1.nii.gz --output_dir /path/to/cleaned_meshes
"""

import argparse
import os
import numpy as np
import trimesh
import nibabel as nib
from skimage.measure import marching_cubes
import warnings
warnings.filterwarnings("ignore")


def mask_to_mesh(nii_path):
    """Run Marching Cubes on a NIfTI mask and return trimesh object."""
    img = nib.load(nii_path)
    data = img.get_fdata()
    affine = img.affine

    verts, faces, _, _ = marching_cubes(data, level=0.5)
    verts_physical = nib.affines.apply_affine(affine, verts)
    verts_physical[:, 0] *= -1
    verts_physical[:, 1] *= -1

    mesh = trimesh.Trimesh(vertices=verts_physical, faces=faces)
    mesh.fix_normals()
    return mesh


def extract_lcc(mesh):
    """Extract the largest connected component from a mesh."""
    components = mesh.split(only_watertight=False)
    lcc = max(components, key=lambda c: len(c.vertices))
    return lcc, len(components)


def compute_topology(mesh):
    """Compute topology metrics for a mesh."""
    V = len(mesh.vertices)
    F = len(mesh.faces)
    E = len(mesh.edges_unique)
    euler = V - E + F
    genus = (2 - euler) // 2 if euler <= 2 else 0

    return {
        "Vertices": V,
        "Faces": F,
        "Edges": E,
        "Is Watertight": mesh.is_watertight,
        "Euler Characteristic": euler,
        "Genus": genus,
        "Surface Area (mm²)": f"{mesh.area:.1f}",
        "Volume (mm³)": f"{abs(mesh.volume):.1f}" if mesh.is_watertight else "N/A",
    }


def print_subject_table(name, original_topo, lcc_topo, n_components, n_removed_verts):
    """Print comparison table for one subject: Original vs LCC."""
    col1 = 24
    col2 = 22
    col3 = 22
    width = col1 + col2 + col3 + 6

    print(f"\n{'=' * width}")
    print(f"  {name}")
    print(f"{'=' * width}")
    print(f"{'Metric':<{col1}} | {'Original':<{col2}} | {'LCC (cleaned)':<{col3}}")
    print(f"{'-' * width}")

    keys = ["Vertices", "Faces", "Edges", "Is Watertight",
            "Euler Characteristic", "Genus", "Surface Area (mm²)", "Volume (mm³)"]

    for key in keys:
        orig_val = str(original_topo.get(key, ""))
        lcc_val = str(lcc_topo.get(key, ""))
        print(f"{key:<{col1}} | {orig_val:<{col2}} | {lcc_val:<{col3}}")

    print(f"{'-' * width}")
    print(f"{'Components':<{col1}} | {n_components:<{col2}} | {'1 (largest only)':<{col3}}")
    print(f"{'Vertices removed':<{col1}} | {'':<{col2}} | {n_removed_verts} ({n_removed_verts/original_topo['Vertices']*100:.2f}%)")
    print(f"{'Genus reduction':<{col1}} | {original_topo['Genus']:<{col2}} | {lcc_topo['Genus']} ({original_topo['Genus'] - lcc_topo['Genus']} fewer tunnels)")
    print(f"{'=' * width}")


def print_summary_table(all_subjects):
    """Print a final summary comparing all subjects."""
    if len(all_subjects) < 2:
        return

    print(f"\n{'=' * 90}")
    print(f"  SUMMARY COMPARISON")
    print(f"{'=' * 90}")
    print(f"{'Subject':<20} | {'Components':<12} | {'Genus (orig)':<14} | {'Genus (LCC)':<13} | {'Watertight':<12}")
    print(f"{'-' * 90}")

    for s in all_subjects:
        print(f"{s['name']:<20} | {s['n_components']:<12} | {s['orig_genus']:<14} | {s['lcc_genus']:<13} | {s['lcc_watertight']}")

    print(f"{'=' * 90}")


def main():
    parser = argparse.ArgumentParser(description="Mesh topology audit with LCC extraction")
    parser.add_argument("--masks", nargs="+", required=True, help="NIfTI mask files")
    parser.add_argument("--output_dir", default=None,
                        help="Directory to save cleaned LCC meshes (default: same dir as input)")
    args = parser.parse_args()

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    all_subjects = []

    for path in args.masks:
        name = os.path.basename(path).replace(".nii.gz", "")
        print(f"\nProcessing {name}...", flush=True)

        # Run MC
        mesh = mask_to_mesh(path)
        original_topo = compute_topology(mesh)

        # Extract LCC
        lcc, n_components = extract_lcc(mesh)
        lcc_topo = compute_topology(lcc)
        n_removed = len(mesh.vertices) - len(lcc.vertices)

        # Print comparison
        print_subject_table(name, original_topo, lcc_topo, n_components, n_removed)

        # Save cleaned mesh
        out_dir = args.output_dir or os.path.dirname(path)
        out_path = os.path.join(out_dir, f"{name}_LCC.obj")
        lcc.export(out_path)
        print(f"  Saved cleaned mesh → {out_path}")

        # Collect for summary
        all_subjects.append({
            "name": name,
            "n_components": n_components,
            "orig_genus": original_topo["Genus"],
            "lcc_genus": lcc_topo["Genus"],
            "lcc_watertight": lcc_topo["Is Watertight"],
        })

    # Print summary if multiple subjects
    print_summary_table(all_subjects)

    print("\nINTERPRETATION:")
    print("  Genus = 0, Euler = 2  → topologically a sphere (ideal)")
    print("  Genus > 0             → tunnels/handles exist (common in brain meshes)")
    print("  LCC extraction removes small floating fragments from MC noise")
    print("  Cleaned .obj files can be viewed in 3D Slicer or MeshLab")


if __name__ == "__main__":
    main()