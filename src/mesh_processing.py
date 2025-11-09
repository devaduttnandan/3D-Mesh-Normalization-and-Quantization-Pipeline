#!/usr/bin/env python3
import os
import argparse
import numpy as np
import trimesh
import matplotlib.pyplot as plt

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_mesh(path):
    mesh = trimesh.load(path, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(mesh.dump())
        else:
            raise ValueError("Loaded object is not a mesh or scene")
    return mesh

def save_mesh(mesh, path):
    mesh.export(path)
    print(f"Saved: {path}")


def mesh_vertex_stats(vertices):
    stats = {}
    stats['n_vertices'] = vertices.shape[0]
    stats['min'] = vertices.min(axis=0)
    stats['max'] = vertices.max(axis=0)
    stats['mean'] = vertices.mean(axis=0)
    stats['std'] = vertices.std(axis=0)
    return stats

def print_stats(name, stats):
    print(f"\n--- {name} ---")
    print(f"Number of vertices: {stats['n_vertices']}")
    print(f"Min per axis: x={stats['min'][0]:.6f}, y={stats['min'][1]:.6f}, z={stats['min'][2]:.6f}")
    print(f"Max per axis: x={stats['max'][0]:.6f}, y={stats['max'][1]:.6f}, z={stats['max'][2]:.6f}")
    print(f"Mean per axis: x={stats['mean'][0]:.6f}, y={stats['mean'][1]:.6f}, z={stats['mean'][2]:.6f}")
    print(f"Std per axis:  x={stats['std'][0]:.6f}, y={stats['std'][1]:.6f}, z={stats['std'][2]:.6f}")

def normalize_minmax(vertices, a=0.0, b=1.0):
    """Min-Max normalize separately per axis into [a,b]. Returns normalized verts and params."""
    v_min = vertices.min(axis=0)
    v_max = vertices.max(axis=0)
    denom = np.where(v_max - v_min == 0, 1.0, (v_max - v_min))
    normalized = (vertices - v_min) / denom
    normalized = normalized * (b - a) + a
    params = {'method': 'minmax', 'v_min': v_min, 'v_max': v_max, 'a': a, 'b': b}
    return normalized, params

def denormalize_minmax(normalized_vertices, params):
    a = params['a']; b = params['b']
    v_min = params['v_min']; v_max = params['v_max']
    denom = np.where(v_max - v_min == 0, 1.0, (v_max - v_min))
    x = (normalized_vertices - a) / (b - a)
    return x * denom + v_min

def normalize_unit_sphere(vertices):
    centroid = vertices.mean(axis=0)
    centered = vertices - centroid
    # compute max distance
    max_dist = np.max(np.linalg.norm(centered, axis=1))
    if max_dist == 0:
        scale = 1.0
    else:
        scale = 1.0 / max_dist
    normalized = centered * scale
    params = {'method': 'unit_sphere', 'centroid': centroid, 'scale': scale}
    return normalized, params

def denormalize_unit_sphere(norm_vertices, params):
    scale = params['scale']
    centroid = params['centroid']
    return (norm_vertices / scale) + centroid

def quantize_normalized(norm_vertices, n_bins=1024):

    clamped = np.clip(norm_vertices, 0.0, 1.0)
    q = np.floor(clamped * (n_bins - 1.0) + 0.5).astype(np.int64)  # rounding
    return q

def dequantize(q, n_bins=1024):
    return (q.astype(np.float64)) / (n_bins - 1.0)


def compute_mse_per_axis(original, reconstructed):
    diffs = original - reconstructed
    mse_axes = np.mean(diffs ** 2, axis=0)
    mse_overall = np.mean(np.sum(diffs ** 2, axis=1))
    return mse_axes, mse_overall


def plot_errors(mse_axes_minmax, mse_axes_unit, out_folder, mesh_name):
    ensure_dir(out_folder)
    axes = ['x', 'y', 'z']
    x = np.arange(len(axes))
    width = 0.35

    plt.figure(figsize=(6,4))
    plt.bar(x - width/2, mse_axes_minmax, width=width, label='Min-Max')
    plt.bar(x + width/2, mse_axes_unit, width=width, label='Unit-Sphere')
    plt.xticks(x, axes)
    plt.ylabel('MSE (per axis)')
    plt.title(f'Reconstruction MSE per axis — {mesh_name}')
    plt.legend()
    outpath = os.path.join(out_folder, f"{mesh_name}_mse_per_axis.png")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"Saved error plot: {outpath}")


def process_one_mesh(input_path, output_root='./output', n_bins=1024, do_visualize=False):
    mesh_name = os.path.splitext(os.path.basename(input_path))[0]
    out = os.path.join(output_root, mesh_name)
    ensure_dir(out)

    mesh = load_mesh(input_path)
    vertices = mesh.vertices.copy()
    faces = mesh.faces.copy() if mesh.faces is not None else None

    stats_orig = mesh_vertex_stats(vertices)
    print_stats("Original Mesh", stats_orig)
    save_mesh(mesh, os.path.join(out, f"{mesh_name}_original.obj"))
    norm_mm, params_mm = normalize_minmax(vertices, a=0.0, b=1.0)
    mesh_norm_mm = trimesh.Trimesh(vertices=norm_mm, faces=faces, process=False)
    save_mesh(mesh_norm_mm, os.path.join(out, f"{mesh_name}_normalized_minmax.obj"))
    q_mm = quantize_normalized(norm_mm, n_bins=n_bins)  # integers in [0, n_bins-1]
    dq_mm = dequantize(q_mm, n_bins=n_bins)
    mesh_q_mm = trimesh.Trimesh(vertices=dq_mm, faces=faces, process=False)
    save_mesh(mesh_q_mm, os.path.join(out, f"{mesh_name}_quantized_minmax.obj"))
    norm_us_raw, params_us = normalize_unit_sphere(vertices)
    remapped_us = (norm_us_raw + 1.0) / 2.0
    mesh_norm_us = trimesh.Trimesh(vertices=norm_us_raw, faces=faces, process=False)
    save_mesh(mesh_norm_us, os.path.join(out, f"{mesh_name}_normalized_unit_sphere.obj"))

    q_us = quantize_normalized(remapped_us, n_bins=n_bins)
    dq_us_remapped = dequantize(q_us, n_bins=n_bins)
    dq_us = dq_us_remapped * 2.0 - 1.0
    mesh_q_us = trimesh.Trimesh(vertices=dq_us, faces=faces, process=False)
    save_mesh(mesh_q_us, os.path.join(out, f"{mesh_name}_quantized_unit_sphere.obj"))

    reconstructed_norm_mm = dequantize(q_mm, n_bins=n_bins)  # in [0,1]
    reconstructed_mm = denormalize_minmax(reconstructed_norm_mm, params_mm)
    mesh_rec_mm = trimesh.Trimesh(vertices=reconstructed_mm, faces=faces, process=False)
    save_mesh(mesh_rec_mm, os.path.join(out, f"{mesh_name}_reconstructed_minmax.obj"))

    rec_remapped_us = dequantize(q_us, n_bins=n_bins)  # in [0,1]
    rec_us = rec_remapped_us * 2.0 - 1.0  # back to [-1,1]
    reconstructed_us = denormalize_unit_sphere(rec_us, params_us)
    mesh_rec_us = trimesh.Trimesh(vertices=reconstructed_us, faces=faces, process=False)
    save_mesh(mesh_rec_us, os.path.join(out, f"{mesh_name}_reconstructed_unit_sphere.obj"))

    mse_axes_mm, mse_overall_mm = compute_mse_per_axis(vertices, reconstructed_mm)
    mse_axes_us, mse_overall_us = compute_mse_per_axis(vertices, reconstructed_us)

    print("\n--- Reconstruction Errors ---")
    print(f"Min-Max MSE per axis: x={mse_axes_mm[0]:.8e}, y={mse_axes_mm[1]:.8e}, z={mse_axes_mm[2]:.8e}")
    print(f"Min-Max overall MSE: {mse_overall_mm:.8e}")
    print(f"Unit-Sphere MSE per axis: x={mse_axes_us[0]:.8e}, y={mse_axes_us[1]:.8e}, z={mse_axes_us[2]:.8e}")
    print(f"Unit-Sphere overall MSE: {mse_overall_us:.8e}")

    plot_errors(mse_axes_mm, mse_axes_us, out, mesh_name)

    summary_lines = [
        f"Mesh: {mesh_name}",
        f"Vertices: {stats_orig['n_vertices']}",
        "",
        "Min-Max normalization:",
        f"  MSE per axis: x={mse_axes_mm[0]:.8e}, y={mse_axes_mm[1]:.8e}, z={mse_axes_mm[2]:.8e}",
        f"  Overall MSE: {mse_overall_mm:.8e}",
        "",
        "Unit-Sphere normalization:",
        f"  MSE per axis: x={mse_axes_us[0]:.8e}, y={mse_axes_us[1]:.8e}, z={mse_axes_us[2]:.8e}",
        f"  Overall MSE: {mse_overall_us:.8e}",
        "",
        "Short conclusion:",
    ]

    if mse_overall_mm < mse_overall_us:
        summary_lines.append("  Min-Max normalization + uniform quantization produced lower overall MSE for this mesh.")
    elif mse_overall_us < mse_overall_mm:
        summary_lines.append("  Unit-Sphere normalization + uniform quantization produced lower overall MSE for this mesh.")
    else:
        summary_lines.append("  Both normalization methods produced similar overall MSE for this mesh.")

    summary_lines.append("")
    summary_lines.append("Notes:")
    summary_lines.append("  - Min-Max preserves per-axis scale; unit-sphere preserves radial structure.")
    summary_lines.append("  - Quantization uses rounding to the nearest of 1024 bins per axis (after mapping to [0,1]).")
    summary_lines.append("  - For more robust comparisons, run the same pipeline on multiple meshes and compare errors.")

    summary_path = os.path.join(out, f"{mesh_name}_summary.txt")
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    print(f"\nSaved summary: {summary_path}")

    print("\nAll outputs saved in:", out)
    return {
        'mse_axes_mm': mse_axes_mm, 'mse_overall_mm': mse_overall_mm,
        'mse_axes_us': mse_axes_us, 'mse_overall_us': mse_overall_us,
        'out_folder': out
    }


def main():
    parser = argparse.ArgumentParser(description="Mesh normalization, quantization, reconstruction and error analysis (single example).")
    parser.add_argument('--input', '-i', required=False,
                        help="Path to input .obj mesh. Defaults to /mnt/data/girl.obj", 
                        default="/mnt/data/girl.obj")
    parser.add_argument('--output', '-o', required=False, help="Output folder", default="./output")
    parser.add_argument('--bins', '-b', type=int, default=1024, help="Quantization bins")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    print(f"Processing: {args.input}")
    process_one_mesh(args.input, output_root=args.output, n_bins=args.bins)

if __name__ == '__main__':
    main()
