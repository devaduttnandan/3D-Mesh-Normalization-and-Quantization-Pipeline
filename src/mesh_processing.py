#!/usr/bin/env python3
import os
import argparse
import open3d as o3d
import numpy as np
import trimesh
import matplotlib.pyplot as plt

# Utility 
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

# Stats 
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
    print(f"Vertices: {stats['n_vertices']}")
    print(f"Min per axis: {stats['min']}")
    print(f"Max per axis: {stats['max']}")
    print(f"Mean per axis: {stats['mean']}")
    print(f"Std per axis:  {stats['std']}")

# Normalization
def normalize_minmax(vertices, a=0.0, b=1.0):
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
    max_dist = np.max(np.linalg.norm(centered, axis=1))
    scale = 1.0 / max_dist if max_dist != 0 else 1.0
    normalized = centered * scale
    params = {'method': 'unit_sphere', 'centroid': centroid, 'scale': scale}
    return normalized, params

def denormalize_unit_sphere(norm_vertices, params):
    scale = params['scale']
    centroid = params['centroid']
    return (norm_vertices / scale) + centroid

# Quantization
def quantize_normalized(norm_vertices, n_bins=1024):
    clamped = np.clip(norm_vertices, 0.0, 1.0)
    q = np.floor(clamped * (n_bins - 1.0) + 0.5).astype(np.int64)
    return q

def dequantize(q, n_bins=1024):
    return (q.astype(np.float64)) / (n_bins - 1.0)

# Error Calculation
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
    plt.bar(x + width/2, mse_axes_unit, width=width, label='Unit Sphere')
    plt.xticks(x, axes)
    plt.ylabel('MSE (per axis)')
    plt.title(f'Reconstruction MSE — {mesh_name}')
    plt.legend()
    outpath = os.path.join(out_folder, f"{mesh_name}_mse_per_axis.png")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"Saved error plot: {outpath}")

# Visualization
def visualize_mesh(mesh, title="3D Mesh"):
    
    print(f"Visualizing: {title}")
    mesh_o3d = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(mesh.vertices),
        triangles=o3d.utility.Vector3iVector(mesh.faces)
    )
    mesh_o3d.compute_vertex_normals()
    mesh_o3d.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([mesh_o3d], window_name=title)

def visualize_comparison(mesh1, mesh2, title1="Original", title2="Processed", save_path=None):
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(mesh1.vertices[:,0], mesh1.vertices[:,1], mesh1.vertices[:,2], s=1, color='blue')
    ax1.set_title(title1)
    ax1.axis('off')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(mesh2.vertices[:,0], mesh2.vertices[:,1], mesh2.vertices[:,2], s=1, color='red')
    ax2.set_title(title2)
    ax2.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved comparison: {save_path}")
    plt.close()

def save_mesh_screenshot(mesh, filepath):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    mesh_o3d = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(mesh.vertices),
        triangles=o3d.utility.Vector3iVector(mesh.faces)
    )
    mesh_o3d.compute_vertex_normals()
    vis.add_geometry(mesh_o3d)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(filepath)
    vis.destroy_window()
    print(f"Screenshot saved: {filepath}")

# Main Processing
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

    # Visualize Original
    if do_visualize:
        visualize_mesh(mesh, "Original Mesh")
        save_mesh_screenshot(mesh, os.path.join(out, f"{mesh_name}_original.png"))

    #Min-Max Normalization
    norm_mm, params_mm = normalize_minmax(vertices)
    mesh_norm_mm = trimesh.Trimesh(vertices=norm_mm, faces=faces, process=False)
    save_mesh(mesh_norm_mm, os.path.join(out, f"{mesh_name}_normalized_minmax.obj"))

    if do_visualize:
        visualize_comparison(mesh, mesh_norm_mm, "Original", "Min-Max Normalized",
                             os.path.join(out, f"{mesh_name}_compare_minmax.png"))
        save_mesh_screenshot(mesh_norm_mm, os.path.join(out, f"{mesh_name}_normalized_minmax.png"))

    # Quantization + Reconstruction
    q_mm = quantize_normalized(norm_mm, n_bins=n_bins)
    reconstructed_norm_mm = dequantize(q_mm, n_bins=n_bins)
    reconstructed_mm = denormalize_minmax(reconstructed_norm_mm, params_mm)
    mesh_rec_mm = trimesh.Trimesh(vertices=reconstructed_mm, faces=faces, process=False)
    save_mesh(mesh_rec_mm, os.path.join(out, f"{mesh_name}_reconstructed_minmax.obj"))

    if do_visualize:
        visualize_comparison(mesh, mesh_rec_mm, "Original", "Reconstructed (Min-Max)",
                             os.path.join(out, f"{mesh_name}_compare_reconstructed_minmax.png"))

    #Unit Sphere Normalization 
    norm_us_raw, params_us = normalize_unit_sphere(vertices)
    remapped_us = (norm_us_raw + 1.0) / 2.0
    mesh_norm_us = trimesh.Trimesh(vertices=norm_us_raw, faces=faces, process=False)
    save_mesh(mesh_norm_us, os.path.join(out, f"{mesh_name}_normalized_unit_sphere.obj"))

    if do_visualize:
        visualize_comparison(mesh, mesh_norm_us, "Original", "Unit Sphere Normalized",
                             os.path.join(out, f"{mesh_name}_compare_unit_sphere.png"))
        save_mesh_screenshot(mesh_norm_us, os.path.join(out, f"{mesh_name}_normalized_unit_sphere.png"))

    # Quantization + Reconstruction
    q_us = quantize_normalized(remapped_us, n_bins=n_bins)
    dq_us_remapped = dequantize(q_us, n_bins=n_bins)
    dq_us = dq_us_remapped * 2.0 - 1.0
    reconstructed_us = denormalize_unit_sphere(dq_us, params_us)
    mesh_rec_us = trimesh.Trimesh(vertices=reconstructed_us, faces=faces, process=False)
    save_mesh(mesh_rec_us, os.path.join(out, f"{mesh_name}_reconstructed_unit_sphere.obj"))

    if do_visualize:
        visualize_comparison(mesh, mesh_rec_us, "Original", "Reconstructed (Unit Sphere)",
                             os.path.join(out, f"{mesh_name}_compare_reconstructed_unit_sphere.png"))

    # Compute Errors
    mse_axes_mm, mse_overall_mm = compute_mse_per_axis(vertices, reconstructed_mm)
    mse_axes_us, mse_overall_us = compute_mse_per_axis(vertices, reconstructed_us)

    plot_errors(mse_axes_mm, mse_axes_us, out, mesh_name)

    # Save Summary 
    summary_path = os.path.join(out, f"{mesh_name}_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Mesh: {mesh_name}\nVertices: {stats_orig['n_vertices']}\n\n")
        f.write("Min-Max Normalization:\n")
        f.write(f"  MSE per axis: {mse_axes_mm}\n  Overall MSE: {mse_overall_mm:.8e}\n\n")
        f.write("Unit Sphere Normalization:\n")
        f.write(f"  MSE per axis: {mse_axes_us}\n  Overall MSE: {mse_overall_us:.8e}\n\n")
        f.write("Conclusion:\n")
        if mse_overall_mm < mse_overall_us:
            f.write(" Min-Max normalization performed better.\n")
        elif mse_overall_us < mse_overall_mm:
            f.write("  Unit Sphere normalization performed better.\n")
        else:
            f.write("  Both performed similarly.\n")
    print(f"Summary saved: {summary_path}")
    print(f"\nAll results saved in: {out}")

#Main Entry
def main():
    parser = argparse.ArgumentParser(description="3D Mesh Normalization, Quantization & Visualization Pipeline")
    parser.add_argument('--input', '-i', required=False, default="/mnt/data/girl.obj", help="Input .obj file path")
    parser.add_argument('--output', '-o', default="./output", help="Output folder")
    parser.add_argument('--bins', '-b', type=int, default=1024, help="Quantization bins")
    parser.add_argument('--visualize', '-v', action='store_true', help="Enable 3D visualizations")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    print(f"Processing: {args.input}")
    process_one_mesh(args.input, output_root=args.output, n_bins=args.bins, do_visualize=args.visualize)

if __name__ == '__main__':
    main()
