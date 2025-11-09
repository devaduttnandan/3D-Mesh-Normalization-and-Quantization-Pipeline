# Mixar Assignment — Mesh Normalization, Quantization & Error Analysis

## Overview
This project implements **Tasks 1–3** of the Mixar assignment using Python and the Trimesh / Open3D libraries.  
It performs the following operations on 3D `.obj` mesh files:

1. **Load and inspect** the mesh (vertex statistics)
2. **Normalize** meshes using both:
   - Min–Max normalization  
   - Unit Sphere normalization  
3. **Quantize and dequantize** the vertex data into discrete bins (default = 1024)
4. **Reconstruct** meshes and compute per-axis & overall **Mean Squared Error (MSE)**
5. **Generate plots and summaries** comparing both normalization methods

## Setup
```bash
git clone https://github.com/devaduttnandan/3D-Mesh-Normalization-and-Quantization-Pipeline.git
cd 3D-Mesh-Normalization-and-Quantization-Pipeline
python3 -m venv venv
source venv/bin/activate (Linux)
venv\Scripts\activate (windows)
pip install -r requirements.txt
```
## Run

```bash
python run_all.py
```

## Run on a single mesh (with 3D visualizations)
```bash
python src/mesh_processing.py --input data/girl.obj --visualize
```

## Observations
1. **Min–Max normalization** generally gives lower reconstruction error for structured or symmetric models (e.g., girl.obj, person.obj, table.obj).

2. **Unit Sphere normalization** performs better for irregular and organic meshes (e.g., branch.obj, fence.obj).

3. **Quantization** to 1024 bins provides a good balance between precision and file size.

4. **Across all models**, reconstruction MSE remains below 10⁻⁴, showing high accuracy.

