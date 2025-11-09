#!/usr/bin/env python3
import os
import subprocess

# Configuration 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SCRIPT_PATH = os.path.join(BASE_DIR, "src", "mesh_processing.py")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

def main():
    # Validation
    if not os.path.exists(DATA_DIR):
        print(f"Data folder not found: {DATA_DIR}")
        return

    obj_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".obj")]
    if not obj_files:
        print("No .obj files found in the 'data/' folder.")
        return

    print(f"🔍 Found {len(obj_files)} OBJ files: {', '.join(obj_files)}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Processing Loop 
    for obj_file in obj_files:
        input_path = os.path.join(DATA_DIR, obj_file)
        print(f"Processing {obj_file} ...")
        cmd = [
            "python",
            SCRIPT_PATH,
            "--input", input_path,
            "--output", OUTPUT_DIR,
            "--visualize"
        ]

        try:
            subprocess.run(cmd, check=True)
            print(f"Completed: {obj_file}\n")
        except subprocess.CalledProcessError as e:
            print(f"Error while processing {obj_file}: {e}\n")

    print("All meshes processed successfully!")
    print(f"Results saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
