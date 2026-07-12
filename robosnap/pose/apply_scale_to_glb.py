#!/usr/bin/env python3
"""
Apply scale from pose JSON to GLB files.

This script reads GLB files and their corresponding pose JSON files,
applies the scale factor to the mesh vertices, and saves the result.

Usage:
    python scripts/apply_scale_to_glb.py \
        --input_dir /path/to/desk1 \
        --output_dir /path/to/desk1 \
        --start 0 \
        --end 4
"""

import os
import sys
import argparse
import json
import numpy as np
import trimesh


def load_glb_with_texture(glb_path):
    """
    Load GLB file and preserve texture/material information.
    Returns a trimesh that can be used for composition.
    """
    # Load as Scene to preserve materials
    scene = trimesh.load(glb_path, force='scene')

    # If it's a Scene, extract all geometries and apply node transforms
    if isinstance(scene, trimesh.Scene):
        meshes = []
        for node_name in scene.graph.nodes_geometry:
            geom_name = scene.graph[node_name][1]
            m = scene.geometry[geom_name].copy()
            # Apply node transform
            M_node = scene.graph.get(node_name)[0]
            if M_node is not None:
                m.apply_transform(M_node)
            meshes.append(m)

        if not meshes:
            raise RuntimeError(f"No geometries in {glb_path}")

        # Concatenate all meshes into one
        combined = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
        return combined

    return scene


def apply_scale_to_mesh(mesh, scale):
    """
    Apply scale factor to mesh vertices.

    Args:
        mesh: trimesh.Trimesh object
        scale: scale factor (can be scalar or [sx, sy, sz])

    Returns:
        mesh with scaled vertices
    """
    mesh = mesh.copy()

    # Convert scale to array if scalar
    if np.isscalar(scale):
        scale = np.array([scale, scale, scale])
    else:
        scale = np.array(scale).flatten()

    # Apply scale to vertices
    mesh.vertices = mesh.vertices * scale

    return mesh


def process_single_glb(glb_path, pose_path, output_path):
    """
    Process a single GLB file with its pose.
    """
    # Load mesh
    print(f"Loading GLB: {glb_path}")
    mesh = load_glb_with_texture(glb_path)
    print(f"  Original vertices: {len(mesh.vertices)}")
    print(f"  Original bounds: {mesh.bounds}")

    # Load pose
    print(f"Loading pose: {pose_path}")
    with open(pose_path, 'r') as f:
        pose_data = json.load(f)

    scale = np.array(pose_data['scale']).flatten()
    print(f"  Scale: {scale}")

    # Apply scale
    mesh_scaled = apply_scale_to_mesh(mesh, scale)
    print(f"  Scaled bounds: {mesh_scaled.bounds}")

    # Save
    print(f"Saving: {output_path}")
    mesh_scaled.export(output_path)
    print(f"  Done!")

    return mesh_scaled


def main(args):
    # Check if input_dir has trailing slash issue
    input_dir = args.input_dir.rstrip('/')

    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)

    for i in range(args.start, args.end + 1):
        glb_path = os.path.join(input_dir, f"{i}.glb")
        pose_path = os.path.join(input_dir, f"{i}_pose.json")
        output_path = os.path.join(args.output_dir, f"{i}_scaled.glb")

        if not os.path.exists(glb_path):
            print(f"[WARN] GLB not found: {glb_path}, skip.")
            continue

        if not os.path.exists(pose_path):
            print(f"[WARN] Pose not found: {pose_path}, skip.")
            continue

        print(f"\n=== Processing {i}.glb ===")
        process_single_glb(glb_path, pose_path, output_path)

    print("\nAll done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Apply scale from pose JSON to GLB files")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing input GLB files and pose JSONs",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save scaled GLB files",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index (inclusive)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=4,
        help="End index (inclusive)",
    )
    args = parser.parse_args()
    main(args)
