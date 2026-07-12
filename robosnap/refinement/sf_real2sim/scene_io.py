from __future__ import annotations

import glob
import os

import numpy as np
import trimesh


def load_scene_data(results_dir: str, input_pose_name: str) -> tuple[dict, dict, dict]:
    meshes = {}
    mesh_paths = {}
    poses = {}

    obj_dirs = sorted(glob.glob(os.path.join(results_dir, "obj_*")))
    if not obj_dirs:
        raise FileNotFoundError(f"No obj_* directories found in {results_dir}")

    for obj_dir in obj_dirs:
        obj_id = os.path.basename(obj_dir)
        pose_path = os.path.join(obj_dir, f"{input_pose_name}.txt")
        scale_path = os.path.join(obj_dir, "final_scale.txt")
        mesh_path = os.path.join(obj_dir, "mesh_scaled.glb")
        if not os.path.exists(mesh_path):
            mesh_path = os.path.join(obj_dir, "mesh.glb")
        if not os.path.exists(pose_path) or not os.path.exists(mesh_path):
            continue

        mesh = trimesh.load(mesh_path, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        if "scaled" not in os.path.basename(mesh_path) and os.path.exists(scale_path):
            mesh.apply_scale(float(np.loadtxt(scale_path).flatten()[0]))

        pose = np.loadtxt(pose_path).astype(np.float32)
        if pose.shape != (4, 4):
            continue
        meshes[obj_id] = mesh
        mesh_paths[obj_id] = os.path.abspath(mesh_path)
        poses[obj_id] = pose

    print(f"Loaded {len(meshes)} objects from {results_dir}")
    return meshes, mesh_paths, poses


def save_optimized_poses(results_dir: str, optimized_poses: dict, output_pose_name: str) -> None:
    for obj_id, pose in optimized_poses.items():
        out_path = os.path.join(results_dir, obj_id, f"{output_pose_name}.txt")
        np.savetxt(out_path, pose)
        print(f"  Saved {out_path}")
