# Copyright (c) Meta Platforms, Inc. and affiliates.
import sys
import os
import argparse
import json
from pathlib import Path
import numpy as np

# import inference code
_SAM3D_ROOT = Path(__file__).resolve().parents[1]
_NOTEBOOK_DIR = _SAM3D_ROOT / "notebook"
for _path in (_SAM3D_ROOT, _NOTEBOOK_DIR):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)
from inference import Inference, load_image, load_single_mask
from sam3d_objects.model.backbone.tdfy_dit.utils import postprocessing_utils


# -----------------------------
# Proxy (optional)
# -----------------------------
def setup_proxy():
    https_proxy = (
        os.environ.get("SAM3D_HTTPS_PROXY")
        or os.environ.get("HTTPS_PROXY")
        or os.environ.get("https_proxy")
    )
    if not https_proxy:
        return
    os.environ["https_proxy"] = https_proxy
    os.environ["http_proxy"] = https_proxy
    os.environ["HTTPS_PROXY"] = https_proxy
    os.environ["HTTP_PROXY"] = https_proxy

    no_proxy = "localhost,127.0.0.1,::1,.pjlab.org.cn"
    os.environ["no_proxy"] = no_proxy
    os.environ["NO_PROXY"] = no_proxy


# -----------------------------
# Coordinate System Transformation (CRITICAL FIX for Issue #56)
# -----------------------------
# SAM-3D Objects uses different coordinate systems for different outputs:
#
# 1. GLB mesh: Exported in Y-up coordinates (standard for glTF/GLB format)
#    - The Z-up → Y-up rotation is applied in postprocessing_utils.to_glb()
#
# 2. Gaussian splats: Stay in Z-up coordinates (no rotation applied in save_ply())
#
# 3. Transformation parameters (rotation, translation, scale):
#    - Computed in original Z-up/PyTorch3D camera frame
#
# PROBLEM: When applying transformation parameters directly to GLB mesh vertices,
#          the mesh vertices have already been rotated to Y-up, but the transforms
#          haven't been applied in that frame.
#
# SOLUTION: Convert GLB vertices back to Z-up → Apply transforms → Convert back to Y-up
#           This is the same approach as make_scene_untextured_mesh() from alexsax.
#
# Reference: https://github.com/facebookresearch/sam-3d-objects/issues/56

# Rotation matrix: Z-up to Y-up (rotate -90 degrees around X axis)
# This transforms coordinates from PyTorch3D (Z-up) to glTF (Y-up)
_R_ZUP_TO_YUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
# Rotation matrix: Y-up to Z-up (rotate +90 degrees around X axis) - inverse of above
_R_YUP_TO_ZUP = _R_ZUP_TO_YUP.T


def load_glb_with_texture(glb_path):
    """
    Load GLB file and preserve texture/material information.
    Returns a trimesh that can be used for composition.
    """
    import trimesh
    
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


# -----------------------------
# Scene Compose (Mesh) - Coordinate-corrected version
# -----------------------------
def make_scene_from_glb(mask_dir, num_masks):
    """
    Load generated GLB files and compose them into a scene.
    
    This function handles the coordinate system mismatch automatically:
    - GLB mesh vertices: Y-up (exported from to_glb)
    - Transformation parameters: Z-up (PyTorch3D camera frame)
    
    The fix: Convert GLB vertices Y-up → Z-up → Apply pose → Convert back to Y-up
    
    Args:
        mask_dir: Directory containing mask GLB files
        num_masks: Number of masks to compose
    
    Returns:
        scene_mesh: merged trimesh scene
    """
    import trimesh
    import torch
    from pytorch3d.transforms import quaternion_to_matrix
    from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform
    
    def apply_pose_to_mesh_pytorch3d(mesh, quat, trans, scale):
        """
        Apply pose (R, t, s) to mesh vertices.
        
        Coordinate frame conversion:
          1. Convert GLB vertices Y-up → Z-up (inverse of to_glb conversion)
          2. Apply pose transformation in Z-up frame
          3. Convert back Z-up → Y-up for final GLB output
        """
        mesh = mesh.copy()
        V = mesh.vertices.astype(np.float64)
        
        # ================================================================
        # STEP 1: Convert GLB vertices from Y-up to Z-up
        V = V @ _R_YUP_TO_ZUP
        
        # Convert to torch
        V_t = torch.from_numpy(V).float().unsqueeze(0)  # (1, N, 3)
        
        # Convert pose to torch
        quat_t = torch.from_numpy(quat.flatten()).float().unsqueeze(0)  # (1, 4)
        trans_t = torch.from_numpy(trans.flatten()).float().unsqueeze(0)  # (1, 3)
        scale_t = torch.from_numpy(scale.flatten()).float().unsqueeze(0)  # (1, 3)
        
        # Convert quaternion to rotation matrix
        R_mat = quaternion_to_matrix(quat_t)  # (1, 3, 3)
        
        # Apply transformation in Z-up frame
        l2c_transform = compose_transform(
            scale=scale_t,
            rotation=R_mat,
            translation=trans_t
        )
        
        # Transform points
        V_transformed = l2c_transform.transform_points(V_t)  # (1, N, 3)
        
        # STEP 2: Convert transformed vertices from Z-up back to Y-up
        V_final = V_transformed.squeeze(0).numpy()
        V_final = V_final @ _R_ZUP_TO_YUP
        
        mesh.vertices = V_final
        return mesh
    
    all_meshes = []
    pose_info = []
    
    for index in range(num_masks):
        glb_path = os.path.join(mask_dir, f"{index}.glb")
        pose_path = os.path.join(mask_dir, f"{index}_pose.json")
        
        if not os.path.exists(glb_path):
            print(f"[WARN] {glb_path} not found, skip.")
            continue
        
        # Load pose info if exists
        if os.path.exists(pose_path):
            with open(pose_path, 'r') as f:
                pose_data = json.load(f)
                quat = np.array(pose_data['rotation'])
                trans = np.array(pose_data['translation'])
                scale = np.array(pose_data['scale'])
        else:
            # Try to load from inference output if available
            # This requires the output to be saved, so we'll skip if not found
            print(f"[WARN] {pose_path} not found, skip.")
            continue
        
        # Load mesh with texture preserved
        mesh = load_glb_with_texture(glb_path)
        
        # Apply pose to mesh (coordinate-corrected)
        mesh_scene = apply_pose_to_mesh_pytorch3d(mesh, quat, trans, scale)
        all_meshes.append(mesh_scene)
        
        pose_info.append({
            'index': index,
            'glb': glb_path,
            'rotation': quat.tolist(),
            'translation': trans.tolist(),
            'scale': scale.tolist(),
        })
    
    if not all_meshes:
        print("[WARN] No meshes to compose.")
        return None
    
    # Merge all meshes using Scene to preserve materials
    # Create a scene and add each mesh as a separate node
    scene = trimesh.Scene()
    for i, mesh in enumerate(all_meshes):
        scene.add_geometry(mesh, node_name=f"object_{i}")

    print(f"[INFO] Composed {len(all_meshes)} meshes into scene.")

    # # ============================================================
    # # GLOBAL WORLD FIX (camera space → clean world space)
    # # ============================================================

    # # 1️⃣ Global rotate (X +90°)
    # R_global = np.array([
    #     [1, 0, 0],
    #     [0, 0, -1],
    #     [0, 1, 0]
    # ], dtype=np.float64)

    # T_global = np.eye(4)
    # T_global[:3, :3] = R_global
    # scene.apply_transform(T_global)

    # # FLIP_Z = np.diag([1, 1, -1, 1])
    # # scene.apply_transform(FLIP_Z)

    # bbox = scene.bounds
    # center = bbox.mean(axis=0)

    # T_center = np.eye(4)
    # T_center[:3, 3] = -center
    # scene.apply_transform(T_center)
    # # Rz_180 = np.array([
    # # [-1,  0,  0, 0],
    # # [ 0, -1,  0, 0],
    # # [ 0,  0,  1, 0],
    # # [ 0,  0,  0, 1]
    # # ])

    # # scene.apply_transform(Rz_180)

    # print("[INFO] Applied global rotation, flip and centering.")
    # print(np.linalg.det(scene.graph.get(frame_to=None)[0][:3,:3]))

    return scene, pose_info     


def main(args):
    setup_proxy()
        
    # -----------------------------
    # load model once
    # -----------------------------
    print(f"[ℹ️] Loading pipeline config: {args.config}")
    inference = Inference(args.config, compile=False)

    print(f"[ℹ️] Loading image: {args.image_dir}")
    image = load_image(args.image_dir)
    
    # Store outputs for compose
    outputs_for_compose = []
    
    for index in range(args.num_masks):
        glb_path = os.path.join(args.mask_dir, f"{index}.glb")
        if args.skip_existing and os.path.exists(glb_path):
            print(f"[INFO] {glb_path} exists, skip.")
            continue
        
        
        print(f"\n=== Processing mask {index} ===")
        mask = load_single_mask(args.mask_dir, index=index)
        output = inference(image, mask, seed=args.seed,)
        print("output dict keys: ", output.keys())
        
        # Save pose info
        pose_path = os.path.join(args.mask_dir, f"{index}_pose.json")
        pose_data = {
            'rotation': output['rotation'].cpu().numpy().tolist(),
            'translation': output['translation'].cpu().numpy().tolist(),
            'scale': output['scale'].cpu().numpy().tolist(),
        }
        with open(pose_path, 'w') as f:
            json.dump(pose_data, f, indent=2)
        print(f"Saved pose: {pose_path}")
        
        # Store for compose
        outputs_for_compose.append({
            'index': index,
            'output': output,
        })
        
        if args.scale_only:
            print("[INFO] scale_only mode → skip PLY / GLB generation")
            continue
        
        ply_path = os.path.join(args.mask_dir, f"{index}.ply")
        output["gs"].save_ply(ply_path)
        print(f"Saved PLY: {ply_path}")

        mesh = postprocessing_utils.to_glb(
                app_rep=output["gs"],
                mesh=output["mesh"][0],
                fill_holes=False,
                texture_size=1024,
                with_texture_baking=True,   
                # use_vertex_color=True,   
            )

        mesh.export(glb_path)
        print(f"Saved GLB: {glb_path}")

    print("\nAll masks processed.")
    
    # -----------------------------
    # Compose scene from GLB files
    # -----------------------------
    if args.compose_scene:
        print("\n=== Composing scene from GLB files ===")
        scene, pose_info = make_scene_from_glb(args.mask_dir, args.num_masks)
        
        if scene is not None:
            scene_path = os.path.join(args.mask_dir, "scene_composed.glb")
            scene.export(scene_path)
            print(f"Saved scene GLB: {scene_path}")
            
            # Save pose info
            pose_info_path = os.path.join(args.mask_dir, "scene_composed_poses.json")
            with open(pose_info_path, 'w') as f:
                json.dump(pose_info, f, indent=2)
            print(f"Saved pose info: {pose_info_path}")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser("SAM-3D-Objects mask → GLB")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to pipeline.yaml",
    )
    parser.add_argument(
        "--image_dir",
        "--image",
        dest="image_dir",
        type=str,
        required=True,
        help="Path to input image",
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        required=True,
        help="Directory containing mask PNGs",
    )
    parser.add_argument(
        "--num_masks",
        type=int,
        required=True,
        help="Number of masks to process (0..num_masks-1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip processing if output GLB already exists",
    )
    parser.add_argument(
        "--compose_scene",
        action="store_true",
        help="Compose all GLB files into a scene after generation",
    )
    parser.add_argument(
        "--scale_only",
        action="store_true",
        help="Run inference but only save scale/pose json, skip mesh/glb generation",
    )
    args = parser.parse_args()
    main(args)
