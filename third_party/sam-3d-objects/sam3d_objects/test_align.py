#!/usr/bin/env python3
"""
Test the improved align_scene_to_world function.
"""
import sys
import os
from pathlib import Path
import numpy as np
import trimesh

# Add path for imports
_SAM3D_ROOT = Path(__file__).resolve().parents[1]
for _path in (_SAM3D_ROOT / "sam3d_objects", _SAM3D_ROOT / "notebook"):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

# Import the function
from image2glb import align_scene_to_world, load_glb_with_texture

def test_align_scene():
    """Test alignment on the bedside_refrigerator2 scene"""
    
    # Path to the scene
    mask_dir = os.environ.get("SAM3D_TEST_MASK_DIR", "notebook/images/example_video/bedside_refrigerator2")
    
    # Load all individual GLB files and compose
    all_meshes = []
    
    for index in range(4):
        glb_path = os.path.join(mask_dir, f"{index}.glb")
        pose_path = os.path.join(mask_dir, f"{index}_pose.json")
        
        if not os.path.exists(glb_path):
            print(f"[WARN] {glb_path} not found, skip.")
            continue
        
        # Load pose
        import json
        with open(pose_path, 'r') as f:
            pose_data = json.load(f)
            quat = np.array(pose_data['rotation'])
            trans = np.array(pose_data['translation'])
            scale = np.array(pose_data['scale'])
        
        # Load mesh
        mesh = load_glb_with_texture(glb_path)
        
        # Apply pose (same as in make_scene_from_glb)
        from pytorch3d.transforms import quaternion_to_matrix
        from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform
        import torch
        
        mesh = mesh.copy()
        V = mesh.vertices.astype(np.float64)
        
        # Coordinate transform
        _R_ZUP_TO_YUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
        _R_YUP_TO_ZUP = _R_ZUP_TO_YUP.T
        V = V @ _R_YUP_TO_ZUP.T
        
        V_t = torch.from_numpy(V).float().unsqueeze(0)
        quat_t = torch.from_numpy(quat.flatten()).float().unsqueeze(0)
        trans_t = torch.from_numpy(trans.flatten()).float().unsqueeze(0)
        scale_t = torch.from_numpy(scale.flatten()).float().unsqueeze(0)
        
        R_mat = quaternion_to_matrix(quat_t)
        l2c_transform = compose_transform(
            scale=scale_t,
            rotation=R_mat,
            translation=trans_t
        )
        
        V_transformed = l2c_transform.transform_points(V_t)
        V_final = V_transformed.squeeze(0).numpy()
        V_final = V_final @ _R_ZUP_TO_YUP.T
        
        mesh.vertices = V_final
        all_meshes.append(mesh)
        
        print(f"Loaded mesh {index}: {len(mesh.vertices)} vertices")
    
    # Compose scene
    scene = trimesh.Scene()
    for i, mesh in enumerate(all_meshes):
        scene.add_geometry(mesh, node_name=f"object_{i}")
    
    print(f"\n=== BEFORE ALIGNMENT ===")
    print(f"Scene bounds: {scene.bounds}")
    print(f"Y range: [{scene.bounds[0, 1]:.3f}, {scene.bounds[1, 1]:.3f}]")
    
    # Show vertex distribution
    all_verts = np.vstack([m.vertices for m in all_meshes])
    print(f"Total vertices: {len(all_verts)}")
    print(f"Y min: {all_verts[:, 1].min():.3f}")
    print(f"Y max: {all_verts[:, 1].max():.3f}")
    
    # Align scene
    print(f"\n=== RUNNING ALIGNMENT ===")
    scene = align_scene_to_world(scene, debug=True)
    
    print(f"\n=== AFTER ALIGNMENT ===")
    print(f"Scene bounds: {scene.bounds}")
    print(f"Y range: [{scene.bounds[0, 1]:.3f}, {scene.bounds[1, 1]:.3f}]")
    
    # Save aligned scene
    output_path = os.path.join(mask_dir, "scene_composed_aligned.glb")
    scene.export(output_path)
    print(f"\nSaved aligned scene to: {output_path}")

if __name__ == "__main__":
    test_align_scene()
