"""
Place 3D assets according to MesaTask layout JSON
Assumes assets are normalized (bottom at z=0, centered at XY origin)
Compose into single scene GLB
"""

import json
import argparse
import numpy as np
from pathlib import Path
import trimesh
from scipy.spatial.transform import Rotation


def load_layout(layout_path):
    """Load MesaTask layout JSON"""
    with open(layout_path, 'r') as f:
        return json.load(f)


def find_asset_for_object(obj_index, asset_dir):
    """
    Find GLB or PLY file for object by index

    Args:
        obj_index: Object index (0, 1, 2, ...)
        asset_dir: Directory containing asset files

    Returns:
        Path or None: Path to asset file
    """
    asset_dir = Path(asset_dir)

    # Try GLB first, then PLY
    glb_path = asset_dir / f"{obj_index}.glb"
    ply_path = asset_dir / f"{obj_index}.ply"

    if glb_path.exists():
        return glb_path
    elif ply_path.exists():
        return ply_path
    else:
        return None


def scale_mesh_to_target(mesh, target_size_cm):
    """
    Scale normalized mesh to match target size
    Uses largest face area for robust scaling

    IMPORTANT ASSUMPTIONS:
    - Mesh is already normalized (bottom at z=0, centered at XY origin, Z-up)
    - Mesh units are meters (trimesh default)
    - target_size_cm is [width, depth, height] in centimeters

    Args:
        mesh: trimesh.Trimesh (normalized)
        target_size_cm: np.array([width, depth, height]) in cm

    Returns:
        trimesh.Trimesh: Scaled mesh
    """
    # Get current size in mesh units (should be meters after normalization)
    current_bbox = mesh.bounds
    current_size_m = current_bbox[1] - current_bbox[0]  # [width, depth, height] in meters

    # Convert target from cm to meters
    target_size_m = target_size_cm * 0.01

    # Calculate areas of all three faces
    # XY face (footprint), XZ face (front), YZ face (side)
    current_areas = [
        current_size_m[0] * current_size_m[1],  # XY (footprint)
        current_size_m[0] * current_size_m[2],  # XZ (front)
        current_size_m[1] * current_size_m[2]   # YZ (side)
    ]
    target_areas = [
        target_size_m[0] * target_size_m[1],  # XY
        target_size_m[0] * target_size_m[2],  # XZ
        target_size_m[1] * target_size_m[2]   # YZ
    ]

    # Find which face is largest in target
    max_target_idx = np.argmax(target_areas)

    # Scale based on that face's area (sqrt because area scales as scale^2)
    current_max_area = current_areas[max_target_idx]
    target_max_area = target_areas[max_target_idx]
    scale_factor = np.sqrt(target_max_area / (current_max_area + 1e-9))

    face_names = ['XY-footprint', 'XZ-front', 'YZ-side']
    print(f"    Scaling by {face_names[max_target_idx]} area: {scale_factor:.2f}")

    mesh.apply_scale(scale_factor)

    # Verify new size
    new_bbox = mesh.bounds
    new_size_m = new_bbox[1] - new_bbox[0]
    current_str = f"[{current_size_m[0]*100:.1f}, {current_size_m[1]*100:.1f}, {current_size_m[2]*100:.1f}]"
    new_str = f"[{new_size_m[0]*100:.1f}, {new_size_m[1]*100:.1f}, {new_size_m[2]*100:.1f}]"
    target_str = f"[{target_size_cm[0]:.1f}, {target_size_cm[1]:.1f}, {target_size_cm[2]:.1f}]"
    print(f"    Scaled: {current_str}cm -> {new_str}cm (target: {target_str}cm)")

    return mesh


def create_table_mesh(table_width=100, table_depth=50, table_height=2, table_z=-1):
    """
    Create a simple table surface mesh

    Args:
        table_width: Width in cm
        table_depth: Depth in cm
        table_height: Thickness in cm
        table_z: Z position in cm (usually negative to place below objects)

    Returns:
        trimesh.Trimesh: Table mesh (in meters)
    """
    # Create box mesh (in meters)
    w = table_width * 0.01
    d = table_depth * 0.01
    h = table_height * 0.01

    table = trimesh.creation.box(extents=[w, d, h])

    # Position at correct height
    z_offset = (table_z + table_height / 2) * 0.01  # Center of box in meters
    table.apply_translation([0, 0, z_offset])

    return table


def place_assets_in_scene(layout_data, asset_dir, add_table=True):
    """
    Place all assets according to layout

    COORDINATE SYSTEM:
    - Assets after normalization: bottom center at (0,0,0), units in meters
    - MesaTask layout: positions and sizes in centimeters
    - Final scene: units in meters, table centered at XY origin

    Args:
        layout_data: MesaTask layout JSON
        asset_dir: Directory containing normalized GLB/PLY assets
        add_table: Whether to add table surface

    Returns:
        trimesh.Scene: Combined scene
    """
    scene = trimesh.Scene()

    objects = layout_data.get("objects", [])
    placement_zone = layout_data.get("item_placement_zone", [0, 100, 0, 50])

    # Extract table dimensions from placement zone [x_min, x_max, y_min, y_max] in cm
    table_width = placement_zone[1] - placement_zone[0]
    table_depth = placement_zone[3] - placement_zone[2]

    # Add table surface if requested
    if add_table:
        table_mesh = create_table_mesh(
            table_width=table_width,
            table_depth=table_depth,
            table_height=2,
            table_z=-1
        )
        scene.add_geometry(table_mesh, node_name="table_surface")
        print(f"[INFO] Added table surface: {table_width:.0f}cm × {table_depth:.0f}cm")

    print(f"[INFO] Table centered at origin, width={table_width:.0f}cm, depth={table_depth:.0f}cm")
    print()

    # Place each object
    for idx, obj in enumerate(objects):
        asset_path = find_asset_for_object(idx, asset_dir)

        if asset_path is None:
            print(f"[WARN] Asset not found for object {idx}: {obj.get('instance', 'unknown')}")
            continue

        obj_name = obj.get("instance", f"object_{idx}")
        print(f"[INFO] Placing asset {idx}: {asset_path.name} ({obj_name})")

        # Load mesh
        try:
            mesh = trimesh.load(str(asset_path), force='mesh')
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
        except Exception as e:
            print(f"  [ERROR] Failed to load {asset_path}: {e}")
            continue

        # Check if mesh is normalized (bottom close to z=0)
        initial_bounds = mesh.bounds
        z_min_initial = initial_bounds[0][2]
        if abs(z_min_initial) > 0.01:  # More than 1cm off
            print(f"  [WARN] Mesh not normalized! z_min={z_min_initial:.4f}m")
            print(f"  [WARN] Run: python -m robosnap.rendering.normalize_assets --asset_dir {asset_dir}")

        # Get target size and position from layout (all in cm)
        target_size = np.array(obj.get("size", [10, 10, 10]))  # [width, depth, height] in cm
        target_pos = np.array(obj.get("position", [0, 0, 0]))  # [x, y, z] in cm
        z_rotation = obj.get("z_rotation", 0.0)  # Rotation around z-axis in radians

        # Scale mesh to target size
        mesh = scale_mesh_to_target(mesh, target_size)

        # At this point, mesh bottom is at z=0, centered at XY origin
        # DO NOT subtract centroid - would break normalization!

        # Apply rotation around z-axis BEFORE translation
        # Rotation is around the bottom center point
        if z_rotation != 0:
            rotation = Rotation.from_euler('z', z_rotation, degrees=False)
            transform = np.eye(4, dtype=np.float64)
            transform[:3, :3] = rotation.as_matrix()
            mesh.apply_transform(transform)
            print(f"    Rotated: {np.degrees(z_rotation):.1f}°")

        # Convert MesaTask position to scene position (meters)
        # MesaTask position: [x, y, z] where:
        #   - x, y are horizontal positions on table (cm from table corner)
        #   - z is height of OBJECT CENTER above table surface
        #
        # Since mesh is normalized (bottom at local z=0), we need to:
        # 1. Place at XY position
        # 2. Lift by (target_pos[2] - target_size[2]/2) to align center with layout
        #    But since mesh bottom is at 0, we actually place bottom at: z - height/2
        #    Which simplifies to: place mesh bottom at 0, then center will be at height/2
        #
        # Actually: if position.z is center height, and mesh bottom is at 0,
        # we need to translate to: position.z - (mesh_height / 2)
        # But after scaling, mesh height = target_size[2]
        # So bottom should be at: target_pos[2] - target_size[2]/2

        # Get scaled mesh height
        scaled_bounds = mesh.bounds
        scaled_height = scaled_bounds[1][2] - scaled_bounds[0][2]  # in meters

        # Calculate bottom position: center_z - height/2
        # position.z is center height in cm
        bottom_z_cm = target_pos[2] - (target_size[2] / 2)

        scene_pos = np.array([
            (target_pos[0] - table_width / 2) * 0.01,   # x: centered on table (m)
            (target_pos[1] - table_depth / 2) * 0.01,   # y: centered on table (m)
            bottom_z_cm * 0.01                          # z: bottom height (m)
        ])

        mesh.apply_translation(scene_pos)
        print(f"    Position: layout_center=[{target_pos[0]:.1f}, {target_pos[1]:.1f}, {target_pos[2]:.1f}]cm")
        print(f"              bottom_z={bottom_z_cm:.1f}cm, scene=[{scene_pos[0]:.3f}, {scene_pos[1]:.3f}, {scene_pos[2]:.3f}]m")
        print(f"    Position: layout=[{target_pos[0]:.1f}, {target_pos[1]:.1f}, {target_pos[2]:.1f}]cm")
        print(f"              scene=[{scene_pos[0]:.3f}, {scene_pos[1]:.3f}, {scene_pos[2]:.3f}]m")

        # Verify final position
        final_bounds = mesh.bounds
        final_z_min = final_bounds[0][2]
        print(f"    Final: z_min={final_z_min:.4f}m, z_max={final_bounds[1][2]:.4f}m")

        # Add to scene
        scene.add_geometry(mesh, node_name=obj_name)
        print()

    # Apply global scene rotation to fix coordinate system for rendering
    # Rotate entire scene -90° around X axis to align with Blender's expected orientation
    print("[INFO] Applying global scene rotation (X-axis -90°)...")
    global_rotation = np.array([
        [1,  0,  0,  0],
        [0,  0,  1,  0],
        [0, -1,  0,  0],
        [0,  0,  0,  1]
    ], dtype=np.float64)

    scene.apply_transform(global_rotation)
    print("[INFO] Scene rotation applied")

    return scene


def main():
    parser = argparse.ArgumentParser(description="Place assets according to layout")
    parser.add_argument("--layout", type=str, required=True,
                       help="Path to layout JSON file")
    parser.add_argument("--asset_dir", type=str, required=True,
                       help="Directory containing normalized GLB/PLY assets")
    parser.add_argument("--output", type=str, required=True,
                       help="Output scene GLB path")
    parser.add_argument("--add_table", action="store_true",
                       help="Add table surface mesh")

    args = parser.parse_args()

    print("="*60)
    print("Placing Assets in Scene")
    print("="*60)
    print(f"Layout: {args.layout}")
    print(f"Assets: {args.asset_dir}")
    print(f"Output: {args.output}")
    print(f"Add table: {args.add_table}")
    print()

    # Load layout
    layout_data = load_layout(args.layout)

    # Place assets
    scene = place_assets_in_scene(layout_data, args.asset_dir, add_table=args.add_table)

    # Export scene
    scene.export(args.output)

    print("="*60)
    print(f"✓ Scene exported to: {args.output}")
    print(f"✓ Total geometries: {len(scene.geometry)}")
    print("="*60)

    return 0


if __name__ == "__main__":
    exit(main())
