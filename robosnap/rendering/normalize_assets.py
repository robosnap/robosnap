"""
Normalize mesh assets to standard coordinate system
Ensures bottom of bounding box is at z=0 and centered at XY origin
"""

import argparse
import numpy as np
from pathlib import Path
import trimesh
from scipy.spatial.transform import Rotation
import shutil


def convert_y_up_to_z_up(mesh):
    """
    Convert Y-up to Z-up coordinate system
    Y-up: Y axis is vertical (SAM3D output)
    Z-up: Z axis is vertical (Trimesh/Blender/MesaTask standard)

    Transformation: [X, Y, Z] -> [X, -Z, Y]
    - Old X (width) -> New X (width)
    - Old Y (height, up) -> New Z (height, up)
    - Old Z (depth, forward) -> New Y (depth, backward = -forward)

    This is a -90° rotation around X axis (clockwise when looking from +X)
    """
    transform = np.array([
        [1,  0,  0,  0],
        [0,  0, -1,  0],
        [0,  1,  0,  0],
        [0,  0,  0,  1]
    ], dtype=np.float64)

    mesh.apply_transform(transform)
    return mesh


def detect_coordinate_system(mesh):
    """
    Detect if mesh is Y-up or Z-up

    For SAM3D assets: ALL are Y-up (Y axis is vertical)
    Detection: check if bottom is on Y=0 plane or Z=0 plane

    Returns:
        str: 'Y-up' or 'Z-up'
    """
    bounds = mesh.bounds
    bbox_min, bbox_max = bounds[0], bounds[1]
    bbox_size = bbox_max - bbox_min
    centroid = mesh.centroid

    # Key insight: Y-up means object sits on Y=0 plane (Y_min ≈ 0, centroid_Y > 0)
    # Z-up means object sits on Z=0 plane (Z_min ≈ 0, centroid_Z > 0)

    y_is_vertical = abs(bbox_min[1]) < 0.01 and centroid[1] > 0.01
    z_is_vertical = abs(bbox_min[2]) < 0.01 and centroid[2] > 0.01

    if y_is_vertical and not z_is_vertical:
        return 'Y-up'
    elif z_is_vertical and not y_is_vertical:
        return 'Z-up'
    else:
        # Ambiguous: check which axis has more "vertical" characteristics
        # For SAM3D: if neither clearly bottom-aligned, assume Y-up (SAM3D default)
        # This handles edge cases like very thin objects
        return 'Y-up'  # Default to Y-up for SAM3D assets


def normalize_mesh(mesh, align_orientation=False, auto_convert_y_up=True):
    """
    Normalize mesh to standard coordinate system

    Standard convention:
    - Z-up (Z axis vertical)
    - Bounding box bottom center at (0, 0, 0)
    - Object standing upright (main axis along Z)
    - Optional: align horizontal orientation

    Args:
        mesh: trimesh.Trimesh object
        align_orientation: Whether to align horizontal orientation
        auto_convert_y_up: Auto-detect and convert Y-up to Z-up

    Returns:
        trimesh.Trimesh: Normalized mesh
    """
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    # Step 0: Detect and fix coordinate system if needed
    if auto_convert_y_up:
        coord_system = detect_coordinate_system(mesh)
        if coord_system == 'Y-up':
            print(f"  Detected Y-up, converting to Z-up...")
            mesh = convert_y_up_to_z_up(mesh)
        elif coord_system == 'ambiguous':
            print(f"  Ambiguous orientation (flat object?)")

    # Get bounds after potential coordinate conversion
    bounds = mesh.bounds  # [[xmin, ymin, zmin], [xmax, ymax, zmax]]
    bbox_min, bbox_max = bounds[0], bounds[1]

    # Compute bottom center (center of XY at minimum Z)
    bottom_center = np.array([
        (bbox_min[0] + bbox_max[0]) / 2,
        (bbox_min[1] + bbox_max[1]) / 2,
        bbox_min[2]
    ])

    # Step 1: Translate so bottom center is at origin
    translation = -bottom_center
    mesh.apply_translation(translation)

    # Step 2: Optional orientation alignment
    if align_orientation:
        # Find principal axes using PCA
        centered_verts = mesh.vertices - mesh.vertices.mean(axis=0)
        cov = np.cov(centered_verts.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # Sort by eigenvalue (largest first)
        idx = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, idx]

        # Ensure right-handed coordinate system
        if np.linalg.det(eigenvectors) < 0:
            eigenvectors[:, 2] *= -1

        # Build rotation matrix to align principal axes to XYZ
        # Main axis (largest variance) -> Z axis
        # Second axis -> X axis
        # Third axis -> Y axis
        target_axes = np.eye(3)
        target_axes[:, 2] = eigenvectors[:, 0]  # Main axis to Z
        target_axes[:, 0] = eigenvectors[:, 1]  # Second to X
        target_axes[:, 1] = eigenvectors[:, 2]  # Third to Y

        # Apply rotation
        rotation_matrix = np.eye(4)
        rotation_matrix[:3, :3] = target_axes
        mesh.apply_transform(rotation_matrix)

        # Re-center after rotation
        bounds = mesh.bounds
        bbox_min = bounds[0]
        bottom_center_xy = np.array([
            (bounds[0][0] + bounds[1][0]) / 2,
            (bounds[0][1] + bounds[1][1]) / 2,
            bbox_min[2]
        ])
        mesh.apply_translation(-bottom_center_xy)

    print(f"  Normalized: bottom at z=0, centered at XY origin")
    return mesh


def process_asset_file(file_path, align_orientation=False, backup=True):
    """
    Process a single asset file

    Args:
        file_path: Path to GLB/PLY file
        align_orientation: Whether to align orientation
        backup: Whether to create backup before overwriting

    Returns:
        bool: Success
    """
    file_path = Path(file_path)

    if not file_path.exists():
        print(f"[ERROR] File not found: {file_path}")
        return False

    print(f"[INFO] Processing: {file_path.name}")

    try:
        # Load mesh
        mesh = trimesh.load(str(file_path), force='mesh')
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)

        # Check coordinate system FIRST (before checking z_min)
        coord_system = detect_coordinate_system(mesh)
        bounds = mesh.bounds
        z_min = bounds[0][2]
        y_min = bounds[0][1]

        # If Y-up, need to convert even if z_min is 0
        if coord_system == 'Y-up':
            print(f"  Detected Y-up (y_min={y_min:.4f}), converting...")
        elif abs(z_min) < 0.001:
            # Already Z-up and normalized
            print(f"  Already Z-up normalized (z_min={z_min:.6f})")
            return True
        else:
            print(f"  Z-up but not normalized (z_min={z_min:.4f})")

        # Normalize (will convert Y-up to Z-up if needed)
        normalized_mesh = normalize_mesh(mesh, align_orientation=align_orientation, auto_convert_y_up=True)

        # Backup original file
        if backup:
            backup_path = file_path.with_suffix(file_path.suffix + '.bak')
            if not backup_path.exists():
                shutil.copy2(file_path, backup_path)
                print(f"  Backup saved: {backup_path.name}")

        # Save normalized mesh (overwrite original)
        normalized_mesh.export(str(file_path))

        # Verify
        verify_mesh = trimesh.load(str(file_path), force='mesh')
        if isinstance(verify_mesh, trimesh.Scene):
            verify_mesh = verify_mesh.dump(concatenate=True)
        new_bounds = verify_mesh.bounds
        print(f"  New bounds: z_min={new_bounds[0][2]:.4f}, z_max={new_bounds[1][2]:.4f}")
        print(f"  ✓ Saved: {file_path.name}")

        return True

    except Exception as e:
        print(f"  [ERROR] Failed to process: {e}")
        import traceback
        traceback.print_exc()
        return False


def normalize_directory(asset_dir, pattern="*.glb", align_orientation=False, backup=True):
    """
    Normalize all assets in a directory

    Args:
        asset_dir: Directory containing assets
        pattern: File pattern to match
        align_orientation: Whether to align orientation
        backup: Whether to create backups
    """
    asset_dir = Path(asset_dir)

    if not asset_dir.exists():
        print(f"[ERROR] Directory not found: {asset_dir}")
        return

    # Find all matching files
    files = sorted(asset_dir.glob(pattern))

    if not files:
        print(f"[WARN] No files matching '{pattern}' found in {asset_dir}")
        return

    print("="*60)
    print(f"Normalizing Assets")
    print("="*60)
    print(f"Directory: {asset_dir}")
    print(f"Pattern: {pattern}")
    print(f"Align orientation: {align_orientation}")
    print(f"Backup originals: {backup}")
    print(f"Found {len(files)} files")
    print()

    success_count = 0
    for file_path in files:
        if process_asset_file(file_path, align_orientation=align_orientation, backup=backup):
            success_count += 1
        print()

    print("="*60)
    print(f"✓ Normalized {success_count}/{len(files)} assets")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Normalize mesh assets coordinate system")
    parser.add_argument("--asset_dir", type=str, required=True,
                       help="Directory containing GLB/PLY assets")
    parser.add_argument("--pattern", type=str, default="*.glb",
                       help="File pattern to match (default: *.glb)")
    parser.add_argument("--align_orientation", action="store_true",
                       help="Align object orientation using PCA")
    parser.add_argument("--backup", action="store_true", default=True,
                       help="Create .bak backup files (default: True)")
    parser.add_argument("--no_backup", dest="backup", action="store_false",
                       help="Do not create backup files")

    args = parser.parse_args()

    normalize_directory(
        asset_dir=args.asset_dir,
        pattern=args.pattern,
        align_orientation=args.align_orientation,
        backup=args.backup
    )

    return 0


if __name__ == "__main__":
    exit(main())
