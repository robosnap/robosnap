"""
SDF grid precomputation, differentiable query, and debug export utilities.

Uses Open3D RaycastingScene for SDF computation and PyTorch grid_sample
for differentiable trilinear interpolation during optimization.
"""

import json
import os
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
import open3d as o3d


def compute_sdf_grid(
    mesh: trimesh.Trimesh,
    resolution: int = 128,
    padding: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute a dense SDF voxel grid for a mesh in its local frame.

    Args:
        mesh: A trimesh mesh object.
        resolution: Grid resolution along each axis.
        padding: Extra padding around the bounding box (in mesh units).

    Returns:
        sdf_grid: Float tensor of shape (1, 1, R, R, R) ready for grid_sample.
        grid_bounds: Float tensor of shape (2, 3) with [min_corner, max_corner].
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)

    bbox_min = vertices.min(axis=0) - padding
    bbox_max = vertices.max(axis=0) + padding

    lin = [np.linspace(bbox_min[i], bbox_max[i], resolution) for i in range(3)]
    gx, gy, gz = np.meshgrid(lin[0], lin[1], lin[2], indexing="ij")
    query_points = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3).astype(np.float32)

    o3d_mesh = o3d.t.geometry.TriangleMesh()
    o3d_mesh.vertex.positions = o3d.core.Tensor(vertices)
    o3d_mesh.triangle.indices = o3d.core.Tensor(faces)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d_mesh)

    sdf_values = scene.compute_signed_distance(
        o3d.core.Tensor(query_points, dtype=o3d.core.float32)
    ).numpy()

    sdf_grid = sdf_values.reshape(resolution, resolution, resolution)
    sdf_grid = torch.from_numpy(sdf_grid).float()
    # grid_sample expects (N, C, D, H, W)
    sdf_grid = sdf_grid.unsqueeze(0).unsqueeze(0)

    grid_bounds = torch.tensor(
        np.stack([bbox_min, bbox_max]), dtype=torch.float32
    )

    return sdf_grid, grid_bounds


def query_sdf(
    sdf_grid: torch.Tensor,
    grid_bounds: torch.Tensor,
    points: torch.Tensor,
) -> torch.Tensor:
    """
    Differentiable SDF query via trilinear interpolation.

    Args:
        sdf_grid: Tensor (1, 1, D, H, W) – the precomputed SDF grid.
        grid_bounds: Tensor (2, 3) – [min_corner, max_corner] of the grid.
        points: Tensor (N, 3) – query points in the object's local frame.

    Returns:
        Tensor (N,) – interpolated SDF values (differentiable w.r.t. points).
    """
    grid_min = grid_bounds[0]  # (3,)
    grid_max = grid_bounds[1]  # (3,)
    grid_size = grid_max - grid_min  # (3,)

    # Normalize points to [-1, 1] for grid_sample
    normalized = 2.0 * (points - grid_min) / grid_size - 1.0  # (N, 3)

    # grid_sample 5D coordinate convention: last dim is (x, y, z) mapping to
    # (W, H, D) of the input tensor.  Our SDF grid was built with meshgrid
    # indexing="ij", so dim-2=X, dim-3=Y, dim-4=Z.  grid_sample's x->W=Z,
    # y->H=Y, z->D=X, so we must reverse the coordinate order.
    normalized = normalized[:, [2, 1, 0]]

    # Reshape as (1, 1, 1, N, 3) grid for a single batch query.
    grid_query = normalized.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, N, 3)

    sampled = F.grid_sample(
        sdf_grid,
        grid_query,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    # sampled shape: (1, 1, 1, 1, N) -> flatten to (N,)
    return sampled.reshape(-1)


def sample_surface_points(
    mesh: trimesh.Trimesh,
    num_points: int = 1024,
    seed: int = 42,
) -> torch.Tensor:
    """
    Uniformly sample points on the surface of a mesh.

    Args:
        mesh: A trimesh mesh object.
        num_points: Number of surface points to sample.
        seed: Random seed for reproducibility.

    Returns:
        Tensor (num_points, 3) – surface point positions in the mesh's local frame.
    """
    points, _ = trimesh.sample.sample_surface(mesh, num_points, seed=seed)
    return torch.from_numpy(np.asarray(points, dtype=np.float32))


def save_sdf_debug_artifacts(
    obj_id: str,
    mesh: trimesh.Trimesh,
    sdf_grid: torch.Tensor,
    sdf_bounds: torch.Tensor,
    surface_pts: torch.Tensor,
    debug_root: Optional[str],
    padding: float,
    save_visualizations: bool = True,
) -> None:
    """Persist SDF tensors, metadata, slice images, and optional isosurface."""
    if not debug_root:
        return

    obj_dir = os.path.join(debug_root, obj_id)
    os.makedirs(obj_dir, exist_ok=True)

    grid_np = sdf_grid.detach().cpu().numpy()[0, 0].astype(np.float32)
    bounds_np = sdf_bounds.detach().cpu().numpy().astype(np.float32)
    surface_np = surface_pts.detach().cpu().numpy().astype(np.float32)

    np.savez_compressed(
        os.path.join(obj_dir, "sdf_grid.npz"),
        sdf_grid=grid_np,
        sdf_bounds=bounds_np,
        surface_points=surface_np,
    )

    metadata = {
        "obj_id": obj_id,
        "resolution": int(grid_np.shape[0]),
        "padding": float(padding),
        "bounds_min": bounds_np[0].tolist(),
        "bounds_max": bounds_np[1].tolist(),
        "mesh_bounds_min": np.asarray(mesh.bounds[0], dtype=float).tolist(),
        "mesh_bounds_max": np.asarray(mesh.bounds[1], dtype=float).tolist(),
        "sdf_min": float(grid_np.min()),
        "sdf_max": float(grid_np.max()),
        "sdf_mean": float(grid_np.mean()),
        "negative_voxel_ratio": float(np.mean(grid_np < 0.0)),
    }
    with open(os.path.join(obj_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved SDF debug artifacts to {obj_dir}")

    if not save_visualizations:
        return

    write_sdf_slice_visualizations(grid_np, obj_dir)
    try_export_sdf_isosurface(grid_np, bounds_np, obj_dir)
    print(f"Saved SDF slice visualizations and isosurface to {obj_dir}")


def render_sdf_slice(slice_2d: np.ndarray, clip_value: float) -> np.ndarray:
    """Render one SDF slice with negative and positive values clearly separated."""
    display = np.flipud(slice_2d.T)
    clipped = np.clip(display, -clip_value, clip_value)

    image = np.full((*display.shape, 3), 255, dtype=np.uint8)

    neg_mask = clipped < 0.0
    pos_mask = clipped > 0.0

    neg_strength = np.zeros_like(clipped, dtype=np.float32)
    pos_strength = np.zeros_like(clipped, dtype=np.float32)
    neg_strength[neg_mask] = np.clip(-clipped[neg_mask] / clip_value, 0.0, 1.0)
    pos_strength[pos_mask] = np.clip(clipped[pos_mask] / clip_value, 0.0, 1.0)

    # BGR colors: negative SDF is blue and fades lighter as it gets more negative;
    # positive SDF is orange/red and fades lighter as it gets more positive.
    image[neg_mask, 0] = 255
    image[neg_mask, 1] = (80 + 175 * neg_strength[neg_mask]).astype(np.uint8)
    image[neg_mask, 2] = (80 + 175 * neg_strength[neg_mask]).astype(np.uint8)

    image[pos_mask, 0] = (40 + 215 * pos_strength[pos_mask]).astype(np.uint8)
    image[pos_mask, 1] = (105 + 150 * pos_strength[pos_mask]).astype(np.uint8)
    image[pos_mask, 2] = 255

    inside_mask = (display < 0.0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(inside_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0, 0, 0), 2)
    cv2.drawContours(image, contours, -1, (255, 255, 255), 1)
    return image


def write_sdf_slice_visualizations(grid_np: np.ndarray, obj_dir: str) -> None:
    '''
    Write SDF slice visualizations to a directory.
    Args:
        grid_np: The SDF grid.
        obj_dir: The directory to write the visualizations to.

    Returns:
        None
    '''
    clip_value = float(np.percentile(np.abs(grid_np), 95))
    if not np.isfinite(clip_value) or clip_value <= 1e-8:
        clip_value = float(np.max(np.abs(grid_np)))
    if not np.isfinite(clip_value) or clip_value <= 1e-8:
        clip_value = 1.0

    axis_specs = [
        ("x", lambda i: grid_np[i, :, :]),
        ("y", lambda i: grid_np[:, i, :]),
        ("z", lambda i: grid_np[:, :, i]),
    ]
    center_idx = grid_np.shape[0] // 2
    center_tiles = []

    for axis_name, slicer in axis_specs:
        center_img = render_sdf_slice(slicer(center_idx), clip_value)
        cv2.putText(
            center_img,
            f"{axis_name} center",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        center_tiles.append(center_img)

        indices = np.linspace(0, grid_np.shape[0] - 1, 7, dtype=int)
        montage_tiles = []
        for idx in indices:
            img = render_sdf_slice(slicer(int(idx)), clip_value)
            cv2.putText(
                img,
                f"{axis_name}={idx}",
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            montage_tiles.append(img)
        cv2.imwrite(os.path.join(obj_dir, f"sdf_slices_axis_{axis_name}.png"), np.hstack(montage_tiles))

    cv2.imwrite(os.path.join(obj_dir, "sdf_center_slices.png"), np.hstack(center_tiles))


def try_export_sdf_isosurface(
    grid_np: np.ndarray,
    bounds_np: np.ndarray,
    obj_dir: str,
) -> None:
    """Export a reconstructed zero-level surface when scikit-image is available."""
    try:
        from skimage.measure import marching_cubes
    except ImportError:
        return

    if grid_np.min() > 0.0 or grid_np.max() < 0.0:
        return

    spacing = (bounds_np[1] - bounds_np[0]) / np.maximum(np.array(grid_np.shape) - 1, 1)
    verts, faces, _, _ = marching_cubes(grid_np, level=0.0, spacing=tuple(spacing))
    verts = verts + bounds_np[0]
    iso_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    iso_mesh.export(os.path.join(obj_dir, "sdf_zero_isosurface.glb"))





def _mesh_diagnostics(mesh: trimesh.Trimesh) -> str:
    '''
    Return a string with the mesh diagnostics.
    '''
    return (
        f"watertight={mesh.is_watertight}, "
        f"winding_consistent={mesh.is_winding_consistent}, "
        f"euler={mesh.euler_number}, "
        f"volume={mesh.volume:.6g}, "
        f"components={len(mesh.split(only_watertight=False))}"
    )


def make_mesh_watertight_for_sdf(
    mesh: trimesh.Trimesh,
    obj_id: str,
    require_watertight: bool = True,
    method: str = "voxel",
    voxel_resolution: int = 96,
    voxel_dilation_iterations: int = 2,
    voxel_erosion_iterations: Optional[int] = None,
) -> trimesh.Trimesh:
    """
    Build a watertight geometry proxy before SDF computation.

    Open3D's signed distance query only has reliable inside/outside signs for
    closed surfaces.  The default voxel reconstruction first seals small leaks
    with 3-D morphology, then flood-fills enclosed cavities before extracting a
    surface.  This avoids hollow shells whose centers are still outside.
    """
    if mesh.is_watertight and mesh.is_winding_consistent:
        return mesh.copy()

    repaired = mesh.copy()
    repaired.remove_unreferenced_vertices()
    trimesh.repair.fix_winding(repaired)
    trimesh.repair.fix_normals(repaired, multibody=True)
    trimesh.repair.fill_holes(repaired)
    repaired.remove_unreferenced_vertices()
    if repaired.is_watertight and repaired.is_winding_consistent:
        return repaired

    method = method.lower()
    if method == "trimesh":
        pass
    elif method == "voxel":
        repaired = _make_watertight_with_voxels(
            mesh,
            voxel_resolution=voxel_resolution,
            dilation_iterations=voxel_dilation_iterations,
            erosion_iterations=voxel_erosion_iterations,
        )
    elif method == "pymeshfix":
        repaired = _make_watertight_with_pymeshfix(mesh)
    else:
        raise ValueError(
            f"Unknown SDF watertight repair method '{method}'. "
            "Expected one of: voxel, pymeshfix, trimesh."
        )

    trimesh.repair.fix_winding(repaired)
    trimesh.repair.fix_normals(repaired, multibody=True)
    trimesh.repair.fill_holes(repaired)
    repaired.remove_unreferenced_vertices()

    if require_watertight and not repaired.is_watertight:
        raise RuntimeError(
            f"{obj_id}: '{method}' repair did not produce a watertight mesh for SDF. "
            f"Before: {_mesh_diagnostics(mesh)}; "
            f"After: {_mesh_diagnostics(repaired)}"
        )
    if require_watertight and not repaired.is_winding_consistent:
        raise RuntimeError(
            f"{obj_id}: repaired SDF mesh is watertight but winding is inconsistent. "
            f"Before: {_mesh_diagnostics(mesh)}; "
            f"After: {_mesh_diagnostics(repaired)}"
        )

    return repaired


def _make_watertight_with_voxels(
    mesh: trimesh.Trimesh,
    voxel_resolution: int,
    dilation_iterations: int = 2,
    erosion_iterations: Optional[int] = None,
) -> trimesh.Trimesh:
    extents = np.asarray(mesh.extents, dtype=np.float64)
    longest_extent = float(extents.max())
    if not np.isfinite(longest_extent) or longest_extent <= 0.0:
        raise ValueError(f"Invalid mesh extents for voxel repair: {extents}")

    pitch = longest_extent / max(int(voxel_resolution) - 1, 1)
    voxel_grid = mesh.voxelized(pitch)
    solid_matrix, solid_transform = _solidify_voxel_matrix(
        voxel_grid.matrix,
        voxel_grid.transform,
        dilation_iterations=dilation_iterations,
        erosion_iterations=erosion_iterations,
    )
    solid_grid = trimesh.voxel.VoxelGrid(solid_matrix, transform=solid_transform)
    repaired = solid_grid.marching_cubes
    repaired.apply_transform(solid_grid.transform)
    repaired.process(validate=True)
    return repaired


def _solidify_voxel_matrix(
    occupied: np.ndarray,
    transform: np.ndarray,
    dilation_iterations: int = 2,
    erosion_iterations: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a possibly leaky surface voxelization into a solid occupancy grid.

    The sequence is deliberately image-processing style:
      1. pad exterior air,
      2. dilate the shell to close small tunnels/pinholes,
      3. fill every cavity not reachable from the padded border,
      4. erode back to approximately the original boundary thickness.
    """
    try:
        from scipy import ndimage
    except ImportError as exc:
        raise ImportError(
            "scipy is required for voxel morphology based SDF watertight repair."
        ) from exc

    dilation_iterations = max(0, int(dilation_iterations))
    if erosion_iterations is None:
        erosion_iterations = dilation_iterations
    erosion_iterations = max(0, int(erosion_iterations))

    occupied = np.asarray(occupied, dtype=bool)
    pad = max(dilation_iterations, erosion_iterations) + 2
    padded = np.pad(occupied, pad_width=pad, mode="constant", constant_values=False)

    structure = ndimage.generate_binary_structure(rank=3, connectivity=2)
    solid = padded
    if dilation_iterations > 0:
        solid = ndimage.binary_dilation(
            solid,
            structure=structure,
            iterations=dilation_iterations,
        )

    solid = ndimage.binary_fill_holes(solid, structure=structure)

    if erosion_iterations > 0:
        solid = ndimage.binary_erosion(
            solid,
            structure=structure,
            iterations=erosion_iterations,
            border_value=0,
        )

    if not solid.any():
        raise RuntimeError("Voxel morphology removed all occupied voxels.")

    # Strip empty padding for smaller marching-cubes meshes while preserving the
    # original voxel-to-world mapping.
    filled_indices = np.argwhere(solid)
    min_idx = filled_indices.min(axis=0)
    max_idx = filled_indices.max(axis=0) + 1
    solid = solid[
        min_idx[0]:max_idx[0],
        min_idx[1]:max_idx[1],
        min_idx[2]:max_idx[2],
    ]

    offset = min_idx.astype(np.float64) - float(pad)
    solid_transform = np.array(transform, dtype=np.float64, copy=True)
    linear = solid_transform[:3, :3]
    solid_transform[:3, 3] = solid_transform[:3, 3] + linear @ offset

    return solid, solid_transform


def _make_watertight_with_pymeshfix(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    try:
        import pymeshfix
    except ImportError as exc:
        raise ImportError(
            "pymeshfix is required for watertight SDF mesh repair. "
            "Install it in the foundationpose environment with: "
            "python -m pip install pymeshfix"
        ) from exc

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    meshfix = pymeshfix.MeshFix(vertices, faces)
    meshfix.repair(
        verbose=False,
        joincomp=False,
        remove_smallest_components=False,
    )

    repaired = trimesh.Trimesh(
        vertices=np.asarray(meshfix.points, dtype=np.float64),
        faces=np.asarray(meshfix.faces, dtype=np.int64),
        process=True,
    )
    return repaired