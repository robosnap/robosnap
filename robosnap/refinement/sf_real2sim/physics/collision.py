"""
Convex decomposition with disk caching.

Uses V-HACD or CoACD to decompose each mesh into convex hulls, then
saves them as .glb files under each object's directory for reuse.
"""

import os
import shutil
from typing import Dict, List

import numpy as np
import trimesh


def _split_mesh_into_octants(mesh: trimesh.Trimesh) -> List[trimesh.Trimesh]:
    """Split a mesh into up to 8 pieces by cutting once on each axis."""
    center = np.asarray(mesh.bounding_box.centroid, dtype=np.float64)
    axes = np.eye(3, dtype=np.float64)
    pieces = []

    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                piece = mesh.copy()
                for sign, axis in ((sx, axes[0]), (sy, axes[1]), (sz, axes[2])):
                    piece = piece.slice_plane(center, sign * axis)
                    if piece is None or len(piece.vertices) == 0 or len(piece.faces) == 0:
                        break
                if piece is not None and len(piece.vertices) > 0 and len(piece.faces) > 0:
                    pieces.append(piece)

    return pieces


def _extract_vhacd_collision(mesh: trimesh.Trimesh) -> List[trimesh.Trimesh]:
    """Decompose a mesh into convex hulls via V-HACD."""
    from vhacdx import compute_vhacd


    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    mesh.process(validate=True)

    pts = np.asarray(mesh.vertices, dtype=np.float64)
    faces_flat = np.asarray(mesh.faces, dtype=np.uint32).flatten()
    parts = compute_vhacd(
        pts,
        faces_flat,
        resolution=2000000,
        maxConvexHulls=64,
        minimumVolumePercentErrorAllowed=0.05,
        maxNumVerticesPerCH=128,
        shrinkWrap=True,
        fillMode="surface",
        maxRecursionDepth=15,
        findBestPlane=True
    )
    hulls = []
    for vertices, faces in parts:
        hull = trimesh.Trimesh(
            vertices=np.asarray(vertices),
            faces=np.asarray(faces),
            process=False,
        )
        hulls.append(hull)
    return hulls


def _extract_coacd_collision(mesh: trimesh.Trimesh) -> List[trimesh.Trimesh]:
    """Decompose a mesh into convex hulls via CoACD."""
    import coacd

    coacd_mesh = coacd.Mesh(
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.faces, dtype=np.int32),
    )
    parts = coacd.run_coacd(
        coacd_mesh,
        # threshold=0.05,
        max_convex_hull=128,
        # preprocess_mode="auto",
        # preprocess_resolution=50,
        # resolution=2000,
        # mcts_iterations=100,
        # mcts_max_depth=3,
        # mcts_nodes=20,
    )
    hulls = []
    for vertices, faces in parts:
        hull = trimesh.Trimesh(
            vertices=np.asarray(vertices),
            faces=np.asarray(faces),
            process=False,
        )
        hulls.append(hull)
    return hulls


def extract_vhacd_collision(
    mesh: trimesh.Trimesh,
    method: str = "coacd",
    split_mesh: bool = True,
) -> List[trimesh.Trimesh]:
    """Decompose a mesh into convex hulls via V-HACD or CoACD.

    Args:
        mesh: Input triangle mesh.
        method: Convex decomposition backend, either ``"vhacd"`` or ``"coacd"``.
        split_mesh: If true, split the mesh into 8 axis-aligned pieces first,
            decompose each piece, and combine the produced hulls.

    Returns:
        List of convex hull meshes.
    """
    if split_mesh:
        hulls = []
        pieces = _split_mesh_into_octants(mesh)
        if not pieces:
            pieces = [mesh]
        for piece in pieces:
            hulls.extend(extract_vhacd_collision(
                piece,
                method=method,
                split_mesh=False,
            ))
        return hulls

    method = method.lower()
    if method == "vhacd":
        return _extract_vhacd_collision(mesh)
    if method == "coacd":
        return _extract_coacd_collision(mesh)
    raise ValueError(
        f"Unsupported collision decomposition method: {method!r}. "
        "Expected 'vhacd' or 'coacd'."
    )


def _get_cached_collision_paths(cache_dir: str) -> List[str]:
    if not os.path.isdir(cache_dir):
        return []

    paths = []
    for name in sorted(os.listdir(cache_dir)):
        if name.startswith("collision_part_") and name.endswith(".glb"):
            paths.append(os.path.join(cache_dir, name))
    return paths


def prepare_all_collisions(
    meshes: Dict[str, trimesh.Trimesh],
    results_dir: str,
    method: str = "coacd",
    split_mesh: bool = True,
    use_cached_collisions: bool = False,
) -> Dict[str, List[str]]:
    """Compute collision meshes for all objects, caching to disk.

    Collision parts are saved under ``results_dir/<obj_id>/collision/``.
    Existing collision parts are removed before recomputing unless
    ``use_cached_collisions`` is true.

    Args:
        meshes: Mapping from obj_id to trimesh mesh.
        results_dir: Root results directory containing ``obj_*/`` folders.
        method: Convex decomposition backend, either ``"vhacd"`` or ``"coacd"``.
        split_mesh: If true, split every mesh into 8 pieces before decomposition.
        use_cached_collisions: If true, reuse existing ``collision_part_*.glb``
            files when present and only compute missing object caches.

    Returns:
        Mapping from obj_id to list of collision .glb file paths.
    """
    collision_paths: Dict[str, List[str]] = {}

    for obj_id in sorted(meshes.keys()):
        cache_dir = os.path.join(results_dir, obj_id, "collision")
        cached_paths = _get_cached_collision_paths(cache_dir)
        if use_cached_collisions and cached_paths:
            collision_paths[obj_id] = cached_paths
            print(f"  {obj_id}: using {len(cached_paths)} cached collision parts")
            continue

        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)

        print(
            f"  {obj_id}: computing {method.upper()} decomposition"
            f"{' with 8-way split' if split_mesh else ''} ...",
            end=" ",
            flush=True,
        )
        mesh = meshes[obj_id]
        hulls = extract_vhacd_collision(
            mesh,
            method=method,
            split_mesh=split_mesh,
        )

        if not hulls:
            print(f"[WARNING]: {method.upper()} produced 0 parts for {obj_id}, "
                  "falling back to convex hull")
            hulls = [mesh.convex_hull]

        paths = []
        for i, hull in enumerate(hulls):
            path = os.path.join(cache_dir, f"collision_part_{i}.glb")
            hull.export(path)
            paths.append(path)

        collision_paths[obj_id] = paths
        print(f"{len(hulls)} parts")

        # import pdb; pdb.set_trace()

    return collision_paths
