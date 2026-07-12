"""
SDF-based scene layout optimizer following the CAST paper (Section 5).

Given a set of meshes with initial poses and a scene graph describing
Support / Contact relationships, this optimizer adjusts the rigid-body
transforms of each object to:
  1. Eliminate inter-object penetration,
  2. Satisfy support/contact constraints from the scene graph,
  3. Stay close to the initial pose estimates.
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import trimesh

from .losses import (
    contact_loss,
    penetration_loss,
    regularization_loss,
    support_loss,
)
from .sdf_utils import (
    compute_sdf_grid,
    query_sdf,
    sample_surface_points,

    save_sdf_debug_artifacts,
    make_mesh_watertight_for_sdf,
    _mesh_diagnostics
)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ObjectData:
    """All precomputed data for a single object."""
    obj_id: str
    mesh: trimesh.Trimesh
    initial_pose: np.ndarray  # (4, 4)
    sdf_grid: torch.Tensor    # (1, 1, D, H, W)
    sdf_bounds: torch.Tensor  # (2, 3)
    surface_pts: torch.Tensor # (N, 3) in local frame
    is_root: bool = False


@dataclass
class SceneGraphEdge:
    source_id: int
    target_id: int
    relation: str       # "Support" or "Contact"
    fine_relation: str   # "Stack", "Lean", etc.


@dataclass
class OptimConfig:
    sdf_resolution: int = 128
    sdf_padding: float = 0.1
    num_surface_points: int = 1024
    num_iterations: int = 200
    lr: float = 1e-3
    w_penetration: float = 10.0
    w_support: float = 5.0
    w_contact: float = 5.0
    w_regularization: float = 1.0
    w_global_pen: float = 16.0#8.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    translation_only_iters: int = 10


    # sdf debug
    sdf_debug_dir: Optional[str] = None
    save_sdf_debug_visualizations: bool = True

    # to watertight mesh for SDF computation
    make_sdf_mesh_watertight: bool = True
    require_watertight_sdf_mesh: bool = True
    sdf_watertight_method: str = "voxel"  # "voxel", "pymeshfix", or "trimesh"
    sdf_watertight_voxel_resolution: int = 96
    sdf_watertight_voxel_dilation_iters: int = 3
    sdf_watertight_voxel_erosion_iters: Optional[int] = None
    save_watertight_mesh_debug: bool = True





# ---------------------------------------------------------------------------
# Rotation helpers (axis-angle <-> matrix)
# ---------------------------------------------------------------------------

def _axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle (3,) to rotation matrix (3, 3) using Rodrigues."""
    angle = axis_angle.norm()
    if angle < 1e-8:
        return torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    k = axis_angle / angle
    K = torch.zeros(3, 3, device=axis_angle.device, dtype=axis_angle.dtype)
    K[0, 1] = -k[2]
    K[0, 2] = k[1]
    K[1, 0] = k[2]
    K[1, 2] = -k[0]
    K[2, 0] = -k[1]
    K[2, 1] = k[0]
    return torch.eye(3, device=axis_angle.device) + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)


def _build_transform(
    initial_pose: torch.Tensor,
    delta_t: torch.Tensor,
    delta_r: torch.Tensor,
) -> torch.Tensor:
    """
    Compose the optimised world transform from initial pose and deltas.

    T_opt = delta_T @ T_init  where delta_T is built from (delta_t, delta_r).
    """
    delta_R = _axis_angle_to_matrix(delta_r)
    T = torch.eye(4, device=initial_pose.device, dtype=initial_pose.dtype)
    T[:3, :3] = delta_R @ initial_pose[:3, :3]
    T[:3, 3] = delta_R @ initial_pose[:3, 3] + delta_t
    return T


# ---------------------------------------------------------------------------
# Main optimizer
# ---------------------------------------------------------------------------

class SDFSceneOptimizer:
    """
    Jointly optimises object poses using SDF-based losses and scene graph
    constraints.
    """

    def __init__(
        self,
        meshes: Dict[str, trimesh.Trimesh],
        initial_poses: Dict[str, np.ndarray],
        scene_graph_path: str,
        config: Optional[OptimConfig] = None,
    ):
        """
        Args:
            meshes: Mapping from obj_id (e.g. "obj_0") to a trimesh.Trimesh.
            initial_poses: Mapping from obj_id to (4,4) numpy pose matrix.
            scene_graph_path: Path to ``scene_graph.json``.
            config: Optimisation hyper-parameters.
        """
        self.config = config or OptimConfig()
        self.device = torch.device(self.config.device)

        self.scene_graph = self._load_scene_graph(scene_graph_path)
        self.edges: List[SceneGraphEdge] = self.scene_graph["edges"]
        self.node_id_to_obj_id = self._build_node_mapping(
            self.scene_graph, list(meshes.keys())
        )

        self.objects: Dict[str, ObjectData] = {}
        self._precompute(meshes, initial_poses)

        self.root_ids = self._find_roots()
        for rid in self.root_ids:
            obj_id = self.node_id_to_obj_id.get(rid)
            if obj_id and obj_id in self.objects:
                self.objects[obj_id].is_root = True

        print(f"Root objects (fixed): {[self.node_id_to_obj_id.get(r, r) for r in self.root_ids]}")
        print(f"Scene graph edges: {len(self.edges)}")

    # -----------------------------------------------------------------------
    # Scene graph loading
    # -----------------------------------------------------------------------

    @staticmethod
    def _load_scene_graph(path: str) -> dict:
        with open(path, "r") as f:
            data = json.load(f)
        graph = data.get("graph", {})
        edges = []
        for e in graph.get("edges", []):
            edges.append(SceneGraphEdge(
                source_id=e["source_id"],
                target_id=e["target_id"],
                relation=e.get("relation", "Support"),
                fine_relation=e.get("fine_relation", "Stack"),
            ))
        return {"nodes": graph.get("nodes", []), "edges": edges}

    def _build_node_mapping(
        self,
        scene_graph: dict,
        obj_ids: List[str],
    ) -> Dict[int, str]:
        """
        Map scene-graph node ids (0, 1, 2 …) to object directory names
        (obj_0, obj_1 …).

        Heuristic: node id ``n`` corresponds to ``obj_<original_index>`` if
        available, otherwise to the n-th object directory (sorted).
        """
        obj_ids_sorted = sorted(obj_ids)
        mapping: Dict[int, str] = {}
        for node in scene_graph["nodes"]:
            nid = node["id"]
            candidate = f"obj_{nid}"
            if candidate in obj_ids_sorted:
                mapping[nid] = candidate
            elif nid < len(obj_ids_sorted):
                mapping[nid] = obj_ids_sorted[nid]
        return mapping

    # -----------------------------------------------------------------------
    # Precomputation
    # -----------------------------------------------------------------------

    def _precompute(
        self,
        meshes: Dict[str, trimesh.Trimesh],
        initial_poses: Dict[str, np.ndarray],
    ) -> None:
        print("Precomputing SDF grids and surface samples …")
        t0 = time.time()
        for obj_id in sorted(meshes.keys()):
            mesh = meshes[obj_id]
            pose = initial_poses[obj_id]
            print(f"  {obj_id}: {len(mesh.vertices)} verts, "
                  f"SDF grid {self.config.sdf_resolution}³ …", end=" ", flush=True)
            print(f"\n    input: {_mesh_diagnostics(mesh)}")
            print("    bounds:", mesh.bounds)

            if self.config.make_sdf_mesh_watertight:
                mesh = make_mesh_watertight_for_sdf(
                    mesh,
                    obj_id=obj_id,
                    require_watertight=self.config.require_watertight_sdf_mesh,
                    method=self.config.sdf_watertight_method,
                    voxel_resolution=self.config.sdf_watertight_voxel_resolution,
                    voxel_dilation_iterations=self.config.sdf_watertight_voxel_dilation_iters,
                    voxel_erosion_iterations=self.config.sdf_watertight_voxel_erosion_iters,
                )
                print(f"    sdf mesh: {_mesh_diagnostics(mesh)}")
                if self.config.sdf_debug_dir and self.config.save_watertight_mesh_debug:
                    obj_debug_dir = os.path.join(self.config.sdf_debug_dir, obj_id)
                    os.makedirs(obj_debug_dir, exist_ok=True)
                    mesh.export(os.path.join(obj_debug_dir, "mesh_watertight.glb"))

            sdf_grid, sdf_bounds = compute_sdf_grid(
                mesh,
                resolution=self.config.sdf_resolution,
                padding=self.config.sdf_padding,
            )
            surface_pts = sample_surface_points(
                mesh, num_points=self.config.num_surface_points
            )
            save_sdf_debug_artifacts(
                obj_id=obj_id,
                mesh=mesh,
                sdf_grid=sdf_grid,
                sdf_bounds=sdf_bounds,
                surface_pts=surface_pts,
                debug_root=self.config.sdf_debug_dir,
                padding=self.config.sdf_padding,
                save_visualizations=self.config.save_sdf_debug_visualizations,
            )
            self.objects[obj_id] = ObjectData(
                obj_id=obj_id,
                mesh=mesh,
                initial_pose=pose,
                sdf_grid=sdf_grid.to(self.device),
                sdf_bounds=sdf_bounds.to(self.device),
                surface_pts=surface_pts.to(self.device),
            )
            print("done")
        print(f"Precomputation finished in {time.time() - t0:.1f}s")

    # -----------------------------------------------------------------------
    # Root detection (topological)
    # -----------------------------------------------------------------------

    def _find_roots(self) -> List[int]:
        """
        Root nodes are supporters that are never supported by anything.
        In a Support edge (source supports target), sources with no incoming
        Support edges are roots.
        """
        supported_ids = set()
        supporter_ids = set()
        for e in self.edges:
            if e.relation == "Support":
                supporter_ids.add(e.target_id)
                supported_ids.add(e.source_id)
        # Roots: nodes that appear as targets (supporter) but never as source (supported)
        all_node_ids = {n["id"] for n in self.scene_graph["nodes"]}
        roots = supporter_ids - supported_ids
        if not roots:
            roots = all_node_ids - supported_ids
        return sorted(roots)

    # -----------------------------------------------------------------------
    # Optimisation loop
    # -----------------------------------------------------------------------

    def optimize(self, starting_poses: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, np.ndarray]:
        """
        Run the SDF-based optimisation and return a dict of optimised poses.

        Args:
            starting_poses: Mapping from obj_id to (4, 4) pose matrix to start optimization from.
                            If None, starts from the initial_poses.
                            Note that regularization is always computed w.r.t initial_poses.

        Returns:
            Mapping from obj_id to optimised (4, 4) pose matrix.
        """
        cfg = self.config

        # Build optimisation parameters
        delta_translations: Dict[str, torch.Tensor] = {}
        delta_rotations: Dict[str, torch.Tensor] = {}
        initial_poses_torch: Dict[str, torch.Tensor] = {}

        for obj_id, od in self.objects.items():
            initial_poses_torch[obj_id] = torch.from_numpy(
                od.initial_pose.astype(np.float32)
            ).to(self.device)

            if od.is_root:
                continue  # fixed

            # If starting_poses is provided, compute initial dt and dr so that T_start = _build_transform(T_init, dt, dr)
            if starting_poses is not None and obj_id in starting_poses:
                T_start = starting_poses[obj_id]
                T_init = od.initial_pose

                R_start = T_start[:3, :3]
                t_start = T_start[:3, 3]
                R_init = T_init[:3, :3]
                t_init = T_init[:3, 3]

                # We want: R_opt = delta_R @ R_init => delta_R = R_opt @ R_init.T
                delta_R = R_start @ R_init.T
                # We want: t_opt = delta_R @ t_init + delta_t => delta_t = t_opt - delta_R @ t_init
                delta_t = t_start - delta_R @ t_init

                delta_r_cv2, _ = cv2.Rodrigues(delta_R)
                delta_r = delta_r_cv2.flatten()

                dt = torch.tensor(delta_t, device=self.device, dtype=torch.float32, requires_grad=True)
                dr = torch.tensor(delta_r, device=self.device, dtype=torch.float32, requires_grad=True)

                # trans = _build_transform(
                #     initial_poses_torch[obj_id],
                #     dt,
                #     dr,
                # )
                # print(trans)
                # print(T_start)

                # import pdb; pdb.set_trace()
                # pdb.set_trace()


            else:
                dt = torch.zeros(3, device=self.device, requires_grad=True)
                dr = torch.zeros(3, device=self.device, requires_grad=True)

            delta_translations[obj_id] = dt
            delta_rotations[obj_id] = dr

        params = list(delta_translations.values()) + list(delta_rotations.values())
        if not params:
            print("Nothing to optimise (all objects are roots).")
            return {oid: od.initial_pose for oid, od in self.objects.items()}

        optimizer = torch.optim.Adam(params, lr=cfg.lr)

        obj_ids_sorted = sorted(self.objects.keys())
        all_pairs = [
            (a, b)
            for i, a in enumerate(obj_ids_sorted)
            for b in obj_ids_sorted[i + 1:]
        ]

        print(f"\nOptimising {len(delta_translations)} objects "
              f"({cfg.num_iterations} iters) …")

        for step in range(cfg.num_iterations):
            optimizer.zero_grad()

            # Current transforms
            current_T: Dict[str, torch.Tensor] = {}
            for obj_id in self.objects:
                init_T = initial_poses_torch[obj_id]
                if obj_id in delta_translations:
                    dr = delta_rotations[obj_id]
                    dt = delta_translations[obj_id]
                    # Optionally freeze rotation in early steps
                    if step < cfg.translation_only_iters:
                        dr_eff = torch.zeros_like(dr)
                    else:
                        dr_eff = dr
                    current_T[obj_id] = _build_transform(init_T, dt, dr_eff)
                else:
                    current_T[obj_id] = init_T

            total_loss = torch.tensor(0.0, device=self.device)
            loss_parts: Dict[str, float] = defaultdict(float)

            # --- Scene graph losses ---
            for edge in self.edges:
                src_obj = self.node_id_to_obj_id.get(edge.source_id)
                tgt_obj = self.node_id_to_obj_id.get(edge.target_id)
                if src_obj is None or tgt_obj is None:
                    continue
                if src_obj not in self.objects or tgt_obj not in self.objects:
                    continue

                src_data = self.objects[src_obj]
                tgt_data = self.objects[tgt_obj]

                if edge.relation == "Support":
                    # target supports source  (source stacks on target)
                    l = support_loss(
                        sdf_grid_supporter=tgt_data.sdf_grid,
                        bounds_supporter=tgt_data.sdf_bounds,
                        T_supporter=current_T[tgt_obj],
                        surface_pts_supported=src_data.surface_pts,
                        T_supported=current_T[src_obj],
                    )
                    total_loss = total_loss + cfg.w_support * l
                    loss_parts["support"] += l.item()

                elif edge.relation == "Contact":
                    l = contact_loss(
                        sdf_grid_i=src_data.sdf_grid,
                        bounds_i=src_data.sdf_bounds,
                        T_i=current_T[src_obj],
                        surface_pts_i=src_data.surface_pts,
                        sdf_grid_j=tgt_data.sdf_grid,
                        bounds_j=tgt_data.sdf_bounds,
                        T_j=current_T[tgt_obj],
                        surface_pts_j=tgt_data.surface_pts,
                    )
                    total_loss = total_loss + cfg.w_contact * l
                    loss_parts["contact"] += l.item()

            # --- Global non-penetration for all pairs ---
            for a, b in all_pairs:
                a_data = self.objects[a]
                b_data = self.objects[b]
                l_ab = penetration_loss(
                    a_data.sdf_grid, a_data.sdf_bounds,
                    current_T[a], b_data.surface_pts, current_T[b],
                )
                l_ba = penetration_loss(
                    b_data.sdf_grid, b_data.sdf_bounds,
                    current_T[b], a_data.surface_pts, current_T[a],
                )
                pen = l_ab + l_ba
                total_loss = total_loss + cfg.w_global_pen * pen
                loss_parts["penetration"] += pen.item()

            # --- Regularization ---
            dt_stack = torch.stack(list(delta_translations.values()))
            dr_stack = torch.stack(list(delta_rotations.values()))
            l_reg = regularization_loss(dt_stack, dr_stack)
            total_loss = total_loss + cfg.w_regularization * l_reg
            loss_parts["reg"] = l_reg.item()

            total_loss.backward()
            optimizer.step()

            if step % 50 == 0 or step == cfg.num_iterations - 1:
                parts_str = "  ".join(f"{k}={v:.4f}" for k, v in loss_parts.items())
                print(f"  step {step:4d}  loss={total_loss.item():.5f}  {parts_str}")

        # Build final poses
        optimized_poses: Dict[str, np.ndarray] = {}
        for obj_id in self.objects:
            if obj_id in delta_translations:
                dt = delta_translations[obj_id].detach()
                dr = delta_rotations[obj_id].detach()
                T = _build_transform(initial_poses_torch[obj_id], dt, dr)
                optimized_poses[obj_id] = T.cpu().numpy()
            else:
                optimized_poses[obj_id] = self.objects[obj_id].initial_pose

        return optimized_poses
