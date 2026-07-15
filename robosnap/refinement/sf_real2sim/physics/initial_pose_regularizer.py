"""
Initial pose regularization for LayoutOpt v3.

This file does one thing only: before SDF/SAPIEN optimization begins, it uses the scene graph to roughly adjust obviously unreasonable initial positions. It modifies translation only, leaving rotation unchanged.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import open3d as o3d
import trimesh


@dataclass
class InitialPoseRegularizerConfig:
    # 每个 mesh 采多少表面点来判断 child 是否插进 parent 里。
    sample_points_per_mesh: int = 2048
    # child 到 parent 表面的最小 signed distance 大于该值时，认为已经不穿模。
    collision_clearance: float = 0.001
    # 每次沿世界坐标 +Z 抬高多少米。
    z_step: float = 0.005
    # 单条 support 边最多允许 child 抬高多少米，防止坏 graph 造成无限循环。
    max_z_lift: float = 0.20
    # 每次在 XY 平面朝 parent 移动多少米。
    xy_step: float = 0.005
    # 单条 support 边最多允许 child 在 XY 平面移动多少米。
    max_xy_move: float = 0.20
    # 判断两个 XY 投影是否重叠时留一点数值余量。
    xy_overlap_epsilon: float = 1e-5
    # support 链可能是 book -> box -> table，多跑几轮可以传播父节点移动带来的影响。
    num_passes: int = 2
    seed: int = 42


@dataclass
class SupportEdge:
    # 在现有 scene graph 里：source_id 是被支撑的物体，target_id 是支撑它的物体。
    # 这里把 source 叫 child，把 target 叫 parent，读起来更直观。
    child_obj_id: str
    parent_obj_id: str
    source_id: int
    target_id: int


def regularize_initial_poses(
    meshes: Dict[str, trimesh.Trimesh],
    initial_poses: Dict[str, np.ndarray],
    scene_graph_path: str,
    config: InitialPoseRegularizerConfig | None = None,
) -> Dict[str, np.ndarray]:
    """根据 scene graph 里的 Support 边修正初始 pose。"""
    cfg = config or InitialPoseRegularizerConfig()
    # 复制一份 pose，避免原地改调用方传进来的 initial_poses。
    poses = {
        obj_id: pose.astype(np.float32).copy()
        for obj_id, pose in initial_poses.items()
    }
    scene_graph = _load_scene_graph(scene_graph_path)
    node_id_to_obj_id = _build_node_mapping(scene_graph, list(meshes.keys()))
    support_edges = _collect_support_edges(scene_graph, node_id_to_obj_id, meshes, poses)

    if not support_edges:
        print("[InitialPoseRegularizer] No valid Support edges found.")
        return poses

    print(
        f"[InitialPoseRegularizer] Regularizing {len(support_edges)} Support edges "
        f"for {cfg.num_passes} pass(es)."
    )

    # child-parent 穿模检测：用 child 的采样点去查 parent 的 signed distance。
    sampled_points = _sample_points(meshes, cfg.sample_points_per_mesh, cfg.seed)
    # 每个 parent mesh 建一个 Open3D RaycastingScene，用来快速查 signed distance。
    sdf_scenes = {
        obj_id: _build_raycasting_scene(mesh)
        for obj_id, mesh in meshes.items()
    }
    # 尽量先处理 table/box 这种 parent，再处理上面的 child。
    ordered_edges = _order_support_edges(support_edges)

    for pass_idx in range(cfg.num_passes):
        print(f"[InitialPoseRegularizer] Pass {pass_idx + 1}/{cfg.num_passes}")
        any_moved = False
        for edge in ordered_edges:
            moved_xy = _move_child_until_xy_overlap(
                edge=edge,
                meshes=meshes,
                poses=poses,
                cfg=cfg,
            )
            moved_z = _lift_child_until_pair_clear(
                edge=edge,
                poses=poses,
                sampled_points=sampled_points,
                sdf_scenes=sdf_scenes,
                cfg=cfg,
            )
            any_moved = any_moved or moved_xy or moved_z

        if not any_moved:
            print("[InitialPoseRegularizer] No movement in this pass; stopping.")
            break

    return poses


def _load_scene_graph(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("graph", {})


def _build_node_mapping(scene_graph: dict, obj_ids: List[str]) -> Dict[int, str]:
    # scene graph 用整数 id，磁盘目录是 obj_0、obj_1...
    # 正常情况下 node id == obj 后缀；如果不匹配，就退化到排序后的 obj 列表。
    obj_ids_sorted = sorted(obj_ids)
    mapping = {}
    for node in scene_graph.get("nodes", []):
        node_id = node.get("id")
        if node_id is None:
            continue
        candidate = f"obj_{node_id}"
        if candidate in obj_ids_sorted:
            mapping[int(node_id)] = candidate
        elif int(node_id) < len(obj_ids_sorted):
            mapping[int(node_id)] = obj_ids_sorted[int(node_id)]
    return mapping


def _collect_support_edges(
    scene_graph: dict,
    node_id_to_obj_id: Dict[int, str],
    meshes: Dict[str, trimesh.Trimesh],
    poses: Dict[str, np.ndarray],
) -> List[SupportEdge]:
    support_edges = []
    for item in scene_graph.get("edges", []):
        if item.get("relation", "Support") != "Support":
            continue

        # 注意方向：source 被 target 支撑，不要反过来。
        source_id = int(item["source_id"])
        target_id = int(item["target_id"])
        child_obj_id = node_id_to_obj_id.get(source_id)
        parent_obj_id = node_id_to_obj_id.get(target_id)
        if child_obj_id is None or parent_obj_id is None:
            continue
        if child_obj_id not in meshes or parent_obj_id not in meshes:
            continue
        if child_obj_id not in poses or parent_obj_id not in poses:
            continue

        support_edges.append(
            SupportEdge(
                child_obj_id=child_obj_id,
                parent_obj_id=parent_obj_id,
                source_id=source_id,
                target_id=target_id,
            )
        )
    return support_edges


def _order_support_edges(edges: List[SupportEdge]) -> List[SupportEdge]:
    """尽量先处理 parent，再处理被它支撑的 child。"""
    child_to_edges = {}
    children = set()
    parents = set()
    for edge in edges:
        child_to_edges.setdefault(edge.parent_obj_id, []).append(edge)
        children.add(edge.child_obj_id)
        parents.add(edge.parent_obj_id)

    roots = sorted(parents - children)
    ordered = []
    visited = set()
    queue = roots[:] if roots else sorted(parents)

    while queue:
        parent = queue.pop(0)
        for edge in sorted(child_to_edges.get(parent, []), key=lambda e: e.child_obj_id):
            key = (edge.child_obj_id, edge.parent_obj_id)
            if key in visited:
                continue
            visited.add(key)
            ordered.append(edge)
            queue.append(edge.child_obj_id)

    for edge in edges:
        key = (edge.child_obj_id, edge.parent_obj_id)
        if key not in visited:
            ordered.append(edge)
    return ordered


def _sample_points(
    meshes: Dict[str, trimesh.Trimesh],
    num_points: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    sampled = {}
    for offset, (obj_id, mesh) in enumerate(sorted(meshes.items())):
        points, _ = trimesh.sample.sample_surface(mesh, num_points, seed=seed + offset)
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        # 表面随机采样点 + 所有顶点一起用，避免只采样时漏掉尖角/边界穿模。
        sampled[obj_id] = np.concatenate(
            [np.asarray(points, dtype=np.float32), vertices],
            axis=0,
        )
    return sampled


def _build_raycasting_scene(mesh: trimesh.Trimesh) -> o3d.t.geometry.RaycastingScene:
    # Open3D 的 RaycastingScene 可以对 mesh 查询 signed distance：
    # 正值表示在物体外，负值表示点在物体内部。
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    o3d_mesh = o3d.t.geometry.TriangleMesh()
    o3d_mesh.vertex.positions = o3d.core.Tensor(vertices)
    o3d_mesh.triangle.indices = o3d.core.Tensor(faces)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d_mesh)
    return scene


def _move_child_until_xy_overlap(
    edge: SupportEdge,
    meshes: Dict[str, trimesh.Trimesh],
    poses: Dict[str, np.ndarray],
    cfg: InitialPoseRegularizerConfig,
) -> bool:
    # 规则 2：child 和 parent 在水平 XY 平面上的投影至少要有重叠。
    # 如果没有重叠，就把 child 朝 parent 的 XY 中心一点点平移。
    moved = 0.0
    child = edge.child_obj_id
    parent = edge.parent_obj_id

    while moved < cfg.max_xy_move:
        # 用 mesh 顶点变换到世界坐标后得到 XY 凸包，比单纯 AABB 更贴近物体形状。
        child_poly = _world_xy_hull(meshes[child], poses[child])
        parent_poly = _world_xy_hull(meshes[parent], poses[parent])
        if _polygons_overlap(child_poly, parent_poly, cfg.xy_overlap_epsilon):
            if moved > 0:
                print(
                    f"  [XY] {child} -> {parent}: moved {moved:.4f}m toward parent"
                )
            return moved > 0

        child_center = _poly_center(child_poly)
        parent_center = _poly_center(parent_poly)
        direction = parent_center - child_center
        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            return moved > 0

        step = min(cfg.xy_step, cfg.max_xy_move - moved)
        delta_xy = direction / norm * step
        # 只改 translation 的 x/y，不碰旋转。
        poses[child][:2, 3] += delta_xy.astype(poses[child].dtype)
        moved += step

    print(
        f"  [XY][WARN] {child} -> {parent}: no overlap after {moved:.4f}m move"
    )
    return moved > 0


def _lift_child_until_pair_clear(
    edge: SupportEdge,
    poses: Dict[str, np.ndarray],
    sampled_points: Dict[str, np.ndarray],
    sdf_scenes: Dict[str, o3d.t.geometry.RaycastingScene],
    cfg: InitialPoseRegularizerConfig,
) -> bool:
    # 规则 1：如果 child 插进 parent 里，就只沿世界 +Z 抬 child。
    # 这里按你的要求，只检查当前 child-parent pair，不检查 child 和其它物体。
    lifted = 0.0
    child = edge.child_obj_id
    parent = edge.parent_obj_id

    while lifted < cfg.max_z_lift:
        min_sdf = _min_signed_distance_child_to_parent(
            child_points=sampled_points[child],
            child_pose=poses[child],
            parent_pose=poses[parent],
            parent_scene=sdf_scenes[parent],
        )
        if min_sdf >= cfg.collision_clearance:
            if lifted > 0:
                print(
                    f"  [Z] {child} -> {parent}: lifted {lifted:.4f}m "
                    f"(min_sdf={min_sdf:.5f})"
                )
            return lifted > 0

        step = min(cfg.z_step, cfg.max_z_lift - lifted)
        # 只改 translation 的 z，不碰旋转。
        poses[child][2, 3] += step
        lifted += step

    min_sdf = _min_signed_distance_child_to_parent(
        child_points=sampled_points[child],
        child_pose=poses[child],
        parent_pose=poses[parent],
        parent_scene=sdf_scenes[parent],
    )
    print(
        f"  [Z][WARN] {child} -> {parent}: still colliding after "
        f"{lifted:.4f}m lift (min_sdf={min_sdf:.5f})"
    )
    return lifted > 0


def _min_signed_distance_child_to_parent(
    child_points: np.ndarray,
    child_pose: np.ndarray,
    parent_pose: np.ndarray,
    parent_scene: o3d.t.geometry.RaycastingScene,
) -> float:
    # 把 child 局部点先变到世界坐标，再变到 parent 局部坐标，
    # 因为 parent 的 signed distance field 是在 parent 自己的局部坐标里查询的。
    world_points = _transform_points(child_points, child_pose)
    parent_local_points = _transform_points_to_local(world_points, parent_pose)
    distances = parent_scene.compute_signed_distance(
        o3d.core.Tensor(parent_local_points.astype(np.float32))
    ).numpy()
    return float(np.min(distances))


def _world_xy_hull(mesh: trimesh.Trimesh, pose: np.ndarray) -> np.ndarray:
    # 得到物体当前世界坐标下的 XY 投影凸包。
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    world_vertices = _transform_points(vertices, pose)
    return _convex_hull_2d(world_vertices[:, :2])


def _transform_points(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    # local -> world: p_world = R p_local + t
    return points @ pose[:3, :3].T + pose[:3, 3]


def _transform_points_to_local(world_points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    # world -> local。这里 pose 是刚体变换，所以 R^-1 = R.T；
    # 对 row-vector 写法就是 (p - t) @ R。
    return (world_points - pose[:3, 3]) @ pose[:3, :3]


def _convex_hull_2d(points: np.ndarray) -> np.ndarray:
    # Andrew monotonic chain 算法：把一堆 2D 点变成凸包顶点。
    unique = sorted(set(map(tuple, np.asarray(points, dtype=np.float64))))
    if len(unique) <= 1:
        return np.asarray(unique, dtype=np.float64)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def _polygons_overlap(poly_a: np.ndarray, poly_b: np.ndarray, eps: float) -> bool:
    # 分离轴定理：如果存在某个轴上两个投影不重叠，则两个凸多边形不重叠。
    if len(poly_a) == 0 or len(poly_b) == 0:
        return False
    if len(poly_a) < 3 or len(poly_b) < 3:
        return _aabb_overlap(poly_a, poly_b, eps)

    for axis in _polygon_axes(poly_a) + _polygon_axes(poly_b):
        min_a, max_a = _project(poly_a, axis)
        min_b, max_b = _project(poly_b, axis)
        if max_a < min_b + eps or max_b < min_a + eps:
            return False
    return True


def _polygon_axes(poly: np.ndarray) -> List[np.ndarray]:
    axes = []
    for i in range(len(poly)):
        p0 = poly[i]
        p1 = poly[(i + 1) % len(poly)]
        edge = p1 - p0
        norm = np.linalg.norm(edge)
        if norm < 1e-12:
            continue
        axis = np.array([-edge[1], edge[0]], dtype=np.float64) / norm
        axes.append(axis)
    return axes


def _project(poly: np.ndarray, axis: np.ndarray) -> Tuple[float, float]:
    values = poly @ axis
    return float(values.min()), float(values.max())


def _aabb_overlap(points_a: np.ndarray, points_b: np.ndarray, eps: float) -> bool:
    # 点数太少时没有稳定凸包，退化成 AABB 重叠判断。
    min_a = points_a.min(axis=0)
    max_a = points_a.max(axis=0)
    min_b = points_b.min(axis=0)
    max_b = points_b.max(axis=0)
    return bool(np.all(max_a >= min_b + eps) and np.all(max_b >= min_a + eps))


def _poly_center(poly: np.ndarray) -> np.ndarray:
    # 计算 2D 多边形中心；退化情况用点均值。
    if len(poly) == 0:
        return np.zeros(2, dtype=np.float64)
    if len(poly) < 3:
        return poly.mean(axis=0)

    area_twice = 0.0
    center = np.zeros(2, dtype=np.float64)
    for i in range(len(poly)):
        p0 = poly[i]
        p1 = poly[(i + 1) % len(poly)]
        cross = p0[0] * p1[1] - p1[0] * p0[1]
        area_twice += cross
        center += (p0 + p1) * cross

    if math.isclose(area_twice, 0.0, abs_tol=1e-12):
        return poly.mean(axis=0)
    return center / (3.0 * area_twice)
