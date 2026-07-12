from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy.ndimage import label


def transform_points(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return points @ pose[:3, :3].T + pose[:3, 3]


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene", process=False)
    return loaded.to_geometry() if isinstance(loaded, trimesh.Scene) else loaded


def mask_bbox_xyxy(path: Path) -> tuple[np.ndarray, tuple[int, int]]:
    image = Image.open(path)
    if image.mode in {"RGBA", "LA"}:
        mask = np.asarray(image.getchannel("A")) > 0
    else:
        mask = np.asarray(image.convert("L")) > 0
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        raise ValueError(f"Empty object mask: {path}")
    bbox = np.array(
        [float(cols.min()), float(rows.min()), float(cols.max()) + 1.0, float(rows.max()) + 1.0]
    )
    return bbox, (int(mask.shape[1]), int(mask.shape[0]))


def project_bbox_xyxy(
    points: np.ndarray,
    pose: np.ndarray,
    w2c: np.ndarray,
    intrinsic: np.ndarray,
) -> np.ndarray:
    camera_points = transform_points(transform_points(points, pose), w2c)
    camera_points = camera_points[camera_points[:, 2] > 1e-5]
    if len(camera_points) < 16:
        raise ValueError("Too few mesh samples remain in front of the camera")
    uv = np.column_stack(
        [
            intrinsic[0, 0] * camera_points[:, 0] / camera_points[:, 2] + intrinsic[0, 2],
            intrinsic[1, 1] * camera_points[:, 1] / camera_points[:, 2] + intrinsic[1, 2],
        ]
    )
    return np.array(
        [
            np.percentile(uv[:, 0], 1.0),
            np.percentile(uv[:, 1], 1.0),
            np.percentile(uv[:, 0], 99.0),
            np.percentile(uv[:, 1], 99.0),
        ]
    )


def bbox_iou(left: np.ndarray, right: np.ndarray) -> float:
    size = np.maximum(
        0.0,
        np.minimum(left[2:], right[2:]) - np.maximum(left[:2], right[:2]),
    )
    intersection = float(np.prod(size))
    left_area = float(np.prod(np.maximum(left[2:] - left[:2], 0.0)))
    right_area = float(np.prod(np.maximum(right[2:] - right[:2], 0.0)))
    return intersection / max(left_area + right_area - intersection, 1e-12)


def solve_xy_for_pixel(
    world_z: float,
    pixel: np.ndarray,
    w2c: np.ndarray,
    intrinsic: np.ndarray,
) -> np.ndarray:
    xn = (pixel[0] - intrinsic[0, 2]) / intrinsic[0, 0]
    yn = (pixel[1] - intrinsic[1, 2]) / intrinsic[1, 1]
    rotation = w2c[:3, :3]
    translation = w2c[:3, 3]
    row_u = rotation[0] - xn * rotation[2]
    row_v = rotation[1] - yn * rotation[2]
    matrix = np.array([[row_u[0], row_u[1]], [row_v[0], row_v[1]]])
    rhs = -np.array(
        [
            row_u[2] * world_z + translation[0] - xn * translation[2],
            row_v[2] * world_z + translation[1] - yn * translation[2],
        ]
    )
    return np.linalg.solve(matrix, rhs)


def recenter_projected_bbox(
    points: np.ndarray,
    local_center: np.ndarray,
    pose: np.ndarray,
    target_center: np.ndarray,
    w2c: np.ndarray,
    intrinsic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    result = pose.copy()
    for _ in range(4):
        projected = project_bbox_xyxy(points, result, w2c, intrinsic)
        projected_center = (projected[:2] + projected[2:]) / 2.0
        world_center = transform_points(local_center[None], result)[0]
        camera_center = transform_points(world_center[None], w2c)[0]
        if camera_center[2] <= 1e-5:
            raise ValueError("Object center moved behind the camera")
        center_pixel = np.array(
            [
                intrinsic[0, 0] * camera_center[0] / camera_center[2] + intrinsic[0, 2],
                intrinsic[1, 1] * camera_center[1] / camera_center[2] + intrinsic[1, 2],
            ]
        )
        desired_pixel = center_pixel + target_center - projected_center
        target_xy = solve_xy_for_pixel(
            float(world_center[2]),
            desired_pixel,
            w2c,
            intrinsic,
        )
        result[:2, 3] += target_xy - world_center[:2]
    return result, project_bbox_xyxy(points, result, w2c, intrinsic)


def stabilize_object(
    mesh: trimesh.Trimesh,
    initial_pose: np.ndarray,
    optimized_pose: np.ndarray,
    mask_bbox: np.ndarray,
    w2c: np.ndarray,
    intrinsic: np.ndarray,
    object_id: int,
    min_scale: float,
    max_scale: float,
    scale_samples: int,
) -> tuple[trimesh.Trimesh, np.ndarray, float, float]:
    sampled = trimesh.sample.sample_surface(mesh, 16000, seed=4100 + object_id)[0]
    vertices = np.asarray(mesh.vertices, dtype=np.float64)

    pose = optimized_pose.copy()
    pose[:3, :3] = initial_pose[:3, :3]
    optimized_contact_z = float(transform_points(vertices, optimized_pose)[:, 2].min())
    restored_contact_z = float(transform_points(vertices, pose)[:, 2].min())
    pose[2, 3] += optimized_contact_z - restored_contact_z

    world_vertices = transform_points(vertices, pose)
    contact = vertices[int(np.argmin(world_vertices[:, 2]))]
    bounds_center = np.asarray(mesh.bounds, dtype=np.float64).mean(axis=0)
    target_center = (mask_bbox[:2] + mask_bbox[2:]) / 2.0

    best = None
    for scale in np.geomspace(min_scale, max_scale, scale_samples):
        scaled_points = contact + float(scale) * (sampled - contact)
        scaled_center = contact + float(scale) * (bounds_center - contact)
        try:
            candidate_pose, projected = recenter_projected_bbox(
                scaled_points,
                scaled_center,
                pose,
                target_center,
                w2c,
                intrinsic,
            )
        except (ValueError, IndexError, np.linalg.LinAlgError):
            continue
        if not np.isfinite(projected).all():
            continue
        iou = bbox_iou(mask_bbox, projected)
        target_size = np.maximum(mask_bbox[2:] - mask_bbox[:2], 1e-6)
        projected_size = np.maximum(projected[2:] - projected[:2], 1e-6)
        size_error = float(np.abs(np.log(projected_size / target_size)).mean())
        score = iou - 0.02 * size_error
        if best is None or score > best[0]:
            best = (score, float(scale), candidate_pose, iou)

    if best is None:
        raise RuntimeError(f"No valid projection candidate for object {object_id}")
    _, scale, stabilized_pose, iou = best
    stabilized_mesh = mesh.copy()
    stabilized_mesh.vertices = contact + scale * (vertices - contact)
    return stabilized_mesh, stabilized_pose, scale, float(iou)


def load_gravity_camera(scene_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    gravity = json.loads((scene_dir / "gravity_alignment.json").read_text(encoding="utf-8"))
    transform_gravity_from_camera = np.asarray(
        gravity["transform_gravity_from_camera"],
        dtype=np.float64,
    )
    foreground_w2c = np.asarray(
        gravity["camera_alignment"]["foreground_w2c"],
        dtype=np.float64,
    )
    w2c_gravity = foreground_w2c @ np.linalg.inv(transform_gravity_from_camera)

    camera = json.loads(
        (scene_dir / "sam3d+fpose" / "vggt_single_image" / "camera.json").read_text(
            encoding="utf-8"
        )
    )
    intrinsic = np.asarray(camera["intrinsic_original_pixels"], dtype=np.float64)
    return w2c_gravity, intrinsic


def load_binary_mask(path: Path) -> np.ndarray:
    image = Image.open(path)
    if image.mode in {"RGBA", "LA"}:
        return np.asarray(image.getchannel("A")) > 0
    return np.asarray(image.convert("L")) > 0


def fit_support_front_edge(mask: np.ndarray) -> dict:
    components, count = label(mask)
    if count == 0:
        raise ValueError("Support mask is empty")
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    component = components == int(np.argmax(sizes))
    rows, cols = np.nonzero(component)
    x0, x1 = int(cols.min()), int(cols.max()) + 1
    y0, y1 = int(rows.min()), int(rows.max()) + 1
    box_height = max(y1 - y0, 1)
    low = int(round(y0 + 0.30 * box_height))
    high = int(round(min(y0 + 0.80 * box_height, 0.92 * mask.shape[0])))

    edge_x = []
    edge_y = []
    for x in range(x0, x1):
        candidates = np.flatnonzero(component[low:high, x])
        if len(candidates):
            edge_x.append(x)
            edge_y.append(low + int(candidates[-1]))
    x = np.asarray(edge_x, dtype=np.float64)
    y = np.asarray(edge_y, dtype=np.float64)
    if len(x) < max(32, int(0.2 * (x1 - x0))):
        raise ValueError("Could not extract a stable support front edge")

    for _ in range(6):
        line = np.polyfit(x, y, 1)
        residual = np.abs(y - np.polyval(line, x))
        threshold = max(2.0, float(np.percentile(residual, 70.0)))
        keep = residual <= threshold
        if int(keep.sum()) < 32 or bool(np.all(keep)):
            break
        x, y = x[keep], y[keep]
    line = np.polyfit(x, y, 1)
    return {
        "slope": float(line[0]),
        "intercept": float(line[1]),
        "angle_deg": float(np.degrees(np.arctan(line[0]))),
        "sample_count": int(len(x)),
        "x_span": [float(x.min()), float(x.max())],
        "component_pixels": int(component.sum()),
    }


def support_top_rectangle(
    mesh: trimesh.Trimesh,
    pose: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    world = mesh.copy()
    world.apply_transform(pose)
    centers = np.asarray(world.triangles_center, dtype=np.float64)
    normals = np.asarray(world.face_normals, dtype=np.float64)
    area = np.asarray(world.area_faces, dtype=np.float64)
    top = (
        (normals[:, 2] > 0.75)
        & (centers[:, 2] >= np.percentile(centers[:, 2], 75.0))
    )
    if int(top.sum()) < 8:
        raise ValueError("Could not identify the support top surface")

    vertex_ids = np.unique(np.asarray(world.faces)[top].reshape(-1))
    points = np.asarray(world.vertices, dtype=np.float64)[vertex_ids]
    xy = points[:, :2]
    center_xy = xy.mean(axis=0)
    covariance = np.cov((xy - center_xy).T)
    values, vectors = np.linalg.eigh(covariance)
    axes = vectors[:, np.argsort(values)[::-1]].T
    if np.linalg.det(axes) < 0.0:
        axes[1] *= -1.0
    local = (xy - center_xy) @ axes.T
    lower = local.min(axis=0)
    upper = local.max(axis=0)
    rectangle_local = np.array(
        [
            [lower[0], lower[1]],
            [upper[0], lower[1]],
            [upper[0], upper[1]],
            [lower[0], upper[1]],
        ],
        dtype=np.float64,
    )
    rectangle_xy = rectangle_local @ axes + center_xy
    top_z = float(np.median(points[:, 2]))
    rectangle = np.column_stack(
        [rectangle_xy, np.full(4, top_z, dtype=np.float64)]
    )
    normal = (normals[top] * area[top, None]).sum(axis=0)
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    tilt = float(np.degrees(np.arccos(np.clip(normal[2], -1.0, 1.0))))
    return rectangle, np.array([center_xy[0], center_xy[1], top_z]), normal, tilt


def project_world_points(
    points: np.ndarray,
    w2c: np.ndarray,
    intrinsic: np.ndarray,
) -> np.ndarray:
    camera = transform_points(points, w2c)
    if np.any(camera[:, 2] <= 1e-5):
        raise ValueError("Support edge moved behind the camera")
    return np.column_stack(
        [
            intrinsic[0, 0] * camera[:, 0] / camera[:, 2] + intrinsic[0, 2],
            intrinsic[1, 1] * camera[:, 1] / camera[:, 2] + intrinsic[1, 2],
        ]
    )


def planar_group_transform(
    center_xy: np.ndarray,
    yaw_deg: float,
    translation_xy: np.ndarray,
) -> np.ndarray:
    angle = np.deg2rad(yaw_deg)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:2, :2] = rotation
    transform[:2, 3] = (
        center_xy + translation_xy - rotation @ center_xy
    )
    return transform


def support_group_ids(root_id: int, edges: list[dict]) -> list[int]:
    group = {int(root_id)}
    changed = True
    while changed:
        changed = False
        for edge in edges:
            if edge.get("relation", "Support") != "Support":
                continue
            child = int(edge["source_id"])
            parent = int(edge["target_id"])
            if parent in group and child not in group:
                group.add(child)
                changed = True
    return sorted(group)


def support_front_edge(
    rectangle: np.ndarray,
    camera_xy: np.ndarray,
) -> np.ndarray:
    candidates = []
    for index in range(4):
        edge = rectangle[[index, (index + 1) % 4]]
        distance = float(
            np.linalg.norm(edge[:, :2].mean(axis=0) - camera_xy)
        )
        candidates.append((distance, edge))
    return min(candidates, key=lambda item: item[0])[1]


def optimize_support_group_transform(
    front_edge: np.ndarray,
    center_xy: np.ndarray,
    target_line: dict,
    w2c: np.ndarray,
    intrinsic: np.ndarray,
    image_width: int,
) -> dict:
    camera_xy = np.linalg.inv(w2c)[:2, 3]
    forward = camera_xy - center_xy
    forward /= max(float(np.linalg.norm(forward)), 1e-12)
    reference_x = 0.5 * float(image_width)
    target_y = (
        float(target_line["slope"]) * reference_x
        + float(target_line["intercept"])
    )

    def evaluate(yaw_deg: float, distance: float) -> dict | None:
        transform = planar_group_transform(
            center_xy,
            yaw_deg,
            distance * forward,
        )
        projected = project_world_points(
            transform_points(front_edge, transform),
            w2c,
            intrinsic,
        )
        delta_x = float(projected[1, 0] - projected[0, 0])
        if abs(delta_x) < 1e-6:
            return None
        slope = float(
            (projected[1, 1] - projected[0, 1]) / delta_x
        )
        intercept = float(projected[0, 1] - slope * projected[0, 0])
        y_at_reference = slope * reference_x + intercept
        score = (
            ((slope - float(target_line["slope"])) / 0.002) ** 2
            + ((y_at_reference - target_y) / 3.0) ** 2
            + 0.0004 * yaw_deg**2
            + 0.16 * (distance / 0.05) ** 2
        )
        return {
            "score": float(score),
            "yaw_delta_deg": float(yaw_deg),
            "forward_translation_m": float(distance),
            "slope": slope,
            "angle_deg": float(np.degrees(np.arctan(slope))),
            "intercept": intercept,
            "y_at_reference": float(y_at_reference),
            "front_uv": projected.tolist(),
            "transform": transform,
        }

    before = evaluate(0.0, 0.0)
    if before is None:
        raise RuntimeError("Invalid initial support projection")
    best = before
    for yaw in np.linspace(-12.0, 12.0, 97):
        for distance in np.linspace(-0.18, 0.18, 145):
            candidate = evaluate(float(yaw), float(distance))
            if candidate is not None and candidate["score"] < best["score"]:
                best = candidate

    coarse_yaw = best["yaw_delta_deg"]
    coarse_distance = best["forward_translation_m"]
    for yaw in np.linspace(coarse_yaw - 0.35, coarse_yaw + 0.35, 71):
        for distance in np.linspace(
            coarse_distance - 0.012,
            coarse_distance + 0.012,
            49,
        ):
            candidate = evaluate(float(yaw), float(distance))
            if candidate is not None and candidate["score"] < best["score"]:
                best = candidate

    best["applied"] = bool(best["score"] <= 0.8 * before["score"])
    best["before"] = {
        key: value
        for key, value in before.items()
        if key != "transform"
    }
    best["target_y_at_reference"] = float(target_y)
    best["forward_axis_xy"] = forward.tolist()
    return best


def stabilize_support_group_projection(
    scene_dir: Path,
    results_dir: Path,
    manifest: dict,
    output_path: Path,
    *,
    input_pose_name: str,
    output_pose_name: str = "pose_pre_sf_projected",
) -> dict:
    object_ids = [int(obj["id"]) for obj in manifest["objects"]]
    support_ids = [int(value) for value in manifest.get("support_object_ids", [])]
    graph = json.loads(Path(manifest["scene_graph_path"]).read_text(encoding="utf-8"))
    edges = graph.get("graph", graph).get("edges", [])
    report_path = results_dir.parent / "pre_sf_projection.json"

    try:
        if not support_ids:
            raise ValueError("No support root was detected")
        root_id = support_ids[0]
        root_dir = results_dir / f"obj_{root_id}"
        mesh = load_mesh(root_dir / "mesh_scaled.glb")
        root_pose = np.loadtxt(
            root_dir / f"{input_pose_name}.txt"
        ).astype(np.float64)
        mask = load_binary_mask(root_dir / "mask.png")
        target_line = fit_support_front_edge(mask)
        w2c, intrinsic = load_gravity_camera(scene_dir)
        rectangle, center, normal, tilt = support_top_rectangle(
            mesh,
            root_pose,
        )
        camera_xy = np.linalg.inv(w2c)[:2, 3]
        front_edge = support_front_edge(rectangle, camera_xy)
        optimization = optimize_support_group_transform(
            front_edge,
            center[:2],
            target_line,
            w2c,
            intrinsic,
            mask.shape[1],
        )
        group_ids = support_group_ids(root_id, edges)
        transform = (
            np.asarray(optimization["transform"], dtype=np.float64)
            if optimization["applied"]
            else np.eye(4, dtype=np.float64)
        )

        for object_id in object_ids:
            obj_dir = results_dir / f"obj_{object_id}"
            pose = np.loadtxt(
                obj_dir / f"{input_pose_name}.txt"
            ).astype(np.float64)
            if object_id in group_ids:
                pose = transform @ pose
            np.savetxt(
                obj_dir / f"{output_pose_name}.txt",
                pose,
                fmt="%.10g",
            )

        compose_stabilized_scene(
            results_dir,
            object_ids,
            output_path,
            "mesh_scaled",
            output_pose_name,
        )
        optimization.pop("transform", None)
        report = {
            "status": "ok",
            "applied": bool(optimization["applied"]),
            "support_root_id": root_id,
            "support_group_ids": group_ids,
            "input_pose_name": input_pose_name,
            "output_pose_name": output_pose_name,
            "target_front_edge": target_line,
            "support_top_normal": normal.tolist(),
            "support_top_tilt_deg": tilt,
            "optimization": optimization,
            "transform": transform.tolist(),
            "output": str(output_path),
        }
    except Exception as exc:
        for object_id in object_ids:
            obj_dir = results_dir / f"obj_{object_id}"
            pose = np.loadtxt(
                obj_dir / f"{input_pose_name}.txt"
            ).astype(np.float64)
            np.savetxt(
                obj_dir / f"{output_pose_name}.txt",
                pose,
                fmt="%.10g",
            )
        compose_stabilized_scene(
            results_dir,
            object_ids,
            output_path,
            "mesh_scaled",
            output_pose_name,
        )
        report = {
            "status": "skipped",
            "applied": False,
            "reason": str(exc),
            "input_pose_name": input_pose_name,
            "output_pose_name": output_pose_name,
            "output": str(output_path),
        }

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report"] = str(report_path)
    return report


def compose_stabilized_scene(
    results_dir: Path,
    object_ids: list[int],
    output_path: Path,
    mesh_name: str,
    pose_name: str,
) -> int:
    scene = trimesh.Scene()
    for object_id in object_ids:
        obj_dir = results_dir / f"obj_{object_id}"
        mesh = load_mesh(obj_dir / f"{mesh_name}.glb")
        pose = np.loadtxt(obj_dir / f"{pose_name}.txt").astype(np.float64)
        mesh.apply_transform(pose)
        scene.add_geometry(mesh, node_name=f"obj_{object_id}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(output_path)
    return len(object_ids)


def stabilize_scene_projection(
    scene_dir: Path,
    results_dir: Path,
    manifest: dict,
    output_path: Path,
    *,
    input_pose_name: str,
    optimized_pose_name: str,
    reference_pose_name: str | None = None,
    output_pose_name: str = "pose_projection_stabilized",
    output_mesh_name: str = "mesh_projection_stabilized",
    min_scale: float = 0.35,
    max_scale: float = 1.5,
    scale_samples: int = 81,
    min_reprojection_iou: float = 0.55,
    raw_pose_keep_ratio: float = 0.9,
) -> dict:
    w2c, intrinsic = load_gravity_camera(scene_dir)
    root_ids = {int(value) for value in manifest.get("root_object_ids", [])}
    graph = json.loads(Path(manifest["scene_graph_path"]).read_text(encoding="utf-8"))
    edges = graph.get("graph", graph).get("edges", [])
    supported_ids = {
        int(edge["source_id"])
        for edge in edges
        if edge.get("relation", "Support") == "Support"
    }

    records = []
    object_ids = [int(obj["id"]) for obj in manifest["objects"]]
    for object_id in object_ids:
        obj_dir = results_dir / f"obj_{object_id}"
        source_mesh_path = obj_dir / "mesh_scaled.glb"
        output_mesh_path = obj_dir / f"{output_mesh_name}.glb"
        output_pose_path = obj_dir / f"{output_pose_name}.txt"
        mask_path = obj_dir / "mask.png"
        mesh = load_mesh(source_mesh_path)
        visual_reference_name = input_pose_name
        if reference_pose_name is not None and object_id not in root_ids:
            reference_path = obj_dir / f"{reference_pose_name}.txt"
            if reference_path.exists():
                visual_reference_name = reference_pose_name
        initial_pose = np.loadtxt(
            obj_dir / f"{visual_reference_name}.txt"
        ).astype(np.float64)
        optimized_pose = np.loadtxt(obj_dir / f"{optimized_pose_name}.txt").astype(np.float64)
        sampled = trimesh.sample.sample_surface(mesh, 16000, seed=5100 + object_id)[0]
        mask_bbox, image_size = mask_bbox_xyxy(mask_path)
        width, height = image_size
        border_mask = bool(
            mask_bbox[0] <= 1.0
            or mask_bbox[1] <= 1.0
            or mask_bbox[2] >= width - 1.0
            or mask_bbox[3] >= height - 1.0
        )

        initial_iou = bbox_iou(
            mask_bbox,
            project_bbox_xyxy(sampled, initial_pose, w2c, intrinsic),
        )
        optimized_iou = bbox_iou(
            mask_bbox,
            project_bbox_xyxy(sampled, optimized_pose, w2c, intrinsic),
        )
        selected = "initial"
        selected_pose = initial_pose
        selected_mesh = mesh
        selected_iou = initial_iou
        scale = 1.0
        stabilization_error = None

        eligible = object_id in supported_ids and object_id not in root_ids and not border_mask
        if eligible:
            try:
                candidate_mesh, candidate_pose, candidate_scale, candidate_iou = stabilize_object(
                    mesh,
                    initial_pose,
                    optimized_pose,
                    mask_bbox,
                    w2c,
                    intrinsic,
                    object_id,
                    min_scale,
                    max_scale,
                    scale_samples,
                )
                if (
                    optimized_iou >= min_reprojection_iou
                    and optimized_iou >= 0.95 * initial_iou
                    and optimized_iou >= raw_pose_keep_ratio * candidate_iou
                ):
                    selected = "sf_optimized"
                    selected_pose = optimized_pose
                    selected_iou = optimized_iou
                elif (
                    candidate_iou >= min_reprojection_iou
                    and candidate_iou >= 0.95 * initial_iou
                ):
                    selected = "projection_stabilized"
                    selected_pose = candidate_pose
                    selected_mesh = candidate_mesh
                    selected_iou = candidate_iou
                    scale = candidate_scale
            except Exception as exc:
                stabilization_error = str(exc)

        if output_mesh_path.exists():
            output_mesh_path.unlink()
        if selected_mesh is mesh:
            shutil.copy2(source_mesh_path, output_mesh_path)
        else:
            selected_mesh.export(output_mesh_path)
        np.savetxt(output_pose_path, selected_pose, fmt="%.10g")

        records.append(
            {
                "object_id": object_id,
                "selected": selected,
                "is_root": object_id in root_ids,
                "is_supported": object_id in supported_ids,
                "border_mask": border_mask,
                "scale": float(scale),
                "visual_reference_pose_name": visual_reference_name,
                "initial_reprojection_iou": float(initial_iou),
                "sf_reprojection_iou": float(optimized_iou),
                "selected_reprojection_iou": float(selected_iou),
                "error": stabilization_error,
            }
        )

    count = compose_stabilized_scene(
        results_dir,
        object_ids,
        output_path,
        output_mesh_name,
        output_pose_name,
    )
    report = {
        "status": "ok",
        "output": str(output_path),
        "objects": records,
        "object_count": count,
        "root_object_ids": sorted(root_ids),
        "supported_object_ids": sorted(supported_ids),
        "mean_initial_reprojection_iou": float(
            np.mean([record["initial_reprojection_iou"] for record in records])
        ),
        "mean_sf_reprojection_iou": float(
            np.mean([record["sf_reprojection_iou"] for record in records])
        ),
        "mean_selected_reprojection_iou": float(
            np.mean([record["selected_reprojection_iou"] for record in records])
        ),
    }
    report_path = results_dir.parent / "projection_stabilization.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report"] = str(report_path)
    return report


def rotation_change_deg(before: np.ndarray, after: np.ndarray) -> float:
    delta = after[:3, :3] @ before[:3, :3].T
    cosine = float(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def support_contact_metrics(
    child_mesh: trimesh.Trimesh,
    child_pose: np.ndarray,
    parent_mesh: trimesh.Trimesh,
    parent_pose: np.ndarray,
) -> dict:
    child = transform_points(np.asarray(child_mesh.vertices, dtype=np.float64), child_pose)
    parent = transform_points(
        np.asarray(parent_mesh.vertices, dtype=np.float64),
        parent_pose,
    )
    child_min = child[:, :2].min(axis=0)
    child_max = child[:, :2].max(axis=0)
    parent_min = parent[:, :2].min(axis=0)
    parent_max = parent[:, :2].max(axis=0)
    intersection = np.maximum(
        0.0,
        np.minimum(child_max, parent_max) - np.maximum(child_min, parent_min),
    )
    child_area = float(np.prod(np.maximum(child_max - child_min, 1e-9)))
    overlap = float(np.prod(intersection) / child_area)

    parent_top = float(np.percentile(parent[:, 2], 99.0))
    margin = 0.01
    inside = np.logical_and(
        np.all(child[:, :2] >= parent_min - margin, axis=1),
        np.all(child[:, :2] <= parent_max + margin, axis=1),
    )
    contact_points = child[inside] if np.any(inside) else child
    contact_z = float(np.percentile(contact_points[:, 2], 2.0))
    return {
        "xy_overlap_ratio": overlap,
        "vertical_gap_m": contact_z - parent_top,
        "parent_top_z_m": parent_top,
        "child_contact_z_m": contact_z,
    }


def support_relation_valid(
    metrics: dict,
    *,
    min_overlap: float = 0.05,
    max_penetration_m: float = 0.015,
    max_floating_m: float = 0.03,
) -> bool:
    overlap = float(metrics["xy_overlap_ratio"])
    gap = float(metrics["vertical_gap_m"])
    return bool(
        overlap >= min_overlap
        and -max_penetration_m <= gap <= max_floating_m
    )


def support_relation_preserved(before: dict, after: dict) -> bool:
    if not support_relation_valid(after):
        return False
    minimum_overlap = max(
        0.05,
        0.65 * min(float(before["xy_overlap_ratio"]), 1.0),
    )
    overlap_ok = float(after["xy_overlap_ratio"]) >= minimum_overlap

    before_gap = float(before["vertical_gap_m"])
    after_gap = float(after["vertical_gap_m"])
    gap_error_limit = max(0.06, abs(before_gap) + 0.02)
    gap_ok = abs(after_gap) <= gap_error_limit
    penetration_ok = after_gap >= min(-0.06, before_gap - 0.02)
    floating_ok = after_gap <= max(0.08, before_gap + 0.03)
    return bool(overlap_ok and gap_ok and penetration_ok and floating_ok)


def validate_second_pass_projection(
    scene_dir: Path,
    results_dir: Path,
    manifest: dict,
    output_path: Path,
    *,
    input_pose_name: str,
    optimized_pose_name: str,
    output_pose_name: str = "pose_second_pass_validated",
    output_mesh_name: str = "mesh_second_pass_validated",
    min_reprojection_iou: float = 0.55,
    min_reprojection_ratio: float = 0.85,
) -> dict:
    w2c, intrinsic = load_gravity_camera(scene_dir)
    root_ids = {int(value) for value in manifest.get("root_object_ids", [])}
    object_ids = [int(obj["id"]) for obj in manifest["objects"]]
    graph = json.loads(Path(manifest["scene_graph_path"]).read_text(encoding="utf-8"))
    support_parent = {
        int(edge["source_id"]): int(edge["target_id"])
        for edge in graph.get("graph", graph).get("edges", [])
        if edge.get("relation", "Support") == "Support"
    }

    states = {}
    for object_id in object_ids:
        obj_dir = results_dir / f"obj_{object_id}"
        source_mesh = obj_dir / "mesh_scaled.glb"
        mesh = load_mesh(source_mesh)
        before = np.loadtxt(obj_dir / f"{input_pose_name}.txt").astype(np.float64)
        after = np.loadtxt(obj_dir / f"{optimized_pose_name}.txt").astype(np.float64)
        mask_bbox, _ = mask_bbox_xyxy(obj_dir / "mask.png")
        sampled = trimesh.sample.sample_surface(
            mesh,
            16000,
            seed=6100 + object_id,
        )[0]
        before_iou = bbox_iou(
            mask_bbox,
            project_bbox_xyxy(sampled, before, w2c, intrinsic),
        )
        after_iou = bbox_iou(
            mask_bbox,
            project_bbox_xyxy(sampled, after, w2c, intrinsic),
        )
        ratio = after_iou / max(before_iou, 1e-12)
        translation = float(np.linalg.norm(after[:3, 3] - before[:3, 3]))
        rotation = rotation_change_deg(before, after)
        if object_id in root_ids:
            accepted = translation <= 1e-5 and rotation <= 0.05
        else:
            accepted = (
                after_iou >= min_reprojection_iou
                and ratio >= min_reprojection_ratio
            )
        states[object_id] = {
            "obj_dir": obj_dir,
            "source_mesh": source_mesh,
            "mesh": mesh,
            "before": before,
            "after": after,
            "before_iou": float(before_iou),
            "after_iou": float(after_iou),
            "ratio": float(ratio),
            "translation": translation,
            "rotation": rotation,
            "accepted": bool(accepted),
            "reprojection_accepted": bool(accepted),
            "support_parent_id": support_parent.get(object_id),
            "support_before": None,
            "support_after": None,
            "support_before_valid": object_id not in support_parent,
            "support_after_valid": object_id not in support_parent,
            "support_preserved": object_id not in support_parent,
            "physical_override": False,
        }

    for _ in range(len(object_ids)):
        changed = False
        for child_id, parent_id in support_parent.items():
            if child_id not in states or parent_id not in states:
                continue
            child = states[child_id]
            parent = states[parent_id]
            parent_selected = (
                parent["after"] if parent["accepted"] else parent["before"]
            )
            before_metrics = support_contact_metrics(
                child["mesh"],
                child["before"],
                parent["mesh"],
                parent["before"],
            )
            after_metrics = support_contact_metrics(
                child["mesh"],
                child["after"],
                parent["mesh"],
                parent_selected,
            )
            before_valid = support_relation_valid(before_metrics)
            after_valid = support_relation_valid(after_metrics)
            preserved = support_relation_preserved(
                before_metrics,
                after_metrics,
            )
            physical_override = bool(not before_valid and after_valid)
            next_accepted = bool(
                (child["reprojection_accepted"] and preserved)
                or physical_override
            )
            child["support_before"] = before_metrics
            child["support_after"] = after_metrics
            child["support_before_valid"] = before_valid
            child["support_after_valid"] = after_valid
            child["support_preserved"] = (
                after_valid if next_accepted else before_valid
            )
            child["physical_override"] = bool(
                physical_override and next_accepted
            )
            if child["accepted"] != next_accepted:
                child["accepted"] = next_accepted
                changed = True
        if not changed:
            break

    records = []
    for object_id in object_ids:
        state = states[object_id]
        accepted = bool(state["accepted"])
        selected_pose = state["after"] if accepted else state["before"]
        selected = "sf_second_pass" if accepted else "projection_fallback"
        selected_support = (
            state["support_after"] if accepted else state["support_before"]
        )
        support_valid = bool(
            state["support_parent_id"] is None
            or (
                selected_support is not None
                and support_relation_valid(selected_support)
            )
        )
        shutil.copy2(
            state["source_mesh"],
            state["obj_dir"] / f"{output_mesh_name}.glb",
        )
        np.savetxt(
            state["obj_dir"] / f"{output_pose_name}.txt",
            selected_pose,
            fmt="%.10g",
        )
        records.append(
            {
                "object_id": object_id,
                "selected": selected,
                "is_root": object_id in root_ids,
                "support_parent_id": state["support_parent_id"],
                "support_preserved": support_valid,
                "support_before_valid": bool(state["support_before_valid"]),
                "support_after_valid": bool(state["support_after_valid"]),
                "support_before": state["support_before"],
                "support_after": state["support_after"],
                "support_selected": selected_support,
                "physical_override": bool(
                    state["physical_override"] and accepted
                ),
                "before_reprojection_iou": state["before_iou"],
                "sf_second_pass_reprojection_iou": state["after_iou"],
                "reprojection_ratio": state["ratio"],
                "translation_change_m": state["translation"],
                "rotation_change_deg": state["rotation"],
            }
        )

    compose_stabilized_scene(
        results_dir,
        object_ids,
        output_path,
        output_mesh_name,
        output_pose_name,
    )
    accepted_ids = [
        record["object_id"]
        for record in records
        if record["selected"] == "sf_second_pass"
    ]
    supported_records = [
        record for record in records if record["support_parent_id"] is not None
    ]
    report = {
        "status": "ok",
        "quality_gate": "reprojection_and_absolute_support_relation",
        "output": str(output_path),
        "object_count": len(records),
        "accepted_second_pass_ids": accepted_ids,
        "fallback_ids": [
            record["object_id"]
            for record in records
            if record["selected"] != "sf_second_pass"
        ],
        "support_relation_count": len(supported_records),
        "preserved_support_relation_count": sum(
            bool(record["support_preserved"]) for record in supported_records
        ),
        "mean_before_reprojection_iou": float(
            np.mean([record["before_reprojection_iou"] for record in records])
        ),
        "mean_second_pass_reprojection_iou": float(
            np.mean(
                [record["sf_second_pass_reprojection_iou"] for record in records]
            )
        ),
        "mean_final_reprojection_iou": float(
            np.mean(
                [
                    record["sf_second_pass_reprojection_iou"]
                    if record["selected"] == "sf_second_pass"
                    else record["before_reprojection_iou"]
                    for record in records
                ]
            )
        ),
        "records": records,
    }
    report_path = results_dir.parent / "second_pass_validation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report"] = str(report_path)
    return report
