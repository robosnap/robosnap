#!/usr/bin/env python3
"""Gravity-align foreground/background assets from a single-image scene."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


PLY_TO_DTYPE = {
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "uchar": "u1",
    "uint8": "u1",
    "char": "i1",
    "int8": "i1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
}


def normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        raise ValueError(f"Cannot normalize near-zero vector: {vec}")
    return vec / norm


def rotation_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = normalize(axis)
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    cc = 1.0 - c
    return np.array(
        [
            [c + x * x * cc, x * y * cc - z * s, x * z * cc + y * s],
            [y * x * cc + z * s, c + y * y * cc, y * z * cc - x * s],
            [z * x * cc - y * s, z * y * cc + x * s, c + z * z * cc],
        ],
        dtype=np.float64,
    )


def rotation_between(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src = normalize(src)
    dst = normalize(dst)
    cross = np.cross(src, dst)
    dot = float(np.clip(np.dot(src, dst), -1.0, 1.0))
    norm_cross = float(np.linalg.norm(cross))
    if norm_cross < 1e-12:
        if dot > 0.0:
            return np.eye(3, dtype=np.float64)
        helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(src, helper))) > 0.9:
            helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        return rotation_axis_angle(np.cross(src, helper), math.pi)
    return rotation_axis_angle(cross / norm_cross, math.atan2(norm_cross, dot))


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    return points @ T[:3, :3].T + T[:3, 3]


def pad_extrinsic(extrinsic: np.ndarray) -> np.ndarray:
    extrinsic = np.asarray(extrinsic, dtype=np.float64)
    if extrinsic.shape == (4, 4):
        return extrinsic
    if extrinsic.shape != (3, 4):
        raise ValueError(f"Expected a 3x4 or 4x4 extrinsic, got {extrinsic.shape}")
    padded = np.eye(4, dtype=np.float64)
    padded[:3, :4] = extrinsic
    return padded


def rotation_matrix_to_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = np.array([0.25 * scale, (m[2, 1] - m[1, 2]) / scale, (m[0, 2] - m[2, 0]) / scale, (m[1, 0] - m[0, 1]) / scale])
    else:
        idx = int(np.argmax(np.diag(m)))
        if idx == 0:
            scale = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            quat = np.array([(m[2, 1] - m[1, 2]) / scale, 0.25 * scale, (m[0, 1] + m[1, 0]) / scale, (m[0, 2] + m[2, 0]) / scale])
        elif idx == 1:
            scale = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            quat = np.array([(m[0, 2] - m[2, 0]) / scale, (m[0, 1] + m[1, 0]) / scale, 0.25 * scale, (m[1, 2] + m[2, 1]) / scale])
        else:
            scale = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            quat = np.array([(m[1, 0] - m[0, 1]) / scale, (m[0, 2] + m[2, 0]) / scale, (m[1, 2] + m[2, 1]) / scale, 0.25 * scale])
    return quat / np.linalg.norm(quat)


def left_multiply_quaternions_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    right = np.asarray(right, dtype=np.float64).reshape(-1, 4)
    lw, lx, ly, lz = np.asarray(left, dtype=np.float64)
    rw, rx, ry, rz = right.T
    result = np.column_stack(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ]
    )
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    return result / np.clip(norms, 1e-12, None)


def load_mask_original(mask_path: Path, original_size_wh: tuple[int, int]) -> np.ndarray:
    mask = Image.open(mask_path).convert("RGBA")
    if mask.size != original_size_wh:
        mask = mask.resize(original_size_wh, Image.Resampling.NEAREST)
    arr = np.asarray(mask)
    alpha = arr[:, :, 3]
    if alpha.max() > alpha.min():
        return alpha > 0
    return np.any(arr[:, :, :3] > 0, axis=2)


def load_mask_native(mask_path: Path) -> np.ndarray:
    image = Image.open(mask_path)
    if image.mode in {"RGBA", "LA"}:
        return np.asarray(image.getchannel("A")) > 0
    return np.asarray(image.convert("L")) > 0


def find_support_object_id(scene_dir: Path, support_mask_path: Path | None) -> tuple[int | None, dict[str, Any]]:
    if support_mask_path is None or not support_mask_path.exists():
        return None, {"status": "missing_support_mask"}
    support_mask = load_mask_native(support_mask_path)
    candidates = []
    for path in sorted((scene_dir / "sam3d").glob("*.png")):
        try:
            object_id = int(path.stem)
        except ValueError:
            continue
        object_mask = load_mask_native(path)
        if object_mask.shape != support_mask.shape or not object_mask.any():
            continue
        coverage = float(np.logical_and(object_mask, support_mask).sum() / object_mask.sum())
        candidates.append((coverage, object_id, str(path)))
    if not candidates:
        return None, {"status": "no_numeric_masks"}
    coverage, object_id, mask_path = max(candidates)
    return object_id, {
        "status": "ok" if coverage >= 0.5 else "low_coverage",
        "object_id": int(object_id),
        "mask": mask_path,
        "support_mask_coverage": float(coverage),
    }


def estimate_support_mesh_normal(
    scene_dir: Path,
    object_id: int,
    reference_normal: np.ndarray,
    max_angle_deg: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import trimesh

    report_path = scene_dir / "depth" / "object_point_clouds" / "icp_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    record = next(
        (item for item in report.get("objects", []) if int(item["object_id"]) == object_id),
        None,
    )
    if record is None:
        raise KeyError(f"Object {object_id} is missing from {report_path}")
    mesh_path = Path(record["mesh"])
    pose_path = Path(record["pose_icp"])
    loaded = trimesh.load(mesh_path, force="scene", process=False)
    mesh = loaded.to_geometry() if isinstance(loaded, trimesh.Scene) else loaded
    pose = np.loadtxt(pose_path).astype(np.float64)

    linear = pose[:3, :3]
    normal_matrix = np.linalg.inv(linear).T
    normals = np.asarray(mesh.face_normals, dtype=np.float64) @ normal_matrix.T
    normals /= np.clip(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12, None)
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    reference = normalize(reference_normal)
    dots = normals @ reference
    keep = np.abs(dots) >= math.cos(math.radians(max_angle_deg))
    if int(keep.sum()) < 8 or float(areas[keep].sum()) <= 1e-12:
        raise ValueError(f"Too few support-aligned faces for object {object_id}")

    selected = normals[keep].copy()
    selected_dots = dots[keep]
    selected[selected_dots < 0.0] *= -1.0
    selected_areas = areas[keep]
    normal = normalize((selected * selected_areas[:, None]).sum(axis=0))
    centers = transform_points(
        np.asarray(mesh.triangles_center, dtype=np.float64),
        pose,
    )[keep]
    offsets = centers @ normal
    order = np.argsort(offsets)
    sorted_offsets = offsets[order]
    sorted_areas = selected_areas[order]
    cumulative = np.cumsum(sorted_areas) / float(sorted_areas.sum())
    top_offset = float(sorted_offsets[np.searchsorted(cumulative, 0.9)])
    mean_center = np.average(centers, axis=0, weights=selected_areas)
    plane_point = mean_center + normal * (
        top_offset - float(np.dot(mean_center, normal))
    )
    angular_residuals = np.degrees(
        np.arccos(np.clip(selected @ normal, -1.0, 1.0))
    )
    return normal, plane_point, {
        "status": "ok",
        "object_id": int(object_id),
        "mesh": str(mesh_path),
        "pose": str(pose_path),
        "selected_faces": int(keep.sum()),
        "selected_area": float(selected_areas.sum()),
        "median_angular_residual_deg": float(np.median(angular_residuals)),
        "normal_camera": normal.tolist(),
        "top_plane_offset_camera": top_offset,
        "top_plane_point_camera": plane_point.tolist(),
    }


def paste_original_mask_to_vggt(mask_orig: np.ndarray, transform: dict[str, Any], out_hw: tuple[int, int]) -> np.ndarray:
    out_h, out_w = out_hw
    resized_w = int(transform["resized_width"])
    resized_h = int(transform["resized_height"])
    offset_x = int(round(transform["offset_x"]))
    offset_y = int(round(transform["offset_y"]))
    resized = Image.fromarray((mask_orig.astype(np.uint8) * 255)).resize((resized_w, resized_h), Image.Resampling.NEAREST)
    resized_np = np.asarray(resized) > 0

    canvas = np.zeros((out_h, out_w), dtype=bool)
    src_x0 = max(0, -offset_x)
    src_y0 = max(0, -offset_y)
    dst_x0 = max(0, offset_x)
    dst_y0 = max(0, offset_y)
    width = min(resized_w - src_x0, out_w - dst_x0)
    height = min(resized_h - src_y0, out_h - dst_y0)
    if width > 0 and height > 0:
        canvas[dst_y0 : dst_y0 + height, dst_x0 : dst_x0 + width] = resized_np[
            src_y0 : src_y0 + height,
            src_x0 : src_x0 + width,
        ]
    return canvas


def load_plane_points(vggt_dir: Path, plane_mask: Path | None, max_points: int, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    camera = json.loads((vggt_dir / "camera.json").read_text(encoding="utf-8"))
    points_world = np.load(vggt_dir / "points_world.npy")
    depth = np.load(vggt_dir / "depth.npy")
    depth_conf_path = vggt_dir / "depth_conf.npy"
    depth_conf = np.load(depth_conf_path) if depth_conf_path.exists() else np.ones_like(depth)
    content_mask = np.load(vggt_dir / "content_mask.npy").astype(bool)
    valid = content_mask & np.isfinite(points_world).all(axis=2) & np.isfinite(depth) & (depth > 0) & np.isfinite(depth_conf)

    mask_used = None
    if plane_mask and plane_mask.exists():
        original_size = tuple(camera["original_size_wh"])
        mask_orig = load_mask_original(plane_mask, original_size)
        mask_used = paste_original_mask_to_vggt(
            mask_orig,
            camera["preprocess_transform_original_to_preprocessed"],
            tuple(camera["preprocessed_size_hw"]),
        )
        valid &= mask_used

    points = points_world[valid].astype(np.float64)
    if len(points) > max_points:
        rng = np.random.default_rng(seed)
        points = points[rng.choice(len(points), size=max_points, replace=False)]
    meta = {
        "vggt_dir": str(vggt_dir),
        "plane_mask": None if plane_mask is None else str(plane_mask),
        "used_mask": mask_used is not None,
        "valid_points": int(len(points)),
    }
    return points, meta


def ransac_plane(points: np.ndarray, iterations: int, threshold: float, seed: int) -> tuple[np.ndarray, float, np.ndarray]:
    if len(points) < 3:
        raise ValueError("Need at least 3 points for plane fitting")
    rng = np.random.default_rng(seed)
    best_inliers = np.zeros(len(points), dtype=bool)
    best_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    best_d = 0.0
    for _ in range(max(iterations, 1)):
        ids = rng.choice(len(points), size=3, replace=False)
        a, b, c = points[ids]
        normal = np.cross(b - a, c - a)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal = normal / norm
        d = -float(np.dot(normal, a))
        dist = np.abs(points @ normal + d)
        inliers = dist < threshold
        if int(inliers.sum()) > int(best_inliers.sum()):
            best_inliers = inliers
            best_normal = normal
            best_d = d
    if best_inliers.sum() >= 3:
        inlier_points = points[best_inliers]
        centroid = inlier_points.mean(axis=0)
        _, _, vh = np.linalg.svd(inlier_points - centroid, full_matrices=False)
        normal = normalize(vh[-1])
        d = -float(np.dot(normal, centroid))
        best_normal, best_d = normal, d
    return best_normal, best_d, best_inliers


def apply_transform_to_glb(input_path: Path, output_path: Path, T: np.ndarray) -> None:
    import trimesh

    scene = trimesh.load(str(input_path), force="scene")
    scene.apply_transform(T)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(output_path))


def parse_ply_header(path: Path) -> tuple[list[bytes], str, int, list[tuple[str, str]], int]:
    header: list[bytes] = []
    fmt = ""
    vertex_count = 0
    vertex_props: list[tuple[str, str]] = []
    in_vertex = False
    offset = 0
    with path.open("rb") as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"PLY header ended unexpectedly: {path}")
            header.append(line)
            offset += len(line)
            text = line.decode("ascii").strip()
            if text.startswith("format "):
                fmt = text.split()[1]
            elif text.startswith("element "):
                parts = text.split()
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
            elif in_vertex and text.startswith("property "):
                parts = text.split()
                if len(parts) == 3:
                    vertex_props.append((parts[2], parts[1]))
            elif text == "end_header":
                break
    return header, fmt, vertex_count, vertex_props, offset


def transform_ply_xyz(input_path: Path, output_path: Path, T: np.ndarray) -> None:
    header, fmt, vertex_count, props, offset = parse_ply_header(input_path)
    prop_names = [name for name, _ in props]
    if not {"x", "y", "z"}.issubset(prop_names):
        raise ValueError(f"PLY has no x/y/z vertex properties: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "binary_little_endian":
        dtype = np.dtype([(name, PLY_TO_DTYPE[prop_type]) for name, prop_type in props])
        with input_path.open("rb") as f:
            f.seek(offset)
            vertices = np.fromfile(f, dtype=dtype, count=vertex_count)
            rest = f.read()
        xyz = np.column_stack([vertices["x"], vertices["y"], vertices["z"]]).astype(np.float64)
        xyz_t = transform_points(xyz, T)
        vertices["x"] = xyz_t[:, 0]
        vertices["y"] = xyz_t[:, 1]
        vertices["z"] = xyz_t[:, 2]
        rot_names = ["rot_0", "rot_1", "rot_2", "rot_3"]
        if all(name in prop_names for name in rot_names):
            rotations = np.column_stack([vertices[name] for name in rot_names])
            rotated = left_multiply_quaternions_wxyz(
                rotation_matrix_to_quaternion_wxyz(T[:3, :3]),
                rotations,
            )
            for idx, name in enumerate(rot_names):
                vertices[name] = rotated[:, idx]
        with output_path.open("wb") as f:
            f.writelines(header)
            vertices.tofile(f)
            f.write(rest)
        return

    if fmt == "ascii":
        x_i, y_i, z_i = prop_names.index("x"), prop_names.index("y"), prop_names.index("z")
        with input_path.open("rb") as f:
            f.seek(offset)
            lines = f.read().decode("ascii").splitlines()
        out_lines = []
        xyz_values = []
        rot_names = ["rot_0", "rot_1", "rot_2", "rot_3"]
        has_rotations = all(name in prop_names for name in rot_names)
        rot_indices = [prop_names.index(name) for name in rot_names] if has_rotations else []
        rotation_values = []
        for line in lines[:vertex_count]:
            parts = line.split()
            xyz_values.append([float(parts[x_i]), float(parts[y_i]), float(parts[z_i])])
            if has_rotations:
                rotation_values.append([float(parts[idx]) for idx in rot_indices])
        xyz_t = transform_points(np.asarray(xyz_values, dtype=np.float64), T)
        rotated = None
        if has_rotations:
            rotated = left_multiply_quaternions_wxyz(
                rotation_matrix_to_quaternion_wxyz(T[:3, :3]),
                np.asarray(rotation_values),
            )
        for idx, line in enumerate(lines[:vertex_count]):
            parts = line.split()
            parts[x_i], parts[y_i], parts[z_i] = (f"{value:.10g}" for value in xyz_t[idx])
            if rotated is not None:
                for quat_idx, prop_idx in enumerate(rot_indices):
                    parts[prop_idx] = f"{rotated[idx, quat_idx]:.10g}"
            out_lines.append(" ".join(parts))
        out_lines.extend(lines[vertex_count:])
        with output_path.open("wb") as f:
            f.writelines(header)
            f.write(("\n".join(out_lines) + "\n").encode("ascii"))
        return

    raise ValueError(f"Unsupported PLY format: {fmt}")


def copy_or_transform_background(input_path: Path | None, output_path: Path, T: np.ndarray) -> str:
    if input_path is None or not input_path.exists():
        return "missing"
    transform_ply_xyz(input_path, output_path, T)
    return "transformed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit a support plane and export gravity-aligned foreground/background.")
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--foreground-glb", type=Path, required=True)
    parser.add_argument("--background-ply", type=Path)
    parser.add_argument("--background-cameras", type=Path)
    parser.add_argument("--vggt-dir", type=Path, required=True)
    parser.add_argument("--plane-mask", type=Path)
    parser.add_argument("--output-foreground", type=Path, required=True)
    parser.add_argument("--output-background", type=Path, required=True)
    parser.add_argument("--transform-json", type=Path, required=True)
    parser.add_argument("--target-up", nargs=3, type=float, default=[0.0, 0.0, 1.0])
    parser.add_argument("--normal-sign-reference", nargs=3, type=float, default=[0.0, -1.0, 0.0])
    parser.add_argument("--ransac-threshold", type=float, default=0.025)
    parser.add_argument("--ransac-iterations", type=int, default=2000)
    parser.add_argument("--max-plane-points", type=int, default=200000)
    parser.add_argument("--support-mesh-weight", type=float, default=0.95)
    parser.add_argument("--support-mesh-max-angle-deg", type=float, default=45.0)
    parser.add_argument("--disable-support-mesh-gravity", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    points, point_meta = load_plane_points(
        args.vggt_dir.expanduser().resolve(),
        args.plane_mask.expanduser().resolve() if args.plane_mask else None,
        args.max_plane_points,
        args.seed,
    )
    if len(points) < 3:
        T = np.eye(4, dtype=np.float64)
        normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        center = np.zeros(3, dtype=np.float64)
        inlier_count = 0
        status = "identity_no_plane_points"
    else:
        normal, plane_d, inliers = ransac_plane(points, args.ransac_iterations, args.ransac_threshold, args.seed)
        ref = normalize(np.asarray(args.normal_sign_reference, dtype=np.float64))
        if float(np.dot(normal, ref)) < 0.0:
            normal = -normal
            plane_d = -plane_d
        vggt_normal = normal.copy()
        support_object_id, support_meta = find_support_object_id(
            args.scene_dir,
            args.plane_mask,
        )
        support_mesh_meta: dict[str, Any] = {"status": "disabled"}
        support_plane_point = None
        if (
            not args.disable_support_mesh_gravity
            and support_object_id is not None
            and 0.0 < args.support_mesh_weight <= 1.0
        ):
            try:
                support_normal, support_plane_point, support_mesh_meta = estimate_support_mesh_normal(
                    args.scene_dir,
                    support_object_id,
                    vggt_normal,
                    args.support_mesh_max_angle_deg,
                )
                if float(np.dot(support_normal, vggt_normal)) < 0.0:
                    support_normal = -support_normal
                weight = float(args.support_mesh_weight)
                normal = normalize(weight * support_normal + (1.0 - weight) * vggt_normal)
                support_mesh_meta["weight"] = weight
                support_mesh_meta["fused_normal_camera"] = normal.tolist()
                support_mesh_meta["vggt_to_mesh_angle_deg"] = float(
                    np.degrees(
                        np.arccos(
                            np.clip(np.dot(vggt_normal, support_normal), -1.0, 1.0)
                        )
                    )
                )
            except Exception as exc:
                support_mesh_meta = {"status": "failed", "reason": str(exc)}
        inlier_points = points[inliers] if inliers.any() else points
        vggt_center = inlier_points.mean(axis=0)
        center = vggt_center.copy()
        if support_plane_point is not None:
            center = center + normal * (
                float(np.dot(normal, support_plane_point))
                - float(np.dot(normal, center))
            )
            support_mesh_meta["gravity_origin_center_camera"] = center.tolist()
        R = rotation_between(normal, np.asarray(args.target_up, dtype=np.float64))
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = -(R @ center)
        inlier_count = int(inliers.sum())
        status = "ok"

    apply_transform_to_glb(args.foreground_glb, args.output_foreground, T)

    foreground_camera = pad_extrinsic(np.load(args.vggt_dir / "extrinsic.npy"))
    background_transform = T.copy()
    camera_alignment = {
        "status": "identity_same_camera_world",
        "foreground_w2c": foreground_camera.tolist(),
    }
    if args.background_cameras and args.background_cameras.exists():
        cameras = np.load(args.background_cameras)
        if "w2c_render" not in cameras:
            raise KeyError(f"w2c_render is missing from {args.background_cameras}")
        background_camera = pad_extrinsic(cameras["w2c_render"][0])
        foreground_from_background = np.linalg.inv(foreground_camera) @ background_camera
        background_transform = T @ foreground_from_background
        camera_alignment = {
            "status": "aligned_first_camera",
            "foreground_w2c": foreground_camera.tolist(),
            "background_w2c": background_camera.tolist(),
            "transform_foreground_from_background": foreground_from_background.tolist(),
        }

    background_status = copy_or_transform_background(
        args.background_ply,
        args.output_background,
        background_transform,
    )
    camera_origin_foreground = np.linalg.inv(foreground_camera) @ np.array([0.0, 0.0, 0.0, 1.0])
    camera_origin_gravity = (T @ camera_origin_foreground)[:3]
    meta = {
        "status": status,
        "scene_dir": str(args.scene_dir),
        "foreground_glb": str(args.foreground_glb),
        "background_ply": None if args.background_ply is None else str(args.background_ply),
        "output_foreground": str(args.output_foreground),
        "output_background": str(args.output_background),
        "background_status": background_status,
        "point_selection": point_meta,
        "plane_normal_camera": normal.tolist(),
        "plane_normal_vggt_camera": (
            vggt_normal.tolist() if len(points) >= 3 else normal.tolist()
        ),
        "plane_center_camera": center.tolist(),
        "plane_center_vggt_camera": (
            vggt_center.tolist() if len(points) >= 3 else center.tolist()
        ),
        "inlier_count": inlier_count,
        "target_up": args.target_up,
        "transform_gravity_from_camera": T.tolist(),
        "transform_gravity_from_background": background_transform.tolist(),
        "background_cameras": None if args.background_cameras is None else str(args.background_cameras),
        "camera_alignment": camera_alignment,
        "camera_origin_in_gravity": camera_origin_gravity.tolist(),
        "support_object": (
            support_meta if len(points) >= 3 else {"status": "not_evaluated"}
        ),
        "support_mesh_gravity": (
            support_mesh_meta if len(points) >= 3 else {"status": "not_evaluated"}
        ),
    }
    args.transform_json.parent.mkdir(parents=True, exist_ok=True)
    args.transform_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[gravity-align] wrote {args.output_foreground}")
    print(f"[gravity-align] wrote {args.transform_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
