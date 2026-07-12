#!/usr/bin/env python3
"""
Refine SAM3D object poses against VGGT single-image object point clouds.

The input object meshes are sam3d+fpose/scaled/{id}_z_up.glb, so SAM3D scale is
already baked into the mesh. This script keeps that scale fixed and only refines
SE(3) with ICP.

Coordinate notes:
  - scene_composed.glb is in SAM3D's final Y-up GLB frame.
  - Applying R_sam_to_vggt to that composed scene gives the VGGT camera frame:
        [x, y, z] -> [-x, z, y]
  - For the per-object z_up meshes used here, the Y-up -> Z-up part is already
    baked into the vertices. Therefore the initial object pose is:
        T_init_vggt = T_fix @ T_sam3d_zup_pose
    where T_fix = diag([-1, -1, 1, 1]).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")

import open3d as o3d
import trimesh
from PIL import Image


R_SAM_TO_VGGT = np.array(
    [
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)

T_YUP_TO_ZUP = np.eye(4, dtype=np.float64)
T_YUP_TO_ZUP[:3, :3] = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)

T_FIX_ZUP_SAM_TO_VGGT = np.diag([-1.0, -1.0, 1.0, 1.0]).astype(np.float64)
T_SAM_SCENE_TO_VGGT = T_FIX_ZUP_SAM_TO_VGGT @ T_YUP_TO_ZUP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene-dir",
        required=True,
        help="Scene directory containing image.png, sam3d, and sam3d+fpose/vggt_single_image.",
    )
    parser.add_argument(
        "--collection-dir",
        default=None,
        help="Directory for composed scene outputs. Default: <scene-dir>/sam3d_vggt_icp_outputs.",
    )
    parser.add_argument("--output-subdir", default="icp", help="Subdirectory under scene-dir for intermediate outputs.")
    parser.add_argument("--object-ids", nargs="*", type=int, default=None, help="Object ids to refine. Default: discover all.")
    parser.add_argument("--target-conf-min", type=float, default=1.0, help="Minimum VGGT depth_conf for target points.")
    parser.add_argument("--mask-alpha-threshold", type=int, default=0, help="Foreground threshold for RGBA alpha masks.")
    parser.add_argument("--mask-erode-px", type=int, default=0, help="Optional erosion on preprocessed object masks.")
    parser.add_argument("--sample-points", type=int, default=60000, help="Mesh surface points sampled per object.")
    parser.add_argument("--voxel-size", type=float, default=0.01, help="Voxel size in VGGT camera units for ICP.")
    parser.add_argument("--max-correspondence-distance", type=float, default=0.20, help="Fine ICP correspondence distance.")
    parser.add_argument("--coarse-factor", type=float, default=2.5, help="Multiplier for coarse voxel/correspondence distance.")
    parser.add_argument("--min-accepted-fitness", type=float, default=0.05, help="Reject ICP below this fine-stage fitness. <0 disables.")
    parser.add_argument("--max-accepted-rmse", type=float, default=0.15, help="Reject ICP above this fine-stage RMSE. <=0 disables.")
    parser.add_argument("--max-alignment-error-ratio", type=float, default=0.98, help="Require symmetric median error to improve by this ratio. <=0 disables.")
    parser.add_argument("--min-reprojection-iou-ratio", type=float, default=0.90, help="Reject ICP when projected mesh bbox IoU falls below this fraction of the initial IoU. <=0 disables.")
    parser.add_argument("--max-iterations", type=int, default=80, help="ICP iterations per stage.")
    parser.add_argument("--min-target-points", type=int, default=100, help="Skip ICP if an object has too few target points.")
    parser.add_argument("--outlier-std", type=float, default=2.0, help="Statistical outlier std ratio. <=0 disables filtering.")
    parser.add_argument("--max-accepted-rotation-deg", type=float, default=45.0, help="Reject ICP delta above this rotation. <=0 disables.")
    parser.add_argument("--max-accepted-translation", type=float, default=0.5, help="Reject ICP delta above this translation. <=0 disables.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scene-output-name", default=None, help="Default: {scene_name}_composed_ours.glb")
    return parser.parse_args()


def quaternion_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(quat)
    if norm <= 0:
        raise RuntimeError("Invalid zero quaternion")
    w, x, y, z = quat / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def load_sam3d_zup_pose(pose_path: Path) -> np.ndarray:
    with open(pose_path, "r", encoding="utf-8") as f:
        pose = json.load(f)
    R_row = quaternion_wxyz_to_matrix(np.asarray(pose["rotation"], dtype=np.float64))
    t = np.asarray(pose["translation"], dtype=np.float64).reshape(3)

    # SAM3D/PyTorch3D Transform3d uses row-vector points: p' = p @ R + t.
    # trimesh/Open3D matrices use column-vector convention, hence R.T.
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_row.T
    T[:3, 3] = t
    return T


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    return points @ T[:3, :3].T + T[:3, 3]


def pcd_from_points(points: np.ndarray, colors: np.ndarray | None = None) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    if colors is not None:
        colors = np.asarray(colors, dtype=np.float64)
        if colors.max(initial=0.0) > 1.0:
            colors = colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
    return pcd


def symmetric_median_distance(source: np.ndarray, target: np.ndarray) -> float:
    if len(source) == 0 or len(target) == 0:
        return float("inf")
    source_pcd = pcd_from_points(source)
    target_pcd = pcd_from_points(target)
    source_dist = np.asarray(source_pcd.compute_point_cloud_distance(target_pcd))
    target_dist = np.asarray(target_pcd.compute_point_cloud_distance(source_pcd))
    source_dist = source_dist[np.isfinite(source_dist)]
    target_dist = target_dist[np.isfinite(target_dist)]
    if len(source_dist) == 0 or len(target_dist) == 0:
        return float("inf")
    return float(0.5 * (np.median(source_dist) + np.median(target_dist)))


def finite_float_or_none(value: float | None) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def mask_bbox_xyxy(mask: np.ndarray) -> np.ndarray | None:
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        return None
    return np.array([cols.min(), rows.min(), cols.max() + 1, rows.max() + 1], dtype=np.float64)


def projected_bbox_xyxy(points: np.ndarray, intrinsic: np.ndarray) -> np.ndarray | None:
    points = np.asarray(points, dtype=np.float64)
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 1e-6)
    if not valid.any():
        return None
    points = points[valid]
    uv = np.column_stack(
        [
            intrinsic[0, 0] * points[:, 0] / points[:, 2] + intrinsic[0, 2],
            intrinsic[1, 1] * points[:, 1] / points[:, 2] + intrinsic[1, 2],
        ]
    )
    return np.array(
        [
            np.percentile(uv[:, 0], 1.0),
            np.percentile(uv[:, 1], 1.0),
            np.percentile(uv[:, 0], 99.0),
            np.percentile(uv[:, 1], 99.0),
        ],
        dtype=np.float64,
    )


def bbox_iou_xyxy(left: np.ndarray | None, right: np.ndarray | None) -> float | None:
    if left is None or right is None:
        return None
    intersection_size = np.maximum(0.0, np.minimum(left[2:], right[2:]) - np.maximum(left[:2], right[:2]))
    intersection = float(np.prod(intersection_size))
    left_area = float(np.prod(np.maximum(0.0, left[2:] - left[:2])))
    right_area = float(np.prod(np.maximum(0.0, right[2:] - right[:2])))
    union = left_area + right_area - intersection
    return intersection / union if union > 1e-12 else None


def bounds_dict(points: np.ndarray) -> dict:
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return {"min": None, "max": None, "center": None}
    return {
        "min": points.min(axis=0).tolist(),
        "max": points.max(axis=0).tolist(),
        "center": points.mean(axis=0).tolist(),
    }


def rotation_angle_deg(R: np.ndarray) -> float:
    value = (float(np.trace(R)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(value, -1.0, 1.0))))


def load_mesh_as_trimesh(mesh_path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(str(mesh_path), force="scene")
    if isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    elif isinstance(loaded, trimesh.Scene):
        if len(loaded.geometry) == 0:
            raise RuntimeError(f"Empty mesh scene: {mesh_path}")
        mesh = loaded.to_geometry()
    else:
        raise RuntimeError(f"Unsupported mesh type from {mesh_path}: {type(loaded)}")
    if len(mesh.faces) == 0:
        raise RuntimeError(f"Mesh has no faces: {mesh_path}")
    return mesh


def add_transformed_glb(scene: trimesh.Scene, mesh_path: Path, T: np.ndarray, node_prefix: str) -> int:
    loaded = trimesh.load(str(mesh_path), force="scene")
    count = 0
    if isinstance(loaded, trimesh.Trimesh):
        mesh = loaded.copy()
        mesh.apply_transform(T)
        scene.add_geometry(mesh, node_name=node_prefix)
        return 1

    if not isinstance(loaded, trimesh.Scene):
        raise RuntimeError(f"Unsupported mesh type from {mesh_path}: {type(loaded)}")

    for node_name in loaded.graph.nodes_geometry:
        geom_name = loaded.graph[node_name][1]
        mesh = loaded.geometry[geom_name].copy()
        T_node = loaded.graph.get(node_name)[0]
        if T_node is not None:
            mesh.apply_transform(T_node)
        mesh.apply_transform(T)
        scene.add_geometry(mesh, node_name=f"{node_prefix}_{node_name}")
        count += 1
    return count


def load_foreground_mask(mask_path: Path, alpha_threshold: int) -> np.ndarray:
    image = Image.open(mask_path).convert("RGBA")
    arr = np.asarray(image)
    alpha = arr[:, :, 3]
    if alpha.max() > alpha.min():
        return alpha > alpha_threshold
    rgb = arr[:, :, :3]
    return np.any(rgb > 0, axis=2)


def paste_original_mask_to_vggt(mask_orig: np.ndarray, transform: dict, out_hw: tuple[int, int]) -> np.ndarray:
    out_h, out_w = out_hw
    resized_w = int(transform["resized_width"])
    resized_h = int(transform["resized_height"])
    offset_x = int(round(transform["offset_x"]))
    offset_y = int(round(transform["offset_y"]))

    resized = Image.fromarray((mask_orig.astype(np.uint8) * 255)).resize(
        (resized_w, resized_h),
        Image.Resampling.NEAREST,
    )
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


def maybe_erode_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    if pixels <= 0:
        return mask
    from scipy.ndimage import binary_erosion

    return binary_erosion(mask, iterations=int(pixels))


def save_mask_debug(mask_process_dir: Path, object_id: int, mask_orig: np.ndarray, mask_pre: np.ndarray, image_pre: np.ndarray) -> None:
    mask_process_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask_orig.astype(np.uint8) * 255)).save(mask_process_dir / f"{object_id}_mask_orig_binary.png")
    Image.fromarray((mask_pre.astype(np.uint8) * 255)).save(mask_process_dir / f"{object_id}_mask_preprocessed.png")

    overlay = image_pre.copy()
    red = np.zeros_like(overlay)
    red[:, :, 0] = 255
    alpha = mask_pre[:, :, None].astype(np.float32) * 0.45
    overlay = (overlay.astype(np.float32) * (1.0 - alpha) + red.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)
    Image.fromarray(overlay).save(mask_process_dir / f"{object_id}_mask_overlay_preprocessed.png")


def sample_mesh_points(mesh: trimesh.Trimesh, count: int, seed: int) -> np.ndarray:
    np.random.seed(seed)
    count = max(int(count), 1000)
    return mesh.sample(count)


def downsample_for_icp(pcd: o3d.geometry.PointCloud, voxel_size: float) -> o3d.geometry.PointCloud:
    if voxel_size <= 0:
        return pcd
    return pcd.voxel_down_sample(voxel_size=float(voxel_size))


def run_fixed_scale_icp(
    source_init: np.ndarray,
    target: np.ndarray,
    voxel_size: float,
    max_corr: float,
    coarse_factor: float,
    max_iterations: int,
) -> tuple[np.ndarray, list[dict]]:
    source_pcd = pcd_from_points(source_init)
    target_pcd = pcd_from_points(target)

    delta = np.eye(4, dtype=np.float64)
    stages = [
        (voxel_size * coarse_factor, max_corr * coarse_factor, max(20, max_iterations // 2), "coarse"),
        (voxel_size, max_corr, max_iterations, "fine"),
    ]
    logs: list[dict] = []
    estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint(False)

    for voxel, corr, iterations, label in stages:
        src_down = downsample_for_icp(source_pcd, voxel)
        tgt_down = downsample_for_icp(target_pcd, voxel)
        if len(src_down.points) < 10 or len(tgt_down.points) < 10:
            logs.append(
                {
                    "stage": label,
                    "skipped": True,
                    "source_points": len(src_down.points),
                    "target_points": len(tgt_down.points),
                }
            )
            continue
        reg = o3d.pipelines.registration.registration_icp(
            src_down,
            tgt_down,
            float(corr),
            delta,
            estimation,
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(iterations)),
        )
        delta = np.asarray(reg.transformation, dtype=np.float64)
        logs.append(
            {
                "stage": label,
                "skipped": False,
                "voxel_size": float(voxel),
                "max_correspondence_distance": float(corr),
                "source_points": len(src_down.points),
                "target_points": len(tgt_down.points),
                "fitness": float(reg.fitness),
                "inlier_rmse": float(reg.inlier_rmse),
                "transformation": delta.tolist(),
            }
        )
    return delta, logs


def discover_object_ids(sam3d_dir: Path, mesh_dir: Path) -> list[int]:
    ids = []
    for pose_path in sam3d_dir.glob("*_pose.json"):
        stem = pose_path.name[: -len("_pose.json")]
        if stem.isdigit() and (mesh_dir / f"{stem}_z_up.glb").exists() and (sam3d_dir / f"{stem}.png").exists():
            ids.append(int(stem))
    return sorted(set(ids))


def main() -> None:
    args = parse_args()
    scene_dir = Path(args.scene_dir)
    collection_dir = Path(args.collection_dir) if args.collection_dir else (scene_dir / "sam3d_vggt_icp_outputs")
    sam3d_dir = scene_dir / "sam3d"
    mesh_dir = scene_dir / "sam3d+fpose" / "scaled"
    vggt_dir = scene_dir / "sam3d+fpose" / "vggt_single_image"
    out_dir = scene_dir / args.output_subdir
    mask_process_dir = out_dir / "mask_process"
    out_dir.mkdir(parents=True, exist_ok=True)
    collection_dir.mkdir(parents=True, exist_ok=True)

    with open(vggt_dir / "camera.json", "r", encoding="utf-8") as f:
        camera = json.load(f)
    transform = camera["preprocess_transform_original_to_preprocessed"]
    pre_h, pre_w = camera["preprocessed_size_hw"]
    intrinsic_original = np.asarray(camera["intrinsic_original_pixels"], dtype=np.float64)

    points_world = np.load(vggt_dir / "points_world.npy")
    depth = np.load(vggt_dir / "depth.npy")
    depth_conf = np.load(vggt_dir / "depth_conf.npy")
    content_mask = np.load(vggt_dir / "content_mask.npy").astype(bool)
    image_pre = np.asarray(Image.open(vggt_dir / "image_preprocessed.png").convert("RGB"))

    object_ids = args.object_ids or discover_object_ids(sam3d_dir, mesh_dir)
    if not object_ids:
        raise RuntimeError(f"No objects found in {sam3d_dir} with meshes in {mesh_dir}")

    report = {
        "scene": scene_dir.name,
        "scene_dir": str(scene_dir),
        "coordinate_note": {
            "R_sam_to_vggt": R_SAM_TO_VGGT.tolist(),
            "T_yup_to_zup": T_YUP_TO_ZUP.tolist(),
            "T_fix_zup_sam_to_vggt": T_FIX_ZUP_SAM_TO_VGGT.tolist(),
            "T_sam_scene_to_vggt": T_SAM_SCENE_TO_VGGT.tolist(),
            "per_object_init": "T_init_vggt = T_fix_zup_sam_to_vggt @ T_sam3d_zup_pose because *_z_up.glb already has Y-up -> Z-up baked into vertices.",
        },
        "vggt_dir": str(vggt_dir),
        "mask_preprocess_transform": transform,
        "target_conf_min": float(args.target_conf_min),
        "mask_erode_px": int(args.mask_erode_px),
        "sample_points": int(args.sample_points),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "voxel_size": float(args.voxel_size),
        "max_correspondence_distance": float(args.max_correspondence_distance),
        "objects": [],
        "min_accepted_fitness": float(args.min_accepted_fitness),
        "max_accepted_rmse": float(args.max_accepted_rmse),
        "max_alignment_error_ratio": float(args.max_alignment_error_ratio),
        "min_reprojection_iou_ratio": float(args.min_reprojection_iou_ratio),
    }

    composed = trimesh.Scene()

    for object_id in object_ids:
        pose_path = sam3d_dir / f"{object_id}_pose.json"
        mask_path = sam3d_dir / f"{object_id}.png"
        mesh_path = mesh_dir / f"{object_id}_z_up.glb"

        mask_orig = load_foreground_mask(mask_path, args.mask_alpha_threshold)
        mask_pre_raw = paste_original_mask_to_vggt(mask_orig, transform, (pre_h, pre_w))
        mask_pre = maybe_erode_mask(mask_pre_raw, args.mask_erode_px)
        save_mask_debug(mask_process_dir, object_id, mask_orig, mask_pre, image_pre)

        valid = (
            mask_pre
            & content_mask
            & np.isfinite(points_world).all(axis=2)
            & np.isfinite(depth)
            & (depth > 0)
            & np.isfinite(depth_conf)
            & (depth_conf >= float(args.target_conf_min))
        )
        target_points = points_world[valid].astype(np.float64)
        target_colors = image_pre[valid].astype(np.float64) / 255.0
        target_pcd = pcd_from_points(target_points, target_colors)
        target_ply = out_dir / f"{object_id}_target_vggt_masked.ply"
        o3d.io.write_point_cloud(str(target_ply), target_pcd, write_ascii=True)

        target_for_icp = target_pcd
        filtered_target_ply = None
        if args.outlier_std > 0 and len(target_for_icp.points) >= 30:
            target_for_icp, keep_indices = target_for_icp.remove_statistical_outlier(
                nb_neighbors=20,
                std_ratio=float(args.outlier_std),
            )
            filtered_target_ply = out_dir / f"{object_id}_target_vggt_masked_filtered.ply"
            o3d.io.write_point_cloud(str(filtered_target_ply), target_for_icp, write_ascii=True)

        mesh = load_mesh_as_trimesh(mesh_path)
        source_local = sample_mesh_points(mesh, args.sample_points, args.seed + object_id)

        T_sam3d_zup = load_sam3d_zup_pose(pose_path)
        T_init = T_FIX_ZUP_SAM_TO_VGGT @ T_sam3d_zup
        source_init = transform_points(source_local, T_init)

        np.savetxt(out_dir / f"{object_id}_pose_vggt_init.txt", T_init, fmt="%.10g")
        o3d.io.write_point_cloud(str(out_dir / f"{object_id}_source_init_sample.ply"), pcd_from_points(source_init), write_ascii=True)

        target_array = np.asarray(target_for_icp.points)
        if len(target_points) < args.min_target_points or len(target_for_icp.points) < args.min_target_points:
            delta = np.eye(4, dtype=np.float64)
            icp_logs = [
                {
                    "skipped": True,
                    "reason": "too_few_target_points",
                    "target_points_raw": int(len(target_points)),
                    "target_points_filtered": int(len(target_for_icp.points)),
                }
            ]
        else:
            delta_raw, icp_logs = run_fixed_scale_icp(
                source_init=source_init,
                target=target_array,
                voxel_size=float(args.voxel_size),
                max_corr=float(args.max_correspondence_distance),
                coarse_factor=float(args.coarse_factor),
                max_iterations=int(args.max_iterations),
            )

            delta = delta_raw

        raw_delta = delta.copy()
        T_refined_raw = raw_delta @ T_init
        source_refined_raw = transform_points(source_local, T_refined_raw)
        alignment_error_initial = symmetric_median_distance(source_init, target_array)
        alignment_error_raw = symmetric_median_distance(source_refined_raw, target_array)
        alignment_error_ratio = (
            alignment_error_raw / alignment_error_initial
            if np.isfinite(alignment_error_initial) and alignment_error_initial > 1e-12
            else None
        )
        mask_bbox = mask_bbox_xyxy(mask_orig)
        reprojection_bbox_initial = projected_bbox_xyxy(source_init, intrinsic_original)
        reprojection_bbox_raw = projected_bbox_xyxy(source_refined_raw, intrinsic_original)
        reprojection_iou_initial = bbox_iou_xyxy(mask_bbox, reprojection_bbox_initial)
        reprojection_iou_raw = bbox_iou_xyxy(mask_bbox, reprojection_bbox_raw)
        reprojection_iou_ratio = (
            reprojection_iou_raw / reprojection_iou_initial
            if reprojection_iou_initial is not None and reprojection_iou_initial > 1e-12
            else None
        )
        delta_rotation_deg = rotation_angle_deg(raw_delta[:3, :3])
        delta_translation = float(np.linalg.norm(raw_delta[:3, 3]))
        reject_reasons = []
        last_log = icp_logs[-1]
        if last_log.get("skipped"):
            reject_reasons.append(f"ICP skipped: {last_log.get('reason', 'insufficient correspondences')}")
        else:
            fine_fitness = float(last_log.get("fitness", 0.0))
            fine_rmse = float(last_log.get("inlier_rmse", float("inf")))
            if args.min_accepted_fitness >= 0 and fine_fitness < args.min_accepted_fitness:
                reject_reasons.append(
                    f"fitness {fine_fitness:.6f} < {float(args.min_accepted_fitness):.6f}"
                )
            if args.max_accepted_rmse > 0 and fine_rmse > args.max_accepted_rmse:
                reject_reasons.append(
                    f"rmse {fine_rmse:.6f} > {float(args.max_accepted_rmse):.6f}"
                )
        if args.max_alignment_error_ratio > 0 and (
            alignment_error_ratio is None or alignment_error_ratio > args.max_alignment_error_ratio
        ):
            ratio_text = "undefined" if alignment_error_ratio is None else f"{alignment_error_ratio:.6f}"
            reject_reasons.append(
                f"alignment error ratio {ratio_text} > {float(args.max_alignment_error_ratio):.6f}"
            )
        if (
            args.min_reprojection_iou_ratio > 0
            and reprojection_iou_initial is not None
            and reprojection_iou_initial >= 0.05
            and (
                reprojection_iou_raw is None
                or reprojection_iou_raw
                < args.min_reprojection_iou_ratio * reprojection_iou_initial
            )
        ):
            raw_text = "undefined" if reprojection_iou_raw is None else f"{reprojection_iou_raw:.6f}"
            reject_reasons.append(
                f"reprojection IoU {raw_text} < "
                f"{float(args.min_reprojection_iou_ratio):.6f} * initial "
                f"{reprojection_iou_initial:.6f}"
            )
        if args.max_accepted_rotation_deg > 0 and delta_rotation_deg > args.max_accepted_rotation_deg:
            reject_reasons.append(
                f"rotation {delta_rotation_deg:.3f} > {float(args.max_accepted_rotation_deg):.3f} deg"
            )
        if args.max_accepted_translation > 0 and delta_translation > args.max_accepted_translation:
            reject_reasons.append(
                f"translation {delta_translation:.6f} > {float(args.max_accepted_translation):.6f}"
            )
        icp_accepted = len(reject_reasons) == 0
        delta = raw_delta if icp_accepted else np.eye(4, dtype=np.float64)

        T_refined = delta @ T_init
        source_refined = source_refined_raw if icp_accepted else source_init
        alignment_error_final = alignment_error_raw if icp_accepted else alignment_error_initial
        reprojection_iou_final = reprojection_iou_raw if icp_accepted else reprojection_iou_initial
        np.savetxt(out_dir / f"{object_id}_pose_vggt_icp_raw.txt", T_refined_raw, fmt="%.10g")
        np.savetxt(out_dir / f"{object_id}_icp_delta_raw.txt", raw_delta, fmt="%.10g")
        np.savetxt(out_dir / f"{object_id}_pose_vggt_icp.txt", T_refined, fmt="%.10g")
        np.savetxt(out_dir / f"{object_id}_icp_delta.txt", delta, fmt="%.10g")
        o3d.io.write_point_cloud(str(out_dir / f"{object_id}_source_icp_sample.ply"), pcd_from_points(source_refined), write_ascii=True)

        add_transformed_glb(composed, mesh_path, T_refined, f"object_{object_id:02d}")

        report["objects"].append(
            {
                "object_id": int(object_id),
                "pose_json": str(pose_path),
                "mesh": str(mesh_path),
                "mask": str(mask_path),
                "mask_orig_pixels": int(mask_orig.sum()),
                "mask_pre_pixels_raw": int(mask_pre_raw.sum()),
                "mask_pre_pixels_used": int(mask_pre.sum()),
                "target_points_raw": int(len(target_points)),
                "target_points_filtered": int(len(target_for_icp.points)),
                "target_bounds": bounds_dict(target_points),
                "source_init_bounds": bounds_dict(source_init),
                "source_icp_bounds": bounds_dict(source_refined),
                "target_ply": str(target_ply),
                "target_filtered_ply": None if filtered_target_ply is None else str(filtered_target_ply),
                "pose_init": str(out_dir / f"{object_id}_pose_vggt_init.txt"),
                "pose_icp_raw": str(out_dir / f"{object_id}_pose_vggt_icp_raw.txt"),
                "pose_icp": str(out_dir / f"{object_id}_pose_vggt_icp.txt"),
                "icp_delta_raw": str(out_dir / f"{object_id}_icp_delta_raw.txt"),
                "icp_delta": str(out_dir / f"{object_id}_icp_delta.txt"),
                "icp_delta_rotation_deg": float(delta_rotation_deg),
                "icp_delta_translation": float(delta_translation),
                "alignment_error_initial": finite_float_or_none(alignment_error_initial),
                "alignment_error_raw": finite_float_or_none(alignment_error_raw),
                "alignment_error_final": finite_float_or_none(alignment_error_final),
                "alignment_error_ratio": finite_float_or_none(alignment_error_ratio),
                "mask_bbox_xyxy": None if mask_bbox is None else mask_bbox.tolist(),
                "reprojection_bbox_initial_xyxy": (
                    None if reprojection_bbox_initial is None else reprojection_bbox_initial.tolist()
                ),
                "reprojection_bbox_raw_xyxy": (
                    None if reprojection_bbox_raw is None else reprojection_bbox_raw.tolist()
                ),
                "reprojection_iou_initial": finite_float_or_none(reprojection_iou_initial),
                "reprojection_iou_raw": finite_float_or_none(reprojection_iou_raw),
                "reprojection_iou_final": finite_float_or_none(reprojection_iou_final),
                "reprojection_iou_ratio": finite_float_or_none(reprojection_iou_ratio),
                "icp_accepted": bool(icp_accepted),
                "icp_reject_reasons": reject_reasons,
                "icp_logs": icp_logs,
            }
        )
        last_log = icp_logs[-1]
        if last_log.get("skipped"):
            print(f"object {object_id}: skipped ICP, target={len(target_points)}")
        else:
            suffix = "accepted" if icp_accepted else "rejected: " + "; ".join(reject_reasons)
            print(
                f"object {object_id}: target={len(target_points)}, "
                f"filtered={len(target_for_icp.points)}, "
                f"fitness={last_log['fitness']:.4f}, rmse={last_log['inlier_rmse']:.4f}, "
                f"delta_t={delta_translation:.4f}, delta_R={delta_rotation_deg:.2f}deg, {suffix}"
            )

    scene_output_name = args.scene_output_name or f"{scene_dir.name}_composed_ours.glb"
    scene_out = collection_dir / scene_output_name
    composed.export(str(scene_out))
    report["scene_output"] = str(scene_out)

    report_path = out_dir / "icp_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"wrote scene: {scene_out}")
    print(f"wrote report: {report_path}")


if __name__ == "__main__":
    main()
