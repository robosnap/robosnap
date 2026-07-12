#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compose object poses with a VGGT single-image point cloud.

For this scene_droid pipeline, FoundationPose is optimized in the OpenCV camera
frame and VGGT single-image point_cloud.ply is used as the same camera-at-origin
frame. The default is therefore:

    T_o2scene = T_o2c

VGGT also writes a camera-from-world extrinsic. That frame is a VGGT gauge, and
is optional here via --scene_frame vggt_world:

    T_o2scene = inv(T_w2c) @ T_o2c

SAM3D pose json is also supported. Batch mode still loads
sam3d+fpose/scaled/{id}_z_up.glb by default, so SAM3D scale is assumed to be
already baked into the mesh and only rotation + translation are applied. This
zup_then_pose mode is algebraically equivalent to SAM3D's own
scene_composed.glb followed by a global Y-up -> Z-up conversion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


T_YUP_TO_ZUP = np.eye(4, dtype=np.float64)
T_YUP_TO_ZUP[:3, :3] = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


def load_T_txt(path: str | Path) -> np.ndarray:
    T = np.loadtxt(path).astype(np.float64)
    if T.shape != (4, 4):
        raise RuntimeError(f"Pose must be 4x4, got {T.shape}: {path}")
    return T


def as_homogeneous(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    if T.shape == (4, 4):
        return T
    if T.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :] = T
        return out
    raise RuntimeError(f"Expected 3x4 or 4x4 transform, got {T.shape}")


def load_extrinsic(path: str | Path | None) -> np.ndarray:
    if path is None:
        return np.eye(4, dtype=np.float64)

    path = Path(path)
    if path.suffix == ".npy":
        T = np.load(path)
    else:
        T = np.loadtxt(path)
    return as_homogeneous(T)


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


def load_sam3d_pose_json(path: str | Path, apply_scale: bool = False) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        pose = json.load(f)

    quat = np.asarray(pose["rotation"], dtype=np.float64).reshape(4)
    trans = np.asarray(pose["translation"], dtype=np.float64).reshape(3)
    R_row = quaternion_wxyz_to_matrix(quat)

    if apply_scale:
        scale = np.asarray(pose["scale"], dtype=np.float64).reshape(3)
    else:
        scale = np.ones(3, dtype=np.float64)

    # SAM3D/PyTorch3D Transform3d applies row-vector points as:
    #     p' = (p * scale) @ R + t
    # trimesh.apply_transform uses column-vector matrices, so the equivalent
    # top-left block is (diag(scale) @ R).T.
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = (np.diag(scale) @ R_row).T
    T[:3, 3] = trans
    return T


def load_object_pose(
    path: str | Path,
    pose_source: str,
    sam3d_apply_scale: bool,
) -> np.ndarray:
    if pose_source == "foundationpose":
        return load_T_txt(path)
    if pose_source == "sam3d":
        return load_sam3d_pose_json(path, apply_scale=sam3d_apply_scale)
    raise RuntimeError(f"Unsupported pose source: {pose_source}")


def load_mesh_trimesh(mesh_path: str | Path):
    import trimesh

    tm = trimesh.load(str(mesh_path), force="scene")
    if isinstance(tm, trimesh.Scene):
        if len(tm.geometry) == 0:
            raise RuntimeError(f"Scene contains no geometry: {mesh_path}")
        tm = tm.dump(concatenate=True)

    if not isinstance(tm, trimesh.Trimesh):
        raise RuntimeError(f"Loaded mesh is not Trimesh: {mesh_path}")
    if len(tm.faces) == 0:
        raise RuntimeError(f"Mesh has no faces: {mesh_path}")
    return tm


def load_scene_cloud(scene_ply: str | Path, grey: float):
    import open3d as o3d

    scene = o3d.io.read_point_cloud(str(scene_ply))
    if scene.is_empty():
        raise RuntimeError(f"Scene point cloud empty: {scene_ply}")
    g = float(grey)
    scene.paint_uniform_color([g, g, g])
    return scene


def write_matrix(path: Path, T: np.ndarray) -> None:
    np.savetxt(path, T, fmt="%.10g")


def object_color(index: int) -> list[float]:
    palette = [
        [1.0, 0.0, 0.0],
        [0.0, 0.45, 1.0],
        [0.0, 0.75, 0.2],
        [1.0, 0.55, 0.0],
        [0.75, 0.1, 1.0],
        [1.0, 0.0, 0.55],
        [0.1, 0.8, 0.8],
        [0.55, 0.35, 0.0],
    ]
    return palette[index % len(palette)]


def export_one(
    *,
    object_id: int,
    pose_path: Path,
    pose_source: str,
    sam3d_apply_scale: bool,
    sam3d_compose_order: str,
    mesh_path: Path,
    scene,
    T_scene_from_camera: np.ndarray,
    out_prefix: Path,
    sample_points: int,
    export_combined_ply: bool,
) -> dict:
    import open3d as o3d

    T_o2c = load_object_pose(pose_path, pose_source, sam3d_apply_scale)
    if pose_source == "sam3d" and sam3d_compose_order == "pose_then_zup":
        T_mesh_to_camera = T_YUP_TO_ZUP @ T_o2c
    else:
        T_mesh_to_camera = T_o2c
    T_o2scene = T_scene_from_camera @ T_mesh_to_camera

    tm = load_mesh_trimesh(mesh_path)
    bounds_before = tm.bounds.copy()
    tm.apply_transform(T_o2scene)
    bounds_after = tm.bounds.copy()

    mesh_out = out_prefix.with_suffix(".glb")
    tm.export(str(mesh_out))

    mesh_ply_out = out_prefix.with_name(out_prefix.name + "_mesh.ply")
    tm.export(str(mesh_ply_out))

    T_out = out_prefix.with_name(out_prefix.name + "_T_o2scene.txt")
    write_matrix(T_out, T_o2scene)

    combined_out = None
    if export_combined_ply:
        n = max(int(sample_points), 1000)
        obj_pts = o3d.geometry.PointCloud()
        obj_pts.points = o3d.utility.Vector3dVector(tm.sample(n))
        obj_pts.paint_uniform_color(object_color(object_id))

        combined = scene + obj_pts
        combined_out = out_prefix.with_name(out_prefix.name + "_combined.ply")
        o3d.io.write_point_cloud(str(combined_out), combined, write_ascii=True)

    return {
        "object_id": int(object_id),
        "pose_source": pose_source,
        "pose": str(pose_path),
        "sam3d_apply_scale": bool(sam3d_apply_scale),
        "sam3d_compose_order": sam3d_compose_order,
        "mesh": str(mesh_path),
        "mesh_note": (
            "zup_then_pose uses sam3d+fpose/scaled/{id}_z_up.glb: scale is baked, "
            "then SAM3D rotation+translation are applied. This matches "
            "scene_composed.glb after global Y-up -> Z-up conversion. "
            "pose_then_zup is a debug mode using {id}_scaled.glb first."
        ),
        "mesh_is_z_up_glb": mesh_path.name.endswith("_z_up.glb"),
        "mesh_is_scaled_glb": mesh_path.name.endswith("_scaled.glb"),
        "mesh_out": str(mesh_out),
        "mesh_ply_out": str(mesh_ply_out),
        "T_o2scene": str(T_out),
        "combined_ply": None if combined_out is None else str(combined_out),
        "bounds_before": bounds_before.tolist(),
        "bounds_after": bounds_after.tolist(),
    }


def discover_foundationpose_object_ids(foundationpose_dir: Path, mesh_dir: Path) -> list[int]:
    ids = []
    for pose in foundationpose_dir.glob("*_fpose_zup/ob_in_cam/000000.txt"):
        name = pose.parents[1].name
        if not name.endswith("_fpose_zup"):
            continue
        stem = name[: -len("_fpose_zup")]
        if stem.isdigit() and (mesh_dir / f"{stem}_z_up.glb").exists():
            ids.append(int(stem))
    return sorted(set(ids))


def discover_sam3d_object_ids(sam3d_dir: Path, mesh_dir: Path, mesh_suffix: str) -> list[int]:
    ids = []
    for pose in sam3d_dir.glob("*_pose.json"):
        stem = pose.name[: -len("_pose.json")]
        if stem.isdigit() and (mesh_dir / f"{stem}{mesh_suffix}").exists():
            ids.append(int(stem))
    return sorted(set(ids))


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_ply", required=True, help="VGGT point_cloud.ply")
    ap.add_argument("--vggt_extrinsic", default=None, help="VGGT extrinsic.npy/txt, camera-from-world [R|t]")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument(
        "--scene_frame",
        choices=["camera", "vggt_world"],
        default="camera",
        help="camera: use FoundationPose object->camera directly. vggt_world: apply inv(VGGT camera-from-world extrinsic).",
    )
    ap.add_argument(
        "--output_label",
        default=None,
        help="Suffix used in output names. Default: vggt_camera or vggt_world based on --scene_frame.",
    )

    ap.add_argument("--pose_source", choices=["foundationpose", "sam3d"], default="foundationpose")
    ap.add_argument("--foundationpose_dir", help="Directory containing {i}_fpose_zup/ob_in_cam/000000.txt")
    ap.add_argument("--sam3d_dir", help="Directory containing {i}_pose.json")
    ap.add_argument("--mesh_dir", help="Directory containing {i}_z_up.glb meshes")
    ap.add_argument("--object_ids", nargs="*", type=int, help="Object ids to export. Default: discover all.")

    ap.add_argument("--ob_in_cam_txt", help="Single-object FoundationPose pose")
    ap.add_argument("--sam3d_pose_json", help="Single-object SAM3D {id}_pose.json")
    ap.add_argument("--mesh", help="Single-object mesh")
    ap.add_argument("--object_id", type=int, default=0)
    ap.add_argument(
        "--sam3d_apply_scale",
        action="store_true",
        help="Apply SAM3D pose scale. Keep false when using sam3d+fpose/scaled/{id}_z_up.glb.",
    )
    ap.add_argument(
        "--sam3d_compose_order",
        choices=["zup_then_pose", "pose_then_zup"],
        default="zup_then_pose",
        help=(
            "zup_then_pose: use {id}_z_up.glb and apply SAM3D R,t; this matches "
            "SAM3D scene_composed.glb after global Y-up -> Z-up. "
            "pose_then_zup: debug mode using {id}_scaled.glb, then global Y-up -> Z-up."
        ),
    )

    ap.add_argument("--export_combined_ply", action="store_true")
    ap.add_argument("--sample_points", type=int, default=50000)
    ap.add_argument("--scene_grey", type=float, default=0.7)
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import open3d as o3d
    import trimesh

    T_w2c = load_extrinsic(args.vggt_extrinsic)
    if args.scene_frame == "camera":
        T_scene_from_camera = np.eye(4, dtype=np.float64)
    else:
        T_scene_from_camera = np.linalg.inv(T_w2c)

    output_label = args.output_label
    if output_label is None:
        frame_label = "vggt_world" if args.scene_frame == "vggt_world" else "vggt_camera"
        output_label = f"{frame_label}_{args.pose_source}"
    scene = load_scene_cloud(args.scene_ply, args.scene_grey)
    scene_out = out_dir / "vggt_scene_grey.ply"
    o3d.io.write_point_cloud(str(scene_out), scene, write_ascii=True)

    jobs: list[tuple[int, Path, Path, Path]] = []
    if args.mesh_dir and args.pose_source == "foundationpose" and args.foundationpose_dir:
        foundationpose_dir = Path(args.foundationpose_dir)
        mesh_dir = Path(args.mesh_dir)
        object_ids = args.object_ids or discover_foundationpose_object_ids(foundationpose_dir, mesh_dir)
        for object_id in object_ids:
            jobs.append(
                (
                    object_id,
                    foundationpose_dir / f"{object_id}_fpose_zup" / "ob_in_cam" / "000000.txt",
                    mesh_dir / f"{object_id}_z_up.glb",
                    out_dir / f"object_{object_id:02d}_{output_label}",
                )
            )
    elif args.mesh_dir and args.pose_source == "sam3d" and args.sam3d_dir:
        sam3d_dir = Path(args.sam3d_dir)
        mesh_dir = Path(args.mesh_dir)
        mesh_suffix = "_scaled.glb" if args.sam3d_compose_order == "pose_then_zup" else "_z_up.glb"
        object_ids = args.object_ids or discover_sam3d_object_ids(sam3d_dir, mesh_dir, mesh_suffix)
        for object_id in object_ids:
            jobs.append(
                (
                    object_id,
                    sam3d_dir / f"{object_id}_pose.json",
                    mesh_dir / f"{object_id}{mesh_suffix}",
                    out_dir / f"object_{object_id:02d}_{output_label}",
                )
            )
    elif args.mesh and args.pose_source == "foundationpose" and args.ob_in_cam_txt:
        jobs.append((args.object_id, Path(args.ob_in_cam_txt), Path(args.mesh), out_dir / f"object_{args.object_id:02d}_{output_label}"))
    elif args.mesh and args.pose_source == "sam3d" and args.sam3d_pose_json:
        jobs.append((args.object_id, Path(args.sam3d_pose_json), Path(args.mesh), out_dir / f"object_{args.object_id:02d}_{output_label}"))
    else:
        raise RuntimeError("Use batch args (--mesh_dir plus --foundationpose_dir/--sam3d_dir) or single-object args.")

    if not jobs:
        raise RuntimeError("No object jobs found")

    all_scene = trimesh.Scene()
    all_samples = scene
    report = {
        "scene_ply": str(args.scene_ply),
        "scene_grey_ply": str(scene_out),
        "scene_frame": args.scene_frame,
        "output_label": output_label,
        "pose_source": args.pose_source,
        "sam3d_apply_scale": bool(args.sam3d_apply_scale),
        "sam3d_compose_order": args.sam3d_compose_order,
        "vggt_extrinsic": args.vggt_extrinsic,
        "T_w2c": T_w2c.tolist(),
        "T_scene_from_camera": T_scene_from_camera.tolist(),
        "T_yup_to_zup": T_YUP_TO_ZUP.tolist(),
        "mesh_source_note": (
            "For --pose_source sam3d, zup_then_pose loads mesh_dir/{object_id}_z_up.glb. "
            "Those files are produced from {object_id}_scaled.glb, so scale is already "
            "applied before the y-up to z-up conversion. This output is equivalent to "
            "SAM3D scene_composed.glb followed by a global Y-up -> Z-up transform. "
            "Keep --sam3d_apply_scale false for scaled meshes."
        ),
        "objects": [],
    }

    for object_id, pose_path, mesh_path, out_prefix in jobs:
        if not pose_path.exists():
            raise FileNotFoundError(pose_path)
        if not mesh_path.exists():
            raise FileNotFoundError(mesh_path)

        item = export_one(
            object_id=object_id,
            pose_path=pose_path,
            pose_source=args.pose_source,
            sam3d_apply_scale=args.sam3d_apply_scale,
            sam3d_compose_order=args.sam3d_compose_order,
            mesh_path=mesh_path,
            scene=scene,
            T_scene_from_camera=T_scene_from_camera,
            out_prefix=out_prefix,
            sample_points=args.sample_points,
            export_combined_ply=args.export_combined_ply,
        )
        report["objects"].append(item)

        tm = load_mesh_trimesh(mesh_path)
        tm.apply_transform(load_T_txt(Path(item["T_o2scene"])))
        all_scene.add_geometry(tm, node_name=f"object_{object_id:02d}")

        if args.export_combined_ply:
            obj_pts = o3d.geometry.PointCloud()
            obj_pts.points = o3d.utility.Vector3dVector(tm.sample(max(int(args.sample_points), 1000)))
            obj_pts.paint_uniform_color(object_color(object_id))
            all_samples = all_samples + obj_pts

        print(f"object {object_id}: {item['mesh_out']}")

    all_glb = out_dir / f"objects_{output_label}.glb"
    all_scene.export(str(all_glb))
    report["objects_glb"] = str(all_glb)

    if args.export_combined_ply:
        all_combined = out_dir / f"objects_{output_label}_combined.ply"
        o3d.io.write_point_cloud(str(all_combined), all_samples, write_ascii=True)
        report["objects_combined_ply"] = str(all_combined)

    report_path = out_dir / "compose_scene_vggt_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nExported:")
    print("scene cloud :", scene_out)
    print("objects glb :", all_glb)
    if args.export_combined_ply:
        print("combined    :", report["objects_combined_ply"])
    print("report      :", report_path)


if __name__ == "__main__":
    main()
