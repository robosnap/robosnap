#!/usr/bin/env python3
"""Prepare scaled and z-up SAM3D meshes for VGGT/ICP alignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


T_YUP_TO_ZUP = np.eye(4, dtype=np.float64)
T_YUP_TO_ZUP[:3, :3] = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


def load_scene(path: Path) -> trimesh.Scene:
    scene = trimesh.load(str(path), force="scene")
    if isinstance(scene, trimesh.Trimesh):
        out = trimesh.Scene()
        out.add_geometry(scene)
        return out
    if not isinstance(scene, trimesh.Scene):
        raise RuntimeError(f"Unsupported mesh type from {path}: {type(scene)}")
    if len(scene.geometry) == 0:
        raise RuntimeError(f"Empty mesh scene: {path}")
    return scene


def load_scale(path: Path) -> np.ndarray:
    data = json.loads(path.read_text(encoding="utf-8"))
    scale = np.asarray(data.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64).reshape(-1)
    if scale.size == 1:
        scale = np.repeat(scale.item(), 3)
    if scale.size != 3:
        raise ValueError(f"Expected scalar or 3-vector scale in {path}, got {scale}")
    return scale


def discover_ids(sam3d_dir: Path) -> list[int]:
    ids = []
    for pose_path in sam3d_dir.glob("*_pose.json"):
        stem = pose_path.name[: -len("_pose.json")]
        if stem.isdigit() and (sam3d_dir / f"{stem}.glb").exists():
            ids.append(int(stem))
    return sorted(set(ids))


def prepare_one(sam3d_dir: Path, scaled_dir: Path, object_id: int, overwrite: bool) -> dict:
    mask_png = sam3d_dir / f"{object_id}.png"
    src_glb = sam3d_dir / f"{object_id}.glb"
    pose_json = sam3d_dir / f"{object_id}_pose.json"
    scaled_glb = scaled_dir / f"{object_id}_scaled.glb"
    zup_glb = scaled_dir / f"{object_id}_z_up.glb"

    if not mask_png.exists():
        raise FileNotFoundError(mask_png)
    if not src_glb.exists():
        raise FileNotFoundError(src_glb)
    if not pose_json.exists():
        raise FileNotFoundError(pose_json)

    scale = load_scale(pose_json)
    if overwrite or not scaled_glb.exists():
        scene = load_scene(src_glb)
        scale_mat = np.eye(4, dtype=np.float64)
        scale_mat[:3, :3] = np.diag(scale)
        scene.apply_transform(scale_mat)
        scene.export(str(scaled_glb))

    if overwrite or not zup_glb.exists():
        scene = load_scene(scaled_glb)
        scene.apply_transform(T_YUP_TO_ZUP)
        scene.export(str(zup_glb))

    return {
        "object_id": object_id,
        "mask_png": str(mask_png),
        "source_glb": str(src_glb),
        "pose_json": str(pose_json),
        "scale": scale.tolist(),
        "scaled_glb": str(scaled_glb),
        "z_up_glb": str(zup_glb),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create sam3d+fpose/scaled/*_scaled.glb and *_z_up.glb from SAM3D output.")
    parser.add_argument("--scene-dir", type=Path, required=True, help="Directory containing sam3d/ and image.png.")
    parser.add_argument("--sam3d-dir", type=Path, help="Default: <scene-dir>/sam3d")
    parser.add_argument("--scaled-dir", type=Path, help="Default: <scene-dir>/sam3d+fpose/scaled")
    parser.add_argument("--object-ids", nargs="*", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scene_dir = args.scene_dir.expanduser().resolve()
    sam3d_dir = (args.sam3d_dir or scene_dir / "sam3d").expanduser().resolve()
    scaled_dir = (args.scaled_dir or scene_dir / "sam3d+fpose" / "scaled").expanduser().resolve()
    scaled_dir.mkdir(parents=True, exist_ok=True)

    object_ids = args.object_ids or discover_ids(sam3d_dir)
    if not object_ids:
        raise RuntimeError(f"No SAM3D object meshes found in {sam3d_dir}")

    report = {
        "scene_dir": str(scene_dir),
        "sam3d_dir": str(sam3d_dir),
        "scaled_dir": str(scaled_dir),
        "T_yup_to_zup": T_YUP_TO_ZUP.tolist(),
        "objects": [prepare_one(sam3d_dir, scaled_dir, object_id, args.overwrite) for object_id in object_ids],
    }
    report_path = scaled_dir / "mesh_prepare_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[prepare-sam3d] wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
