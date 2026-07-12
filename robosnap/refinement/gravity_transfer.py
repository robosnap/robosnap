#!/usr/bin/env python3
"""Transfer an estimated gravity direction into a scene coordinate frame.

This utility is intentionally lightweight: it does not import VIPE or GeoCalib.
It consumes a gravity vector or roll/pitch estimate, builds a 4x4 rotation that
aligns that direction to a target up axis, and optionally applies it to a GLB.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

AXES = {
    "x": np.array([1.0, 0.0, 0.0], dtype=np.float64),
    "y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
    "z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
}


def normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        raise ValueError(f"Cannot normalize near-zero vector: {vec}")
    return vec / norm


def gravity_from_roll_pitch(roll: float, pitch: float) -> np.ndarray:
    """Match the VIPE/GeoCalib gravity convention from roll and pitch."""
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    return normalize(np.array([-sr * cp, -cr * cp, sp], dtype=np.float64))


def read_json_key(path: Path, key: str) -> Any:
    data = json.loads(path.read_text())
    value: Any = data
    for part in key.split("."):
        if isinstance(value, dict):
            value = value[part]
        elif isinstance(value, list):
            value = value[int(part)]
        else:
            raise KeyError(f"Cannot descend into {part!r} while reading {key!r}")
    return value


def vector_from_json(path: Path, key: str) -> np.ndarray:
    value = read_json_key(path, key)
    if isinstance(value, dict):
        for candidate in ("vec3d", "gravity", "vector", "down", "up"):
            if candidate in value:
                value = value[candidate]
                break
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"Expected JSON key {key!r} to contain a 3-vector, got shape {arr.shape}")
    return normalize(arr)


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
        axis = normalize(np.cross(src, helper))
        return rotation_axis_angle(axis, math.pi)
    axis = cross / norm_cross
    angle = math.atan2(norm_cross, dot)
    return rotation_axis_angle(axis, angle)


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


def centroid_transform(scene: Any, rotation: np.ndarray, center_mode: str) -> tuple[np.ndarray, np.ndarray]:
    bounds = np.asarray(scene.bounds, dtype=np.float64)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
        raise ValueError("Scene has invalid bounds")
    if center_mode == "origin":
        pivot = np.zeros(3, dtype=np.float64)
    elif center_mode == "bounds_center":
        pivot = (bounds[0] + bounds[1]) * 0.5
    elif center_mode == "floor_center":
        pivot = (bounds[0] + bounds[1]) * 0.5
        pivot[2] = bounds[0, 2]
    else:
        raise ValueError(f"Unknown center mode: {center_mode}")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    translate_to_origin = np.eye(4, dtype=np.float64)
    translate_to_origin[:3, 3] = -pivot
    translate_back = np.eye(4, dtype=np.float64)
    translate_back[:3, 3] = pivot
    return translate_back @ transform @ translate_to_origin, pivot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build/apply a gravity alignment transform for scene refinement.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--gravity-vector", nargs=3, type=float, metavar=("X", "Y", "Z"), help="Estimated gravity 3-vector.")
    source.add_argument("--gravity-json", type=Path, help="JSON file containing a gravity 3-vector.")
    source.add_argument("--roll-pitch", nargs=2, type=float, metavar=("ROLL", "PITCH"), help="Roll/pitch estimate in radians unless --angles-degrees is set.")
    parser.add_argument("--gravity-key", default="gravity", help="Dot path inside --gravity-json. Default: gravity")
    parser.add_argument("--angles-degrees", action="store_true", help="Interpret --roll-pitch values as degrees.")
    parser.add_argument("--source-vector-is", choices=["down", "up"], default="down", help="Whether the source vector points with gravity/down or against it/up.")
    parser.add_argument("--target-up", choices=sorted(AXES), default="z", help="Target scene up axis. Default: z")
    parser.add_argument("--input-scene", type=Path, help="Optional input GLB/scene to rotate.")
    parser.add_argument("--output-scene", type=Path, help="Output GLB/scene when --input-scene is set.")
    parser.add_argument("--transform-json", type=Path, required=True, help="Output JSON transform metadata.")
    parser.add_argument("--center-mode", choices=["origin", "bounds_center", "floor_center"], default="origin", help="Pivot used when applying the transform to a scene.")
    parser.add_argument("--dry-run", action="store_true", help="Print/write metadata without exporting a scene.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.gravity_vector is not None:
        source_vec = normalize(np.asarray(args.gravity_vector, dtype=np.float64))
        source_kind = "gravity-vector"
    elif args.gravity_json is not None:
        source_vec = vector_from_json(args.gravity_json, args.gravity_key)
        source_kind = f"gravity-json:{args.gravity_key}"
    else:
        roll, pitch = args.roll_pitch
        if args.angles_degrees:
            roll = math.radians(roll)
            pitch = math.radians(pitch)
        source_vec = gravity_from_roll_pitch(roll, pitch)
        source_kind = "roll-pitch"

    source_down = source_vec if args.source_vector_is == "down" else -source_vec
    target_up = AXES[args.target_up]
    target_down = -target_up
    rotation = rotation_between(source_down, target_down)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    pivot = None

    if args.input_scene:
        if not args.output_scene and not args.dry_run:
            raise ValueError("--output-scene is required when --input-scene is set unless --dry-run is used")
        import trimesh

        scene = trimesh.load(str(args.input_scene), force="scene")
        transform, pivot = centroid_transform(scene, rotation, args.center_mode)
        scene.apply_transform(transform)
        if args.output_scene and not args.dry_run:
            args.output_scene.parent.mkdir(parents=True, exist_ok=True)
            scene.export(str(args.output_scene))
            print(f"[OK] wrote {args.output_scene}")

    meta = {
        "source": source_kind,
        "source_vector_is": args.source_vector_is,
        "source_vector": source_vec.tolist(),
        "source_down": source_down.tolist(),
        "target_up_axis": args.target_up,
        "target_down": target_down.tolist(),
        "center_mode": args.center_mode,
        "pivot": None if pivot is None else pivot.tolist(),
        "input_scene": None if args.input_scene is None else str(args.input_scene),
        "output_scene": None if args.output_scene is None else str(args.output_scene),
        "transform_4x4": transform.tolist(),
    }
    args.transform_json.parent.mkdir(parents=True, exist_ok=True)
    args.transform_json.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(f"[OK] wrote {args.transform_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
