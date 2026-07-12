#!/usr/bin/env python3
"""Roughly fit a scene GLB into a Gaussian background coordinate frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

AXIS = {"x": 0, "y": 1, "z": 2}


def numpy_dtype(prop_type: str) -> str:
    table = {
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
    return table[prop_type]


def load_binary_ply_xyz(path: Path) -> np.ndarray:
    props = []
    count = None
    fmt = None
    offset = 0
    with path.open("rb") as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError("PLY header ended unexpectedly")
            offset += len(line)
            text = line.decode("ascii").strip()
            if text.startswith("format "):
                fmt = text.split()[1]
            elif text.startswith("element vertex "):
                count = int(text.split()[-1])
            elif text.startswith("property ") and count is not None:
                parts = text.split()
                props.append((parts[1], parts[2]))
            elif text == "end_header":
                break
    if fmt != "binary_little_endian":
        raise ValueError(f"Expected binary_little_endian PLY, got {fmt}")
    if count is None:
        raise ValueError("PLY has no vertex element")
    dtype = np.dtype([(name, numpy_dtype(prop_type)) for prop_type, name in props])
    with path.open("rb") as f:
        f.seek(offset)
        data = np.fromfile(f, dtype=dtype, count=count)
    return np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float64)


def robust_bounds(points: np.ndarray, crop_percent: float) -> tuple[np.ndarray, np.ndarray]:
    lo = np.percentile(points, crop_percent, axis=0)
    hi = np.percentile(points, 100.0 - crop_percent, axis=0)
    return lo, hi


def coord_transform(scene_up: str, background_up: str) -> np.ndarray:
    if scene_up == background_up:
        return np.eye(4)
    if scene_up == "z" and background_up == "y":
        mat = np.eye(4)
        mat[:3, :3] = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
        return mat
    if scene_up == "y" and background_up == "z":
        mat = np.eye(4)
        mat[:3, :3] = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
        return mat
    raise ValueError(f"Unsupported up-axis conversion: {scene_up} -> {background_up}")


def bounds_from_scene(scene: trimesh.Scene | trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    bounds = scene.bounds
    if bounds is None or not np.isfinite(bounds).all():
        raise ValueError("Scene has invalid bounds")
    return np.asarray(bounds[0], dtype=np.float64), np.asarray(bounds[1], dtype=np.float64)


def scale_from_extents(scene_extent: np.ndarray, bg_extent: np.ndarray, axes: list[int], mode: str) -> float:
    ratios = []
    for axis in axes:
        if scene_extent[axis] > 1e-9 and bg_extent[axis] > 1e-9:
            ratios.append(bg_extent[axis] / scene_extent[axis])
    if not ratios:
        raise ValueError("Cannot compute scale from degenerate bounds")
    if mode == "min":
        return float(np.min(ratios))
    if mode == "max":
        return float(np.max(ratios))
    return float(np.median(ratios))


def parse_args():
    parser = argparse.ArgumentParser(description="Roughly align a scene GLB to a Gaussian PLY background by coordinate conversion and robust bbox fitting.")
    parser.add_argument("--scene", type=Path, required=True, help="Input scene GLB.")
    parser.add_argument("--background-ply", type=Path, required=True, help="Gaussian/background PLY used as target frame.")
    parser.add_argument("--output-scene", type=Path, required=True, help="Aligned output GLB.")
    parser.add_argument("--transform-json", type=Path, help="Output transform metadata JSON.")
    parser.add_argument("--scene-up", choices=["y", "z"], default="z")
    parser.add_argument("--background-up", choices=["y", "z"], default="y")
    parser.add_argument("--horizontal-axes", default="xz", help="Axes used for fitting after conversion, default xz for Y-up background.")
    parser.add_argument("--up-axis", choices=["x", "y", "z"], default="y")
    parser.add_argument("--crop-percent", type=float, default=1.0, help="Robust percentile crop for background bounds.")
    parser.add_argument("--scale-mode", choices=["median", "min", "max"], default="median")
    parser.add_argument("--scale-multiplier", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import trimesh

    scene = trimesh.load(str(args.scene), force="scene")
    background_xyz = load_binary_ply_xyz(args.background_ply)
    bg_lo, bg_hi = robust_bounds(background_xyz, args.crop_percent)
    bg_center = (bg_lo + bg_hi) * 0.5
    bg_extent = bg_hi - bg_lo

    convert = coord_transform(args.scene_up, args.background_up)
    scene.apply_transform(convert)
    scene_lo, scene_hi = bounds_from_scene(scene)
    scene_center = (scene_lo + scene_hi) * 0.5
    scene_extent = scene_hi - scene_lo

    h_axes = [AXIS[ch] for ch in args.horizontal_axes]
    up_axis = AXIS[args.up_axis]
    scale = scale_from_extents(scene_extent, bg_extent, h_axes, args.scale_mode) * args.scale_multiplier

    scale_mat = np.eye(4)
    scale_mat[:3, :3] *= scale
    scene.apply_transform(scale_mat)
    scaled_lo, scaled_hi = bounds_from_scene(scene)
    scaled_center = (scaled_lo + scaled_hi) * 0.5

    translation = np.zeros(3, dtype=np.float64)
    for axis in h_axes:
        translation[axis] = bg_center[axis] - scaled_center[axis]
    translation[up_axis] = bg_lo[up_axis] - scaled_lo[up_axis]
    trans_mat = np.eye(4)
    trans_mat[:3, 3] = translation
    scene.apply_transform(trans_mat)

    transform = trans_mat @ scale_mat @ convert
    meta = {
        "scene": str(args.scene),
        "background_ply": str(args.background_ply),
        "output_scene": str(args.output_scene),
        "scene_up": args.scene_up,
        "background_up": args.background_up,
        "horizontal_axes": args.horizontal_axes,
        "up_axis": args.up_axis,
        "crop_percent": args.crop_percent,
        "scale": scale,
        "translation": translation.tolist(),
        "transform_4x4": transform.tolist(),
        "background_bounds": {"min": bg_lo.tolist(), "max": bg_hi.tolist()},
    }
    print(json.dumps(meta, indent=2))
    if args.dry_run:
        return 0
    args.output_scene.parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(args.output_scene))
    transform_path = args.transform_json or args.output_scene.with_suffix(".transform.json")
    transform_path.parent.mkdir(parents=True, exist_ok=True)
    transform_path.write_text(json.dumps(meta, indent=2))
    print(f"[OK] wrote {args.output_scene}")
    print(f"[OK] wrote {transform_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
