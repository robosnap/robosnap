# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the VGGT repository.

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_model(args, device):
    from vggt.models.vggt import VGGT

    if args.checkpoint:
        model = VGGT()
        state_dict = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state_dict)
    else:
        model = VGGT.from_pretrained(args.model)
    return model.eval().to(device)


def save_depth_preview(depth: np.ndarray, output_path: Path) -> None:
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        Image.fromarray(np.zeros(depth.shape, dtype=np.uint8)).save(output_path)
        return

    lo, hi = np.percentile(depth[valid], [2, 98])
    if hi <= lo:
        hi = lo + 1e-6
    normalized = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    Image.fromarray((normalized * 255).astype(np.uint8)).save(output_path)


def save_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(points, colors):
            handle.write(
                f"{point[0]:.7g} {point[1]:.7g} {point[2]:.7g} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def save_pcd(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    rgb = (
        (colors[:, 0].astype(np.uint32) << 16)
        | (colors[:, 1].astype(np.uint32) << 8)
        | colors[:, 2].astype(np.uint32)
    )
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# .PCD v0.7 - Point Cloud Data file format\n")
        handle.write("VERSION 0.7\n")
        handle.write("FIELDS x y z rgb\n")
        handle.write("SIZE 4 4 4 4\n")
        handle.write("TYPE F F F U\n")
        handle.write("COUNT 1 1 1 1\n")
        handle.write(f"WIDTH {len(points)}\n")
        handle.write("HEIGHT 1\n")
        handle.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        handle.write(f"POINTS {len(points)}\n")
        handle.write("DATA ascii\n")
        for point, packed_rgb in zip(points, rgb):
            handle.write(
                f"{point[0]:.7g} {point[1]:.7g} {point[2]:.7g} {int(packed_rgb)}\n"
            )


def select_points(
    points_world: np.ndarray,
    depth: np.ndarray,
    depth_conf: np.ndarray | None,
    colors: np.ndarray,
    args,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    stride = max(args.stride, 1)
    points = points_world[::stride, ::stride]
    depth = depth[::stride, ::stride]
    colors = colors[::stride, ::stride]
    conf = depth_conf[::stride, ::stride] if depth_conf is not None else None

    valid = np.isfinite(points).all(axis=-1) & np.isfinite(depth) & (depth > 0)
    if valid_mask is not None:
        valid &= valid_mask[::stride, ::stride]
    if args.conf_threshold is not None and conf is not None:
        valid &= conf >= args.conf_threshold

    points = points[valid].reshape(-1, 3)
    colors = colors[valid].reshape(-1, 3)

    if args.max_points is not None and args.max_points > 0 and len(points) > args.max_points:
        rng = np.random.default_rng(args.seed)
        keep = rng.choice(len(points), size=args.max_points, replace=False)
        keep.sort()
        points = points[keep]
        colors = colors[keep]

    return points.astype(np.float32), colors.astype(np.uint8)
