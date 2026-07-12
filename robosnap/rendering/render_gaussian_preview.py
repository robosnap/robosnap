#!/usr/bin/env python3
"""Lightweight Gaussian PLY preview renderer for release sanity checks."""

from __future__ import annotations

import argparse
import colorsys
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageStat

SH_C0 = 0.28209479


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


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


def parse_ply_header(path: Path):
    props: list[tuple[str, str]] = []
    vertex_count = None
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
                vertex_count = int(text.split()[-1])
            elif text.startswith("property ") and vertex_count is not None:
                parts = text.split()
                props.append((parts[1], parts[2]))
            elif text == "end_header":
                break
    if fmt != "binary_little_endian":
        raise ValueError(f"Expected binary_little_endian PLY, got {fmt}")
    if vertex_count is None:
        raise ValueError("PLY has no vertex element")
    return vertex_count, props, offset


def load_gaussian_ply(path: Path):
    count, props, offset = parse_ply_header(path)
    dtype = np.dtype([(name, numpy_dtype(prop_type)) for prop_type, name in props])
    with path.open("rb") as f:
        f.seek(offset)
        data = np.fromfile(f, dtype=dtype, count=count)
    fields = set(data.dtype.names or [])
    xyz = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32)
    if {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(fields):
        rgb = np.column_stack([
            0.5 + SH_C0 * data["f_dc_0"],
            0.5 + SH_C0 * data["f_dc_1"],
            0.5 + SH_C0 * data["f_dc_2"],
        ]).astype(np.float32)
        rgb = np.clip(rgb, 0.0, 1.0)
    elif {"red", "green", "blue"}.issubset(fields):
        rgb = np.column_stack([data["red"], data["green"], data["blue"]]).astype(np.float32) / 255.0
    else:
        rgb = np.full((len(data), 3), (0.35, 0.65, 1.0), dtype=np.float32)
    opacity = sigmoid_np(data["opacity"].astype(np.float32)) if "opacity" in fields else np.full(len(data), 0.5, dtype=np.float32)
    if {"scale_0", "scale_1", "scale_2"}.issubset(fields):
        scales = np.exp(np.column_stack([data["scale_0"], data["scale_1"], data["scale_2"]]).astype(np.float32))
        scale_size = np.cbrt(np.maximum(scales[:, 0] * scales[:, 1] * scales[:, 2], 1e-12))
    else:
        scale_size = np.full(len(data), 0.01, dtype=np.float32)
    return xyz, rgb, opacity.astype(np.float32), scale_size.astype(np.float32)


def boost_color(rgb: np.ndarray) -> np.ndarray:
    out = np.empty_like(rgb)
    for i, (r, g, b) in enumerate(rgb):
        h, s, v = colorsys.rgb_to_hsv(float(r), float(g), float(b))
        out[i] = colorsys.hsv_to_rgb(h, min(1.0, s * 1.35 + 0.06), min(1.0, v * 1.10 + 0.05))
    return out


def make_view_basis(name: str):
    if name == "camera":
        return np.array([1, 0, 0], np.float32), np.array([0, 1, 0], np.float32), np.array([0, 0, -1], np.float32)
    if name == "front":
        return np.array([1, 0, 0], np.float32), np.array([0, 1, 0], np.float32), np.array([0, 0, 1], np.float32)
    if name == "side":
        return np.array([0, 0, 1], np.float32), np.array([0, 1, 0], np.float32), np.array([1, 0, 0], np.float32)
    if name == "iso":
        forward = np.array([0.58, -0.52, 0.63], dtype=np.float32)
        forward /= np.linalg.norm(forward)
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(world_up, forward)
        right /= np.linalg.norm(right)
        up = np.cross(forward, right)
        up /= np.linalg.norm(up)
        return right, up, forward
    raise ValueError(f"Unknown view: {name}")


def weighted_sample(opacity: np.ndarray, sample_count: int, seed: int) -> np.ndarray:
    if sample_count >= len(opacity):
        return np.arange(len(opacity))
    rng = np.random.default_rng(seed)
    weights = np.clip(opacity, 0.03, 1.0).astype(np.float64)
    weights /= weights.sum()
    return rng.choice(len(opacity), size=sample_count, replace=False, p=weights)


def render_view(xyz, rgb, opacity, scale_size, view_name: str, output: Path, args) -> Path:
    idx = weighted_sample(opacity, args.samples, args.seed)
    pts = xyz[idx]
    colors = boost_color(rgb[idx])
    op = opacity[idx]
    sizes = scale_size[idx]
    center = np.median(pts, axis=0)
    pts = pts - center
    right, up, forward = make_view_basis(view_name)
    sx = pts @ right
    sy = pts @ up
    depth = pts @ forward
    x0, x1 = np.percentile(sx, [args.crop, 100.0 - args.crop])
    y0, y1 = np.percentile(sy, [args.crop, 100.0 - args.crop])
    scale = min((args.width * 0.86) / max(x1 - x0, 1e-6), (args.height * 0.86) / max(y1 - y0, 1e-6))
    px = args.width * 0.5 + (sx - (x0 + x1) * 0.5) * scale
    py = args.height * 0.5 - (sy - (y0 + y1) * 0.5) * scale
    inside = (px > -60) & (px < args.width + 60) & (py > -60) & (py < args.height + 60)
    px, py, colors, op, sizes, depth = px[inside], py[inside], colors[inside], op[inside], sizes[inside], depth[inside]
    med_size = np.median(sizes[sizes > 0]) if np.any(sizes > 0) else 0.01
    radius = args.radius_base + args.radius_gain * np.sqrt(np.clip(sizes / max(med_size, 1e-9), 0.05, 60.0))
    radius = np.clip(radius, args.radius_min, args.radius_max)
    alpha = np.clip(args.alpha_base + args.alpha_gain * op, 25, 210).astype(np.uint8)
    rgb8 = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    img = Image.new("RGBA", (args.width, args.height), (248, 249, 250, 255))
    draw = ImageDraw.Draw(img, "RGBA")
    for j in np.argsort(depth):
        x, y, r = float(px[j]), float(py[j]), float(radius[j])
        fill = (int(rgb8[j, 0]), int(rgb8[j, 1]), int(rgb8[j, 2]), int(alpha[j]))
        draw.ellipse((x - r, y - r, x + r, y + r), fill=fill)
    img = img.filter(ImageFilter.UnsharpMask(radius=1.0, percent=115, threshold=3)).convert("RGB")
    img = ImageEnhance.Contrast(img).enhance(1.08)
    img = ImageEnhance.Color(img).enhance(1.10)
    output.parent.mkdir(parents=True, exist_ok=True)
    img.save(output, quality=96)
    stat = ImageStat.Stat(img)
    print(f"{view_name}: saved {output}; mean RGB {[round(v, 2) for v in stat.mean]}")
    return output


def make_contact(paths: list[Path], output: Path) -> None:
    thumbs = []
    for path in paths:
        im = Image.open(path).convert("RGB")
        im.thumbnail((720, 450), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (720, 450), (248, 249, 250))
        canvas.paste(im, ((720 - im.width) // 2, (450 - im.height) // 2))
        thumbs.append(canvas)
    sheet = Image.new("RGB", (1440, 900), (248, 249, 250))
    for i, im in enumerate(thumbs[:4]):
        sheet.paste(im, ((i % 2) * 720, (i // 2) * 450))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=96)
    print(f"contact: saved {output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Render quick preview images from a Gaussian PLY.")
    parser.add_argument("--input", type=Path, required=True, help="Gaussian PLY file.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--views", default="camera,front,side,iso")
    parser.add_argument("--samples", type=int, default=320000)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--width", type=int, default=1800)
    parser.add_argument("--height", type=int, default=1200)
    parser.add_argument("--crop", type=float, default=0.7)
    parser.add_argument("--radius-min", type=float, default=1.25)
    parser.add_argument("--radius-base", type=float, default=1.65)
    parser.add_argument("--radius-gain", type=float, default=3.05)
    parser.add_argument("--radius-max", type=float, default=11.5)
    parser.add_argument("--alpha-base", type=float, default=34.0)
    parser.add_argument("--alpha-gain", type=float, default=96.0)
    parser.add_argument("--no-contact", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    xyz, rgb, opacity, scale_size = load_gaussian_ply(args.input)
    paths = []
    for view in [v.strip() for v in args.views.split(",") if v.strip()]:
        paths.append(render_view(xyz, rgb, opacity, scale_size, view, args.output_dir / f"{view}.png", args))
    if not args.no_contact and paths:
        make_contact(paths, args.output_dir / "contact_sheet.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
