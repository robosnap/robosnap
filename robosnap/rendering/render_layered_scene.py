#!/usr/bin/env python3
"""Render a camera-aligned Gaussian background and mesh foreground."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render layered Gaussian and mesh scene assets.")
    parser.add_argument("--foreground", type=Path, required=True)
    parser.add_argument("--background-ply", type=Path, required=True)
    parser.add_argument("--camera-npz", type=Path, required=True)
    parser.add_argument("--gravity-transform", type=Path, required=True)
    parser.add_argument("--foreground-camera-json", type=Path)
    parser.add_argument("--output-image", type=Path, required=True)
    parser.add_argument("--output-ply", type=Path)
    parser.add_argument("--status-json", type=Path)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--image-width", type=int)
    parser.add_argument("--image-height", type=int)
    parser.add_argument("--device", default=os.environ.get("ROBOSNAP_DEVICE", "cuda:0"))
    parser.add_argument("--background-color", nargs=3, type=float, default=[0.96, 0.96, 0.96])
    parser.add_argument("--min-background-std", type=float, default=0.03)
    parser.add_argument("--min-background-coverage", type=float, default=0.05)
    parser.add_argument("--min-foreground-coverage", type=float, default=0.001)
    parser.add_argument("--foreground-samples", type=int, default=100000)
    parser.add_argument("--background-samples", type=int, default=300000)
    return parser.parse_args()


def load_camera(path: Path, frame_index: int, width: int | None, height: int | None):
    import numpy as np

    data = np.load(path)
    if "w2c_render" not in data:
        raise KeyError(f"w2c_render is missing from {path}")
    intrinsic_key = next(
        (key for key in ("intrinsics_render", "intrinsics_vipe", "intrinsics_da3_pred") if key in data),
        None,
    )
    if intrinsic_key is None:
        raise KeyError(f"No render intrinsics found in {path}")
    w2cs = np.asarray(data["w2c_render"], dtype=np.float64)
    intrinsics = np.asarray(data[intrinsic_key], dtype=np.float64)
    idx = max(0, min(int(frame_index), len(w2cs) - 1))
    w2c = w2cs[idx]
    if w2c.shape == (3, 4):
        padded = np.eye(4, dtype=np.float64)
        padded[:3, :4] = w2c
        w2c = padded
    k = intrinsics[min(idx, len(intrinsics) - 1)].copy()
    source_width = max(1, int(round(2.0 * float(k[0, 2]))))
    source_height = max(1, int(round(2.0 * float(k[1, 2]))))
    target_width = int(width or source_width)
    target_height = int(height or source_height)
    if (target_width, target_height) != (source_width, source_height):
        k[0, :] *= target_width / source_width
        k[1, :] *= target_height / source_height
    return w2c, k, target_width, target_height, idx, intrinsic_key


def load_foreground_intrinsics(path: Path, width: int, height: int):
    import numpy as np

    data = json.loads(path.read_text(encoding="utf-8"))
    if "intrinsic_original_pixels" not in data or "original_size_wh" not in data:
        raise KeyError(f"VGGT original-pixel intrinsics are missing from {path}")
    k = np.asarray(data["intrinsic_original_pixels"], dtype=np.float64).copy()
    source_width, source_height = (int(value) for value in data["original_size_wh"])
    k[0, :] *= width / source_width
    k[1, :] *= height / source_height
    return k, "intrinsic_original_pixels"


def load_gravity_camera(path: Path, w2c_background):
    import numpy as np

    data = json.loads(path.read_text(encoding="utf-8"))
    transform = np.asarray(data["transform_gravity_from_background"], dtype=np.float64)
    return np.asarray(w2c_background, dtype=np.float64) @ np.linalg.inv(transform), data


def load_gaussians(path: Path, device: str):
    import numpy as np
    import torch
    from plyfile import PlyData

    vertices = PlyData.read(str(path))["vertex"].data
    names = set(vertices.dtype.names or ())
    required = {
        "x",
        "y",
        "z",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    }
    missing = sorted(required - names)
    if missing:
        raise ValueError(f"{path} is not a Gaussian PLY; missing: {', '.join(missing)}")

    stack = lambda keys: np.column_stack([vertices[key].astype(np.float32, copy=False) for key in keys])
    means_np = stack(["x", "y", "z"])
    scales_np = np.exp(stack(["scale_0", "scale_1", "scale_2"]))
    quats_np = stack(["rot_0", "rot_1", "rot_2", "rot_3"])
    quats_np /= np.clip(np.linalg.norm(quats_np, axis=1, keepdims=True), 1e-12, None)
    opacities_np = 1.0 / (1.0 + np.exp(-vertices["opacity"].astype(np.float32, copy=False)))
    dc = stack(["f_dc_0", "f_dc_1", "f_dc_2"])
    rest_names = sorted(
        (name for name in names if name.startswith("f_rest_")),
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    if rest_names:
        rest = stack(rest_names)
        if rest.shape[1] % 3:
            raise ValueError(f"Unexpected SH field count in {path}: {rest.shape[1]}")
        rest = rest.reshape(len(rest), 3, rest.shape[1] // 3)
        harmonics_np = np.concatenate([dc[:, :, None], rest], axis=2).transpose(0, 2, 1)
    else:
        harmonics_np = dc[:, None, :]

    to_device = lambda array: torch.from_numpy(np.ascontiguousarray(array)).to(device=device)
    return {
        "means": to_device(means_np),
        "scales": to_device(scales_np),
        "quats": to_device(quats_np),
        "opacities": to_device(opacities_np),
        "harmonics": to_device(harmonics_np),
        "means_np": means_np,
        "dc_np": dc,
    }


def render_gaussians(gaussians, w2c, k, width: int, height: int, background_color):
    import numpy as np
    import torch
    from gsplat import rasterization

    view = torch.from_numpy(np.asarray(w2c, dtype=np.float32)).to(gaussians["means"].device)[None]
    intrinsics = torch.from_numpy(np.asarray(k, dtype=np.float32)).to(gaussians["means"].device)[None]
    background = torch.tensor(background_color, dtype=torch.float32, device=gaussians["means"].device)[None]
    degree = int(round(gaussians["harmonics"].shape[1] ** 0.5)) - 1
    rendered, alpha, _ = rasterization(
        means=gaussians["means"],
        quats=gaussians["quats"],
        scales=gaussians["scales"],
        opacities=gaussians["opacities"],
        colors=gaussians["harmonics"],
        viewmats=view,
        Ks=intrinsics,
        width=width,
        height=height,
        backgrounds=background,
        render_mode="RGB+ED",
        packed=False,
        sh_degree=degree,
    )
    rgb = rendered[0, :, :, :3].clamp(0.0, 1.0).detach().cpu().numpy()
    depth = rendered[0, :, :, 3].detach().cpu().numpy()
    coverage = alpha[0, :, :, 0].detach().cpu().numpy()
    return rgb, depth, coverage


def render_mesh(path: Path, w2c, k, width: int, height: int):
    import numpy as np
    import pyrender
    import trimesh

    mesh_scene = trimesh.load(str(path), force="scene", process=False)
    if not isinstance(mesh_scene, trimesh.Scene):
        wrapped = trimesh.Scene()
        wrapped.add_geometry(mesh_scene)
        mesh_scene = wrapped
    scene = pyrender.Scene.from_trimesh_scene(
        mesh_scene,
        bg_color=np.array([0.0, 0.0, 0.0, 0.0]),
        ambient_light=np.array([0.75, 0.75, 0.75]),
    )
    camera = pyrender.IntrinsicsCamera(
        fx=float(k[0, 0]),
        fy=float(k[1, 1]),
        cx=float(k[0, 2]),
        cy=float(k[1, 2]),
        znear=0.01,
        zfar=10000.0,
    )
    cv_to_gl = np.diag([1.0, -1.0, -1.0, 1.0])
    camera_pose = np.linalg.inv(w2c) @ cv_to_gl
    scene.add(camera, pose=camera_pose)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.0), pose=camera_pose)
    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    try:
        color, depth = renderer.render(
            scene,
            flags=pyrender.RenderFlags.RGBA | pyrender.RenderFlags.SKIP_CULL_FACES,
        )
    finally:
        renderer.delete()
    return color.astype(np.float32) / 255.0, depth.astype(np.float32)


def composite_layers(background, background_depth, background_alpha, foreground, foreground_depth):
    import numpy as np

    mesh_alpha = foreground[:, :, 3]
    mesh_mask = (foreground_depth > 0.0) & (mesh_alpha > 0.0)
    background_valid = (background_depth > 0.0) & (background_alpha > 0.02)
    visible = mesh_mask & (~background_valid | (foreground_depth <= background_depth * 1.02))
    depth_fallback = False
    if mesh_mask.mean() > 0.001 and visible.mean() < 0.15 * mesh_mask.mean():
        visible = mesh_mask
        depth_fallback = True
    alpha = (mesh_alpha * visible.astype(np.float32))[:, :, None]
    output = foreground[:, :, :3] * alpha + background * (1.0 - alpha)
    return output.clip(0.0, 1.0), mesh_mask, visible, depth_fallback


def write_debug_ply(path: Path, gaussians, foreground: Path, bg_limit: int, fg_limit: int) -> None:
    import numpy as np
    import trimesh

    means = gaussians["means_np"]
    colors = np.clip(0.5 + 0.28209479177387814 * gaussians["dc_np"], 0.0, 1.0)
    if len(means) > bg_limit:
        rng = np.random.default_rng(17)
        keep = rng.choice(len(means), size=bg_limit, replace=False)
        means, colors = means[keep], colors[keep]
    scene = trimesh.load(str(foreground), force="scene", process=False)
    mesh = scene.to_geometry() if isinstance(scene, trimesh.Scene) and hasattr(scene, "to_geometry") else scene.dump(concatenate=True)
    fg_points, _ = trimesh.sample.sample_surface(mesh, max(1000, fg_limit))
    fg_colors = np.tile(np.array([[1.0, 0.2, 0.05]]), (len(fg_points), 1))
    points = np.concatenate([means, fg_points], axis=0)
    rgb = (np.concatenate([colors, fg_colors], axis=0) * 255.0).clip(0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for point, color in zip(points, rgb):
            handle.write(
                f"{point[0]:.10g} {point[1]:.10g} {point[2]:.10g} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def main() -> int:
    args = parse_args()
    import numpy as np
    from PIL import Image

    w2c_background, background_k, width, height, frame_idx, intrinsic_key = load_camera(
        args.camera_npz,
        args.frame_index,
        args.image_width,
        args.image_height,
    )
    foreground_k = background_k.copy()
    foreground_intrinsic_key = f"shared:{intrinsic_key}"
    if args.foreground_camera_json:
        foreground_k, foreground_intrinsic_key = load_foreground_intrinsics(
            args.foreground_camera_json,
            width,
            height,
        )
    w2c_gravity, gravity_data = load_gravity_camera(args.gravity_transform, w2c_background)
    gaussians = load_gaussians(args.background_ply, args.device)
    background, background_depth, background_alpha = render_gaussians(
        gaussians,
        w2c_gravity,
        background_k,
        width,
        height,
        args.background_color,
    )
    bg_std = float(background.std())
    bg_coverage = float((background_alpha > 0.02).mean())
    if bg_std < args.min_background_std or bg_coverage < args.min_background_coverage:
        raise RuntimeError(
            f"Gaussian render failed quality gate: std={bg_std:.5f}, coverage={bg_coverage:.5f}"
        )

    foreground, foreground_depth = render_mesh(
        args.foreground,
        w2c_gravity,
        foreground_k,
        width,
        height,
    )
    output, mesh_mask, visible, depth_fallback = composite_layers(
        background,
        background_depth,
        background_alpha,
        foreground,
        foreground_depth,
    )
    foreground_coverage = float(mesh_mask.mean())
    visible_coverage = float(visible.mean())
    if foreground_coverage < args.min_foreground_coverage:
        raise RuntimeError(f"Foreground render is empty: coverage={foreground_coverage:.6f}")

    args.output_image.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((output * 255.0).round().astype(np.uint8)).save(args.output_image)
    if args.output_ply:
        write_debug_ply(
            args.output_ply,
            gaussians,
            args.foreground,
            args.background_samples,
            args.foreground_samples,
        )

    status_path = args.status_json or args.output_image.with_suffix(".json")
    status = {
        "status": "ok",
        "renderer": "gsplat+pyrender",
        "background_ply": str(args.background_ply),
        "foreground": str(args.foreground),
        "camera_npz": str(args.camera_npz),
        "gravity_transform": str(args.gravity_transform),
        "camera_frame": frame_idx,
        "intrinsic_key": intrinsic_key,
        "background_intrinsic_key": intrinsic_key,
        "foreground_intrinsic_key": foreground_intrinsic_key,
        "background_intrinsics": background_k.tolist(),
        "foreground_intrinsics": foreground_k.tolist(),
        "image_size_wh": [width, height],
        "gaussians": int(len(gaussians["means_np"])),
        "background_std": bg_std,
        "background_coverage": bg_coverage,
        "foreground_coverage": foreground_coverage,
        "visible_foreground_coverage": visible_coverage,
        "depth_test_fallback": depth_fallback,
        "camera_alignment": gravity_data.get("camera_alignment"),
    }
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(f"[render-layered] wrote {args.output_image}")
    print(f"[render-layered] wrote {status_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
