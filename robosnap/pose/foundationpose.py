#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run FoundationPose on ViPE outputs (Real2Sim pipeline).

Inputs:
  - ViPE depth EXR sequence:  vipe_results/depth/exr_file/*.exr  (channel 'Z', meters)
  - ViPE intrinsics:          vipe_results/intrinsics/video.npz   (data[i]=[fx,fy,cx,cy])
  - RGB source:               video.mp4 (preferred) OR single image.png
  - Object mask (RGBA png):   alpha>0 is object
  - Object mesh:              .glb/.obj (must be a triangle mesh). Use *scaled* mesh.

Outputs:
  - poses: out_dir/ob_in_cam/000000.txt ...  (4x4, object->camera, OpenCV camera)
  - optional visualization: out_dir/track_vis/*.png

Example:
python robosnap/pose/foundationpose.py \
  --vipe_base /path/to/scene/sam3d+fpose/vipe_results \
  --image     /path/to/scene/image.png \
  --mask_rgba /path/to/scene/sam3d/0.png \
  --mesh      /path/to/scene/sam3d+fpose/scaled/0_z_up.glb \
  --out_dir   /path/to/scene/sam3d+fpose/foundationpose/0_fpose_zup \
  --max_frames 1 \
  --debug 2
"""

import os
import sys
import glob
import json
import argparse
from pathlib import Path

import numpy as np
import cv2

# ---------- FoundationPose import path ----------
ROBOSNAP_ROOT = Path(
    os.environ.get("ROBOSNAP_ROOT", Path(__file__).resolve().parents[2])
).expanduser().resolve()
FP_DIR = Path(
    os.environ.get("FOUNDATIONPOSE_DIR", ROBOSNAP_ROOT / "third_party" / "FoundationPose")
).expanduser().resolve()
sys.path.insert(0, str(FP_DIR))

import trimesh
import nvdiffrast.torch as dr

from estimater import FoundationPose
from learning.training.predict_score import ScorePredictor
from learning.training.predict_pose_refine import PoseRefinePredictor
from Utils import draw_xyz_axis, draw_posed_3d_box


# ------------------------- IO helpers -------------------------

import trimesh
import numpy as np

def load_mesh_for_foundationpose(mesh_path: str, rgba=(200, 200, 200, 255)) -> trimesh.Trimesh:
    obj = trimesh.load(mesh_path, force='scene')  # GLB 常常是 Scene

    if isinstance(obj, trimesh.Scene):
        if len(obj.geometry) == 0:
            raise RuntimeError(f"No geometry in mesh file: {mesh_path}")
        mesh = trimesh.util.concatenate(tuple(obj.geometry.values()))
    else:
        mesh = obj

    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise RuntimeError(f"Loaded mesh is not a valid triangle mesh: {type(mesh)} from {mesh_path}")

    # float32
    mesh.vertices = np.ascontiguousarray(mesh.vertices.astype(np.float32))
    # ensure normals exist
    _ = mesh.vertex_normals
    mesh.vertex_normals = np.ascontiguousarray(mesh.vertex_normals.astype(np.float32))

    # 强制 vertex color，绕开 PBRMaterial.image
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh,
        vertex_colors=np.tile(np.array(rgba, dtype=np.uint8), (len(mesh.vertices), 1))
    )
    return mesh

def load_exr_depth(path: str) -> np.ndarray:
    import OpenEXR
    import Imath

    exr = OpenEXR.InputFile(path)
    dw = exr.header()["dataWindow"]
    W = dw.max.x - dw.min.x + 1
    H = dw.max.y - dw.min.y + 1

    z = np.frombuffer(
        exr.channel("Z", Imath.PixelType(Imath.PixelType.FLOAT)),
        dtype=np.float32
    ).reshape(H, W).copy()

    # Clean
    z[~np.isfinite(z)] = 0.0
    z[z < 1e-3] = 0.0
    return np.ascontiguousarray(z.astype(np.float32))


def load_vipe_intrinsics_npz(npz_path: str) -> np.ndarray:
    data = np.load(npz_path)
    intr = data["data"]  # (N,4) [fx, fy, cx, cy]
    return intr.astype(np.float32)


def make_K(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
    return K


def load_rgba_mask_as_binary(path: str, H: int, W: int, thresh: int = 0) -> np.ndarray:
    """
    RGBA png: alpha>thresh is object.
    Returns uint8 mask {0,1}, shape (H,W)
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Failed to read mask: {path}")

    if img.ndim == 2:
        # grayscale
        m = (img > 0).astype(np.uint8)
    else:
        if img.shape[2] == 4:
            alpha = img[:, :, 3]
            m = (alpha > thresh).astype(np.uint8)
        else:
            # RGB but no alpha: treat non-black as object (fallback)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            m = (gray > 0).astype(np.uint8)

    if m.shape != (H, W):
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    return np.ascontiguousarray(m.astype(np.uint8))


def read_video_frames_by_indices(video_path: str, indices: list[int], max_frames: int | None) -> list[np.ndarray]:
    """
    Returns RGB uint8 frames, contiguous, shape (H,W,3).
    Assumes index i corresponds to frame i in video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frames = []
    count = 0
    for idx in indices:
        if max_frames is not None and count >= max_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = bgr[:, :, ::-1].copy()
        frames.append(np.ascontiguousarray(rgb.astype(np.uint8)))
        count += 1

    cap.release()
    if len(frames) == 0:
        raise RuntimeError("No frames read from video (check indices / video length).")
    return frames


def read_single_image(image_path: str) -> np.ndarray:
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    rgb = bgr[:, :, ::-1].copy()
    return np.ascontiguousarray(rgb.astype(np.uint8))


# ------------------------- main pipeline -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vipe_base", required=True, help=".../vipe_results")
    parser.add_argument("--mesh", required=True, help="scaled mesh file: .glb/.obj")
    parser.add_argument("--mask_rgba", required=True, help="RGBA mask png for frame0 (alpha>0 is object)")
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--video", default=None, help="video.mp4 (preferred for tracking)")
    parser.add_argument("--image", default=None, help="single image (fallback)")

    parser.add_argument("--max_frames", type=int, default=None, help="limit number of frames to run")
    parser.add_argument("--est_iter", type=int, default=5)
    parser.add_argument("--track_iter", type=int, default=2)
    parser.add_argument("--debug", type=int, default=1)

    args = parser.parse_args()

    vipe_base = Path(args.vipe_base)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- depth list ----
    depth_dir = vipe_base / "depth" / "exr_file"
    exr_paths = sorted(glob.glob(str(depth_dir / "*.exr")))
    if len(exr_paths) == 0:
        raise FileNotFoundError(f"No EXR found under: {depth_dir}")

    # frame indices derived from filenames (e.g., 00000.exr)
    frame_indices = [int(Path(p).stem) for p in exr_paths]
    # keep ordering aligned with exr_paths
    if args.max_frames is not None:
        exr_paths = exr_paths[:args.max_frames]
        frame_indices = frame_indices[:args.max_frames]

    # read first depth to get H,W
    depth0 = load_exr_depth(exr_paths[0])
    H, W = depth0.shape

    # ---- intrinsics ----
    intr_npz = vipe_base / "intrinsics" / "video.npz"
    intr = load_vipe_intrinsics_npz(str(intr_npz))
    fx, fy, cx, cy = intr[0].tolist()

    # sanity check
    if abs(cx - W * 0.5) > 50 or abs(cy - H * 0.5) > 50:
        print(f"[WARN] cx,cy look off-center: cx={cx},cy={cy}, but W/2={W*0.5},H/2={H*0.5}")

    K = make_K(fx, fy, cx, cy)

    # ---- rgb frames ----
    if args.video is not None:
        rgb_frames = read_video_frames_by_indices(args.video, frame_indices, args.max_frames)
        if len(rgb_frames) != len(exr_paths):
            # align lengths (video may be shorter)
            n = min(len(rgb_frames), len(exr_paths))
            rgb_frames = rgb_frames[:n]
            exr_paths = exr_paths[:n]
            frame_indices = frame_indices[:n]
            print(f"[WARN] video shorter than depth. Using first {n} frames.")
    else:
        if args.image is None:
            raise ValueError("Need either --video or --image.")
        img = read_single_image(args.image)
        if img.shape[:2] != (H, W):
            img = cv2.resize(img[:, :, ::-1], (W, H), interpolation=cv2.INTER_LINEAR)[:, :, ::-1].copy()
        rgb_frames = [img for _ in range(len(exr_paths))]

    # ---- mask (use the same mask for all frames unless you later provide per-frame masks) ----
    mask0 = load_rgba_mask_as_binary(args.mask_rgba, H, W, thresh=0)
    # optional: if mask empty, hard fail
    if int(mask0.sum()) < 10:
        raise RuntimeError("Mask seems empty (alpha all zero). Please verify the RGBA mask.")

    masks = [mask0 for _ in range(len(exr_paths))]

    # ---- mesh ----
    mesh = load_mesh_for_foundationpose(args.mesh)

    # oriented bbox (for drawing) same as demo
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2.0, extents / 2.0], axis=0).reshape(2, 3)

    # ---- init estimator ----
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()

    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug=args.debug,
        debug_dir=str(out_dir),
        glctx=glctx
    )

    # ---- output dirs ----
    pose_dir = out_dir / "ob_in_cam"
    vis_dir = out_dir / "track_vis"
    pose_dir.mkdir(exist_ok=True)
    if args.debug >= 2:
        vis_dir.mkdir(exist_ok=True)

    # ---- run ----
    poses = []
    for i, exr_path in enumerate(exr_paths):
        depth = load_exr_depth(exr_path)
        color = rgb_frames[i]
        ob_mask = masks[i]

        if i == 0:
            pose = est.register(K=K, rgb=color, depth=depth, ob_mask=ob_mask, iteration=args.est_iter)
        else:
            pose = est.track_one(rgb=color, depth=depth, K=K, iteration=args.track_iter)

        poses.append(pose)
        np.savetxt(str(pose_dir / f"{frame_indices[i]:06d}.txt"), pose.reshape(4, 4))

        if args.debug >= 2:
            # follow demo: center_pose = pose @ inv(to_origin)
            center_pose = pose @ np.linalg.inv(to_origin)

            vis = color.copy()
            vis = draw_posed_3d_box(K, img=vis, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3, transparency=0, is_input_rgb=True)
            cv2.imwrite(str(vis_dir / f"{frame_indices[i]:06d}.png"), vis[:, :, ::-1])  # to BGR for imwrite

        if (i % 10) == 0:
            print(f"[{i:04d}/{len(exr_paths)}] saved pose for frame {frame_indices[i]}")

    # ---- save manifest ----
    manifest = {
        "vipe_base": str(vipe_base),
        "mesh": args.mesh,
        "mask_rgba": args.mask_rgba,
        "rgb_source": args.video if args.video is not None else args.image,
        "K": K.tolist(),
        "image_size_hw": [int(H), int(W)],
        "num_frames": len(poses),
        "pose_dir": str(pose_dir),
        "vis_dir": str(vis_dir) if args.debug >= 2 else None,
        "pose_convention": "ob_in_cam (object->camera), OpenCV camera: X right, Y down, Z forward"
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("Done.")
    print("Poses:", pose_dir)
    if args.debug >= 2:
        print("Visualizations:", vis_dir)


if __name__ == "__main__":
    main()