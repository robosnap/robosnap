#!/usr/bin/env python3
"""
Run FoundationPose for scene_droid scenes using VGGT single-image depth/K.

This prepares a small ViPE-compatible input folder from:
  {scene}/sam3d+fpose/vggt_single_image/{depth.npy,intrinsic_original_pixels.npy,camera.json}

Then it runs scripts/foundationpose.py for each SAM3D object mesh/mask and writes:
  {scene}/sam3d+fpose/foundationpose/{i}_fpose_zup/
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import Imath
import numpy as np
import OpenEXR
import trimesh


ROBOSNAP_ROOT = Path(
    os.environ.get("ROBOSNAP_ROOT", Path(__file__).resolve().parents[2])
).expanduser().resolve()


DEFAULT_SCENES = [
    "scene01",
    "scene02",
    "scene03",
    "scene05",
    "scene07",
    "scene09",
    "scene10",
    "scene12_backup",
    "scene14_backup",
    "scene15_backup",
]


@dataclass(frozen=True)
class Task:
    scene: str
    obj_idx: int
    mesh: Path
    mask: Path
    image: Path
    vipe_base: Path
    out_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene-root",
        required=True,
        help="Root containing scene folders.",
    )
    parser.add_argument("--scenes", nargs="*", default=DEFAULT_SCENES)
    parser.add_argument("--script-dir", default=str(ROBOSNAP_ROOT), help="RoboSnap repository root.")
    parser.add_argument("--python", default=os.environ.get("PY_FPOSE", os.environ.get("PY_LAYOUT", "python")))
    parser.add_argument("--gpus", default="0", help="Comma-separated physical GPU ids.")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=1)
    parser.add_argument("--est-iter", type=int, default=10)
    parser.add_argument("--track-iter", type=int, default=2)
    parser.add_argument("--debug", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_exr_z(path: Path, depth: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = depth.shape
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    header = OpenEXR.Header(w, h)
    header["channels"] = {"Z": Imath.Channel(pt)}
    exr = OpenEXR.OutputFile(str(path), header)
    exr.writePixels({"Z": np.ascontiguousarray(depth.astype(np.float32)).tobytes()})
    exr.close()


def prepare_vggt_vipe_inputs(scene_dir: Path) -> Path:
    vggt_dir = scene_dir / "sam3d+fpose" / "vggt_single_image"
    vipe_out = scene_dir / "sam3d+fpose" / "vipe_results"
    depth_dir = vipe_out / "depth" / "exr_file"
    intr_dir = vipe_out / "intrinsics"
    rgb_dir = vipe_out / "rgb"
    depth_dir.mkdir(parents=True, exist_ok=True)
    intr_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir.mkdir(parents=True, exist_ok=True)

    with open(vggt_dir / "camera.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    transform = meta["preprocess_transform_original_to_preprocessed"]
    orig_w, orig_h = meta["original_size_wh"]

    depth = np.load(vggt_dir / "depth.npy").astype(np.float32)
    k = np.load(vggt_dir / "intrinsic_original_pixels.npy").astype(np.float32)

    x0 = int(round(transform["offset_x"]))
    y0 = int(round(transform["offset_y"]))
    x1 = x0 + int(transform["resized_width"])
    y1 = y0 + int(transform["resized_height"])
    depth_crop = depth[max(y0, 0) : min(y1, depth.shape[0]), max(x0, 0) : min(x1, depth.shape[1])]
    if depth_crop.size == 0:
        raise RuntimeError(f"Empty VGGT content crop for {scene_dir}")

    depth_orig = cv2.resize(depth_crop, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    depth_orig[~np.isfinite(depth_orig)] = 0.0
    depth_orig[depth_orig < 1e-6] = 0.0

    write_exr_z(depth_dir / "00000.exr", depth_orig)
    intr = np.array([[k[0, 0], k[1, 1], k[0, 2], k[1, 2]]], dtype=np.float32)
    np.savez(intr_dir / "video.npz", data=intr)
    np.save(vipe_out / "depth_original_pixels.npy", depth_orig)
    np.savetxt(vipe_out / "intrinsics.txt", k, fmt="%.10g")

    image_path = scene_dir / "image.png"
    if image_path.exists():
        # FoundationPose reads --image directly, but this mirrors ViPE output layout for debugging.
        import shutil

        shutil.copyfile(image_path, rgb_dir / "00000.png")

    with open(vipe_out / "vggt_adapter_manifest.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "source": "VGGT single-image depth converted to FoundationPose/ViPE-compatible format",
                "vggt_dir": str(vggt_dir),
                "image_size_wh": [int(orig_w), int(orig_h)],
                "K": k.tolist(),
                "depth_note": "VGGT depth crop resized from padded 518 input back to original image pixels.",
            },
            f,
            indent=2,
        )
    return vipe_out


def run_checked(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


T_YUP_TO_ZUP = np.eye(4, dtype=np.float64)
T_YUP_TO_ZUP[:3, :3] = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


def load_glb_mesh(mesh_path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(str(mesh_path), force="scene")
    if isinstance(loaded, trimesh.Scene):
        if len(loaded.geometry) == 0:
            raise RuntimeError(f"Empty mesh scene: {mesh_path}")
        mesh = loaded.dump(concatenate=True)
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise RuntimeError(f"Unsupported mesh type from {mesh_path}: {type(loaded)}")
    if len(mesh.faces) == 0:
        raise RuntimeError(f"Mesh has no faces: {mesh_path}")
    return mesh


def load_scale(pose_path: Path) -> np.ndarray:
    with open(pose_path, "r", encoding="utf-8") as f:
        pose = json.load(f)
    scale = np.asarray(pose["scale"], dtype=np.float64).reshape(-1)
    if scale.size == 1:
        scale = np.repeat(scale, 3)
    if scale.size != 3:
        raise RuntimeError(f"Expected scalar or xyz scale in {pose_path}, got {scale}")
    return scale


def prepare_meshes(args: argparse.Namespace, scene_dir: Path, last_idx: int) -> Path:
    """Create scaled Y-up and Z-up GLBs with the local conversion path."""
    scaled_dir = scene_dir / "sam3d+fpose" / "scaled"
    scaled_dir.mkdir(parents=True, exist_ok=True)
    final_mesh = scaled_dir / f"{last_idx}_z_up.glb"
    if final_mesh.exists() and not args.overwrite:
        return scaled_dir

    sam3d_dir = scene_dir / "sam3d"
    for idx in range(last_idx + 1):
        glb_path = sam3d_dir / f"{idx}.glb"
        pose_path = sam3d_dir / f"{idx}_pose.json"
        if not glb_path.exists() or not pose_path.exists():
            print(f"[WARN] Missing mesh or pose for object {idx}, skip scale/z-up conversion")
            continue

        mesh = load_glb_mesh(glb_path)
        scale = load_scale(pose_path)
        mesh.vertices = mesh.vertices.astype(np.float64) * scale.reshape(1, 3)

        scaled_path = scaled_dir / f"{idx}_scaled.glb"
        mesh.export(str(scaled_path))

        mesh_zup = mesh.copy()
        mesh_zup.apply_transform(T_YUP_TO_ZUP)
        zup_path = scaled_dir / f"{idx}_z_up.glb"
        mesh_zup.export(str(zup_path))
        print(f"prepared {scaled_path} and {zup_path}")

    return scaled_dir


def collect_tasks(args: argparse.Namespace) -> list[Task]:
    root = Path(args.scene_root)
    tasks: list[Task] = []
    for scene in args.scenes:
        scene_dir = root / scene
        sam3d_dir = scene_dir / "sam3d"
        if not sam3d_dir.exists():
            raise FileNotFoundError(f"Missing SAM3D dir: {sam3d_dir}")

        object_ids = sorted(int(p.stem) for p in sam3d_dir.glob("*.glb") if p.stem.isdigit())
        if not object_ids:
            raise RuntimeError(f"No numeric object meshes in {sam3d_dir}")
        last_idx = max(object_ids)

        vipe_base = prepare_vggt_vipe_inputs(scene_dir)
        scaled_dir = prepare_meshes(args, scene_dir, last_idx)

        for obj_idx in object_ids:
            out_dir = scene_dir / "sam3d+fpose" / "foundationpose" / f"{obj_idx}_fpose_zup"
            pose_path = out_dir / "ob_in_cam" / "000000.txt"
            if pose_path.exists() and not args.overwrite:
                continue
            mesh = scaled_dir / f"{obj_idx}_z_up.glb"
            mask = sam3d_dir / f"{obj_idx}.png"
            if not mesh.exists():
                raise FileNotFoundError(f"Missing z-up mesh: {mesh}")
            if not mask.exists():
                raise FileNotFoundError(f"Missing mask: {mask}")
            tasks.append(
                Task(
                    scene=scene,
                    obj_idx=obj_idx,
                    mesh=mesh,
                    mask=mask,
                    image=scene_dir / "image.png",
                    vipe_base=vipe_base,
                    out_dir=out_dir,
                )
            )
    return tasks


def run_foundationpose_task(args: argparse.Namespace, task: Task, gpu: str) -> tuple[Task, int]:
    script_dir = Path(args.script_dir)
    task.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = task.out_dir / "run.log"
    cmd = [
        args.python,
        str(script_dir / "robosnap" / "pose" / "foundationpose.py"),
        "--vipe_base",
        str(task.vipe_base),
        "--mesh",
        str(task.mesh),
        "--mask_rgba",
        str(task.mask),
        "--image",
        str(task.image),
        "--out_dir",
        str(task.out_dir),
        "--max_frames",
        str(args.max_frames),
        "--est_iter",
        str(args.est_iter),
        "--track_iter",
        str(args.track_iter),
        "--debug",
        str(args.debug),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=str(script_dir), env=env, stdout=log, stderr=subprocess.STDOUT)
    return task, proc.returncode


def main() -> None:
    args = parse_args()
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id")

    tasks = collect_tasks(args)
    print(f"Prepared inputs. Pending FoundationPose tasks: {len(tasks)}")
    for task in tasks:
        print(f"  {task.scene} obj {task.obj_idx} -> {task.out_dir}")

    if not tasks:
        return

    workers = max(1, min(args.workers, len(gpus), len(tasks)))
    failures: list[tuple[Task, int]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_task = {}
        for idx, task in enumerate(tasks):
            gpu = gpus[idx % len(gpus)]
            future = pool.submit(run_foundationpose_task, args, task, gpu)
            future_to_task[future] = (task, gpu)

        for future in concurrent.futures.as_completed(future_to_task):
            task, gpu = future_to_task[future]
            try:
                finished_task, code = future.result()
            except Exception as exc:
                print(f"FAIL {task.scene} obj {task.obj_idx} on GPU {gpu}: {exc}")
                failures.append((task, -1))
                continue
            if code == 0:
                print(f"DONE {finished_task.scene} obj {finished_task.obj_idx} on GPU {gpu}")
            else:
                print(f"FAIL {finished_task.scene} obj {finished_task.obj_idx} on GPU {gpu}: code {code}")
                failures.append((finished_task, code))

    if failures:
        print("Failures:")
        for task, code in failures:
            print(f"  {task.scene} obj {task.obj_idx}: code {code}, log={task.out_dir / 'run.log'}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
