#!/usr/bin/env python3
"""One-shot automatic layered scene generation from a single image."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from robosnap.refinement.gravity_align_scene import paste_original_mask_to_vggt


STAGES = [
    "preprocess",
    "sam3d",
    "vggt",
    "icp",
    "background",
    "gravity",
    "refinement",
    "preview",
]


def quote_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def parse_objects(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_files(paths: list[Path], metadata: dict | None = None) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(metadata or {}, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def requested_object_prompts(args: argparse.Namespace) -> list[str] | None:
    if args.objects:
        return [item.strip() for item in args.objects.split(",") if item.strip()]
    if args.object_file:
        path = args.object_file.expanduser().resolve()
        if path.suffix.lower() != ".json":
            return parse_objects(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("objects", data.get("prompts", []))
        prompts = []
        for item in data if isinstance(data, list) else []:
            if isinstance(item, str):
                prompts.append(item.strip())
            elif isinstance(item, dict):
                prompt = (
                    item.get("prompt")
                    or item.get("segmentation_prompt")
                    or item.get("description")
                    or item.get("name")
                    or item.get("label")
                )
                if prompt:
                    prompts.append(str(prompt).strip())
        return prompts
    return None


def preprocess_cache_valid(scene_dir: Path, args: argparse.Namespace | None = None) -> bool:
    prompts = parse_objects(scene_dir / "object.txt")
    if not prompts or not all((scene_dir / "sam3d" / f"{idx}.png").exists() for idx in range(len(prompts))):
        return False
    paths = {
        "image": scene_dir / "image.png",
        "mask": scene_dir / "inpaint_mask.png",
        "output": scene_dir / "complete_background.png",
    }
    if not all(path.exists() for path in paths.values()):
        return False
    status = read_json(scene_dir / "complete_background_status.json")
    if status.get("status") in {None, "not_run", "stale_after_mask_refresh"}:
        return False
    hashes = status.get("sha256", {})
    if not all(hashes.get(name) == sha256_file(file_path) for name, file_path in paths.items()):
        return False

    if args is None:
        return True
    mask_status = read_json(scene_dir / "mask_status.json")
    actual_policy = mask_status.get("inpaint_region_policy")
    if actual_policy not in {None, "instance-union"}:
        return False
    recorded_dilation = mask_status.get("inpaint_dilation")
    if recorded_dilation is not None and recorded_dilation != args.inpaint_dilation:
        return False

    expected_extra_hash = None
    if args.inpaint_extra_mask:
        extra_path = args.inpaint_extra_mask.expanduser().resolve()
        if not extra_path.exists():
            return False
        expected_extra_hash = sha256_file(extra_path)
    return mask_status.get("extra_inpaint_mask_sha256") == expected_extra_hash


def sam3d_input_fingerprint(image: Path, mask_dir: Path, prompts: list[str]) -> str:
    masks = [mask_dir / f"{idx}.png" for idx in range(len(prompts))]
    missing = [path for path in [image, *masks] if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing SAM3D input(s): " + ", ".join(str(path) for path in missing))
    return fingerprint_files([image, *masks], {"objects": prompts})


def sam3d_outputs_valid(mask_dir: Path, count: int) -> bool:
    required = [mask_dir / "scene_composed.glb"]
    for idx in range(count):
        required.extend([mask_dir / f"{idx}.glb", mask_dir / f"{idx}_pose.json"])
    return all(path.exists() and path.stat().st_size > 0 for path in required)


def icp_input_fingerprint(scene_dir: Path, object_count: int, seed: int) -> str:
    sam3d_dir = scene_dir / "sam3d"
    mesh_dir = scene_dir / "sam3d+fpose" / "scaled"
    vggt_dir = scene_dir / "sam3d+fpose" / "vggt_single_image"
    algorithm_path = (
        Path(__file__).resolve().parents[1] / "alignment" / "run_sam3d_vggt_icp.py"
    )
    paths = [
        algorithm_path,
        vggt_dir / "camera.json",
        vggt_dir / "points_world.npy",
        vggt_dir / "depth.npy",
        vggt_dir / "depth_conf.npy",
        vggt_dir / "content_mask.npy",
    ]
    for object_id in range(object_count):
        paths.extend(
            [
                sam3d_dir / f"{object_id}.png",
                sam3d_dir / f"{object_id}_pose.json",
                mesh_dir / f"{object_id}_z_up.glb",
            ]
        )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing ICP input(s): " + ", ".join(str(path) for path in missing))
    return fingerprint_files(
        paths,
        {
            "algorithm": "quality_gated_fixed_scale_icp_v2",
            "object_count": object_count,
            "seed": seed,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "1"),
            "target_conf_min": 1.0,
            "mask_alpha_threshold": 0,
            "mask_erode_px": 0,
            "sample_points": 60000,
            "voxel_size": 0.01,
            "max_correspondence_distance": 0.20,
            "coarse_factor": 2.5,
            "max_iterations": 80,
            "min_target_points": 100,
            "outlier_std": 2.0,
            "min_accepted_fitness": 0.05,
            "max_accepted_rmse": 0.15,
            "max_alignment_error_ratio": 0.98,
            "min_reprojection_iou_ratio": 0.90,
            "max_accepted_rotation_deg": 45.0,
            "max_accepted_translation": 0.5,
        },
    )


def icp_outputs_valid(scene_dir: Path, object_count: int) -> bool:
    scene_path = scene_dir / "refined_scene.glb"
    report = read_json(scene_dir / "depth" / "object_point_clouds" / "icp_report.json")
    return (
        scene_path.is_file()
        and scene_path.stat().st_size > 0
        and len(report.get("objects", [])) == object_count
    )


def cache_manifest_valid(path: Path, input_fingerprint: str, *, expected_status: str | None = None) -> bool:
    manifest = read_json(path)
    if manifest.get("input_fingerprint") != input_fingerprint:
        return False
    return expected_status is None or manifest.get("status") == expected_status


def write_cache_manifest(path: Path, input_fingerprint: str, **metadata: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"input_fingerprint": input_fingerprint, **metadata}, indent=2),
        encoding="utf-8",
    )


def write_ascii_ply(path: Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if colors is None:
        colors = np.full((len(points), 3), 180, dtype=np.uint8)
    else:
        colors = np.asarray(colors)
        if colors.max(initial=0) <= 1:
            colors = (colors * 255.0).clip(0, 255)
        colors = colors.astype(np.uint8).reshape(-1, 3)
    with path.open("w", encoding="ascii") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]:.10g} {p[1]:.10g} {p[2]:.10g} {int(c[0])} {int(c[1])} {int(c[2])}\n")


class Runner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.root = Path(__file__).resolve().parents[2]
        self.output_dir = args.output_dir.expanduser().resolve()
        self.log_path = self.output_dir / "pipeline.log"
        self.feedback_path = self.root / "feedback.md"
        self.report: dict = {"stages": [], "output_dir": str(self.output_dir)}
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def env(self) -> dict[str, str]:
        env = os.environ.copy()
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = f"{self.root}{os.pathsep}{existing}" if existing else str(self.root)
        if self.args.cuda_visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = self.args.cuda_visible_devices
        env.setdefault("ROBOSNAP_ROOT", str(self.root))
        env.setdefault("CHECKPOINT_DIR", str(self.args.checkpoint_dir))
        return env

    def note(self, stage: str, message: str) -> None:
        self.feedback_path.parent.mkdir(parents=True, exist_ok=True)
        with self.feedback_path.open("a", encoding="utf-8") as f:
            f.write(f"\n## {stage}\n\n{message.strip()}\n")

    def should_stop_after(self, stage: str) -> bool:
        return self.args.stop_after == stage

    def run(self, stage: str, cmd: list[str], *, cwd: Path | None = None, allow_failure: bool = False) -> int:
        cmd = [str(part) for part in cmd]
        line = f"\n[{stage}] {quote_cmd(cmd)}\n"
        print(line.strip())
        with self.log_path.open("a", encoding="utf-8") as log:
            log.write(line)
            if self.args.dry_run:
                self.report["stages"].append({"stage": stage, "cmd": cmd, "returncode": 0, "dry_run": True})
                return 0
            env = self.env()
            executable = Path(cmd[0]).expanduser()
            if executable.is_absolute():
                env["PATH"] = f"{executable.parent}{os.pathsep}{env.get('PATH', '')}"
            env["PYTHONUTF8"] = "1"
            env["LANG"] = "C.UTF-8"
            env["LC_ALL"] = "C.UTF-8"
            env["MAX_JOBS"] = env.get("ROBOSNAP_MAX_JOBS", "1")
            env.setdefault("PYOPENGL_PLATFORM", "egl")
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            proc = subprocess.run(
                cmd, cwd=str(cwd or self.root), env=env, stdout=log, stderr=subprocess.STDOUT
            )
        self.report["stages"].append({"stage": stage, "cmd": cmd, "returncode": proc.returncode})
        if proc.returncode != 0:
            message = f"Command failed with code {proc.returncode}: `{quote_cmd(cmd)}`. See `{self.log_path}`."
            self.note(stage, message)
            if not allow_failure:
                raise RuntimeError(message)
        return proc.returncode

    def write_report(self) -> None:
        report_path = self.output_dir / "pipeline_report.json"
        report_path.write_text(json.dumps(self.report, indent=2), encoding="utf-8")


def copy_depth_outputs(vggt_dir: Path, depth_dir: Path) -> None:
    depth_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "depth.npy",
        "depth_conf.npy",
        "points_world.npy",
        "content_mask.npy",
        "intrinsic_preprocessed.npy",
        "intrinsic_original_pixels.npy",
        "extrinsic.npy",
        "geometry.npz",
        "camera.json",
        "image_preprocessed.png",
        "depth_vis.png",
        "point_cloud.ply",
        "point_cloud.pcd",
    ]:
        src = vggt_dir / name
        if src.exists():
            shutil.copy2(src, depth_dir / name)


def create_background_fallback(scene_dir: Path, vggt_dir: Path, foreground_mask: Path, output_ply: Path, max_points: int, seed: int) -> dict:
    points_world = np.load(vggt_dir / "points_world.npy")
    content_mask = np.load(vggt_dir / "content_mask.npy").astype(bool)
    camera = json.loads((vggt_dir / "camera.json").read_text(encoding="utf-8"))
    image_pre = np.asarray(Image.open(vggt_dir / "image_preprocessed.png").convert("RGB"))
    valid = content_mask & np.isfinite(points_world).all(axis=2)
    if foreground_mask.exists():
        mask_orig = Image.open(foreground_mask).convert("RGBA")
        alpha = np.asarray(mask_orig)[:, :, 3] > 0
        mask_pre = paste_original_mask_to_vggt(
            alpha,
            camera["preprocess_transform_original_to_preprocessed"],
            tuple(camera["preprocessed_size_hw"]),
        )
        valid &= ~mask_pre
    points = points_world[valid].astype(np.float64)
    colors = image_pre[valid].astype(np.uint8)
    if len(points) > max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(points), size=max_points, replace=False)
        points = points[keep]
        colors = colors[keep]
    write_ascii_ply(output_ply, points, colors)
    status = {
        "status": "vggt_fallback",
        "reason": "Lyra background reconstruction was not available or not requested; background.ply was built from VGGT points outside the foreground mask.",
        "points": int(len(points)),
        "output_ply": str(output_ply),
    }
    (scene_dir / "background_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status


def find_lyra_ply(output_dir: Path) -> Path | None:
    preferred = output_dir / "reconstructed_scene.ply"
    if preferred.exists():
        return preferred
    candidates = sorted(output_dir.rglob("*.ply"), key=lambda p: (p.name not in {"reconstructed_scene.ply", "point_cloud.ply"}, len(str(p))))
    return candidates[0] if candidates else None


def write_render_script(scene_dir: Path) -> Path:
    root = Path(__file__).resolve().parents[2]
    canonical = root / "scripts" / "render_gravity_aligned_scene.sh"
    script = scene_dir / "render_gravity_aligned_scene.py"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import os\n"
        "import subprocess\n"
        "import sys\n\n"
        "env = os.environ.copy()\n"
        "env.setdefault(\"PY_RENDER\", sys.executable)\n"
        f"cmd = [\"bash\", {str(canonical)!r}, {str(scene_dir)!r}]\n"
        "raise SystemExit(subprocess.call(cmd, env=env))\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return canonical

def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Run the automatic layered scene pipeline from one RGB image.")
    parser.add_argument("--image", type=Path, default=root / "examples" / "image.png")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs" / "release_demo_2")
    parser.add_argument("--objects", help="Comma-separated object prompts. Use this for local debugging when no VLM command is configured.")
    parser.add_argument("--object-file", type=Path)
    parser.add_argument(
        "--vlm-command",
        help="Command template for object listing. Placeholders: {image}, {prompt}, {output_json}, {output_txt}.",
    )
    parser.add_argument("--vlm-prompt", help="Object-discovery prompt text or path.")
    parser.add_argument("--support-prompts", default="table, desk, tabletop, tabletop surface")
    parser.add_argument("--inpaint-command", help="Command template. Placeholders: {image}, {mask}, {output}, {prompt}, {status}.")
    parser.add_argument("--inpaint-prompt", help="Background-completion prompt text or path.")
    parser.add_argument("--inpaint-dilation", type=int, default=7)
    parser.add_argument("--inpaint-extra-mask", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path(os.environ.get("CHECKPOINT_DIR", root / "checkpoints")))
    parser.add_argument("--sam3-python", default=os.environ.get("PY_SAM3", sys.executable))
    parser.add_argument("--asset-python", default=os.environ.get("PY_ASSET", sys.executable))
    parser.add_argument("--vggt-python", default=os.environ.get("PY_VGGT", os.environ.get("PY_ASSET", sys.executable)))
    parser.add_argument("--align-python", default=os.environ.get("PY_ALIGN", os.environ.get("PY_ASSET", sys.executable)))
    parser.add_argument("--lyra-python", default=os.environ.get("PY_LYRA", sys.executable))
    parser.add_argument("--sim-ready-python", default=os.environ.get("PY_SIM_READY", sys.executable))
    parser.add_argument("--sam3-dir", type=Path, default=Path(os.environ.get("SAM3_DIR", root / "third_party" / "sam3")))
    parser.add_argument("--sam3d-dir", type=Path, default=Path(os.environ.get("SAM3D_DIR", root / "third_party" / "sam-3d-objects")))
    parser.add_argument("--sam3-checkpoint", type=Path, default=Path(os.environ.get("SAM3_CKPT", root / "checkpoints" / "sam3" / "sam3.pt")))
    parser.add_argument("--sam3d-config", type=Path, default=Path(os.environ.get("SAM3D_CONFIG", root / "checkpoints" / "sam-3d-objects" / "pipeline.yaml")))
    parser.add_argument("--vggt-checkpoint", type=Path, default=Path(os.environ["VGGT_CKPT"]) if os.environ.get("VGGT_CKPT") else None)
    parser.add_argument("--lyra-dir", type=Path, default=Path(os.environ.get("LYRA_DIR", root / "third_party" / "lyra")))
    parser.add_argument("--lyra-checkpoint-dir", type=Path, default=Path(os.environ.get("LYRA_CHECKPOINT_DIR", root / "checkpoints" / "lyra")))
    parser.add_argument(
        "--lyra2-checkpoint-root",
        type=Path,
        default=Path(os.environ.get("LYRA2_CHECKPOINT_ROOT", root / "checkpoints" / "lyra2")),
    )
    parser.add_argument("--background-video", type=Path)
    parser.add_argument(
        "--lyra-prompt",
        default=os.environ.get(
            "LYRA_PROMPT",
            "A clean realistic indoor environment with stable architecture, lighting, and materials.",
        ),
    )
    parser.add_argument("--lyra-frames-in", type=int, default=int(os.environ.get("LYRA_FRAMES_IN", "81")))
    parser.add_argument("--lyra-frames-out", type=int, default=int(os.environ.get("LYRA_FRAMES_OUT", "81")))
    parser.add_argument("--lyra-zoom-in-strength", type=float, default=float(os.environ.get("LYRA_ZOOM_IN_STRENGTH", "0.35")))
    parser.add_argument("--lyra-zoom-out-strength", type=float, default=float(os.environ.get("LYRA_ZOOM_OUT_STRENGTH", "0.65")))
    parser.add_argument("--lyra-resolution", default=os.environ.get("LYRA_RESOLUTION", "480,832"))
    parser.add_argument(
        "--lyra-use-dmd",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("LYRA_USE_DMD", "1").lower() not in {"0", "false", "no"},
    )
    parser.add_argument(
        "--lyra-offload",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("LYRA_OFFLOAD", "1").lower() not in {"0", "false", "no"},
    )
    parser.add_argument("--lyra-recon-da3-max-frames", type=int, default=int(os.environ.get("LYRA_RECON_DA3_MAX_FRAMES", "32")))
    parser.add_argument("--device", default=os.environ.get("ROBOSNAP_DEVICE", "cuda:0"))
    parser.add_argument("--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES"))
    parser.add_argument("--max-vggt-points", type=int, default=300000)
    parser.add_argument("--max-background-points", type=int, default=400000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-sam3", action="store_true")
    parser.add_argument("--skip-lyra", action="store_true")
    parser.add_argument("--stop-after", choices=STAGES)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runner = Runner(args)
    scene_dir = runner.output_dir
    image_path = args.image.expanduser().resolve()

    preprocess_cmd = [
        sys.executable,
        "-m",
        "robosnap.preprocess.auto_segment_image",
        "--image",
        str(image_path),
        "--output-dir",
        str(scene_dir),
        "--sam3-python",
        args.sam3_python,
        "--sam3-dir",
        str(args.sam3_dir),
        "--sam3-checkpoint",
        str(args.sam3_checkpoint),
        "--support-prompts",
        args.support_prompts,
    ]
    if args.objects:
        preprocess_cmd.extend(["--objects", args.objects])
    if args.object_file:
        preprocess_cmd.extend(["--object-file", str(args.object_file)])
    if args.vlm_command:
        preprocess_cmd.extend(["--vlm-command", args.vlm_command])
    if args.vlm_prompt:
        preprocess_cmd.extend(["--vlm-prompt", args.vlm_prompt])
    if args.inpaint_command:
        preprocess_cmd.extend(["--inpaint-command", args.inpaint_command])
    if args.inpaint_prompt:
        preprocess_cmd.extend(["--inpaint-prompt", args.inpaint_prompt])
    preprocess_cmd.extend(["--inpaint-dilation", str(args.inpaint_dilation)])
    if args.inpaint_extra_mask:
        preprocess_cmd.extend(["--inpaint-extra-mask", str(args.inpaint_extra_mask)])
    if args.skip_sam3:
        preprocess_cmd.append("--skip-sam3")
    if args.dry_run:
        preprocess_cmd.append("--dry-run")
    scene_image_before = scene_dir / "image.png"
    image_unchanged = (
        scene_image_before.exists()
        and image_path.exists()
        and sha256_file(scene_image_before) == sha256_file(image_path)
    )
    requested_prompts = requested_object_prompts(args)
    prompts_unchanged = requested_prompts is None or requested_prompts == parse_objects(scene_dir / "object.txt")
    preprocess_done = (
        preprocess_cache_valid(scene_dir, args)
        and image_unchanged
        and prompts_unchanged
        and not args.vlm_command
    )
    reuse_preprocess = args.skip_existing and preprocess_done
    if reuse_preprocess:
        runner.report["stages"].append(
            {"stage": "preprocess", "skipped": "matching image, objects, masks, and inpaint manifest"}
        )
    else:
        runner.run("preprocess", preprocess_cmd)
    if runner.should_stop_after("preprocess"):
        runner.write_report()
        return 0

    object_prompts = parse_objects(scene_dir / "object.txt")
    if not object_prompts:
        raise RuntimeError(f"No object prompts found in {scene_dir / 'object.txt'}")

    scene_image = scene_dir / "image.png"
    sam3d_mask_dir = scene_dir / "sam3d"
    sam3d_manifest = sam3d_mask_dir / "cache_manifest.json"
    sam3d_fingerprint = sam3d_input_fingerprint(scene_image, sam3d_mask_dir, object_prompts)
    reuse_sam3d = (
        args.skip_existing
        and cache_manifest_valid(sam3d_manifest, sam3d_fingerprint, expected_status="complete")
        and sam3d_outputs_valid(sam3d_mask_dir, len(object_prompts))
    )
    if not reuse_sam3d:
        image2glb = args.sam3d_dir / "sam3d_objects" / "image2glb.py"
        sam3d_cmd = [
            args.asset_python,
            str(image2glb),
            "--config",
            str(args.sam3d_config),
            "--image",
            str(scene_image),
            "--mask_dir",
            str(sam3d_mask_dir),
            "--num_masks",
            str(len(object_prompts)),
            "--seed",
            "43",
            "--compose_scene",
        ]
        runner.run("sam3d", sam3d_cmd, cwd=args.sam3d_dir)
        if not args.dry_run:
            if not sam3d_outputs_valid(sam3d_mask_dir, len(object_prompts)):
                raise RuntimeError(f"SAM3D did not produce all {len(object_prompts)} object meshes in {sam3d_mask_dir}")
            write_cache_manifest(
                sam3d_manifest,
                sam3d_fingerprint,
                status="complete",
                object_count=len(object_prompts),
                objects=object_prompts,
            )
    else:
        runner.report["stages"].append({"stage": "sam3d", "skipped": "matching input manifest"})

    prepare_cmd = [
        args.asset_python,
        "-m",
        "robosnap.preprocess.prepare_sam3d_meshes",
        "--scene-dir",
        str(scene_dir),
        "--object-ids",
        *[str(idx) for idx in range(len(object_prompts))],
    ]
    if not reuse_sam3d:
        prepare_cmd.append("--overwrite")
    runner.run("sam3d", prepare_cmd)
    if runner.should_stop_after("sam3d"):
        runner.write_report()
        return 0

    vggt_dir = scene_dir / "sam3d+fpose" / "vggt_single_image"
    if not (args.skip_existing and image_unchanged and (vggt_dir / "camera.json").exists()):
        vggt_cmd = [
            args.vggt_python,
            "-m",
            "robosnap.alignment.run_vggt_single_image",
            "--scene-root",
            str(scene_dir.parent),
            "--scenes",
            scene_dir.name,
            "--image-name",
            "image.png",
            "--device",
            args.device,
            "--max-points",
            str(args.max_vggt_points),
            "--seed",
            str(args.seed),
        ]
        if args.vggt_checkpoint:
            vggt_cmd.extend(["--checkpoint", str(args.vggt_checkpoint)])
        runner.run("vggt", vggt_cmd)
    else:
        runner.report["stages"].append({"stage": "vggt", "skipped": "existing camera.json"})
    copy_depth_outputs(vggt_dir, scene_dir / "depth")
    if runner.should_stop_after("vggt"):
        runner.write_report()
        return 0

    refined_scene = scene_dir / "refined_scene.glb"
    icp_dir = scene_dir / "depth" / "object_point_clouds"
    icp_manifest = icp_dir / "cache_manifest.json"
    icp_fingerprint = icp_input_fingerprint(scene_dir, len(object_prompts), args.seed)
    reuse_icp = (
        args.skip_existing
        and cache_manifest_valid(icp_manifest, icp_fingerprint, expected_status="complete")
        and icp_outputs_valid(scene_dir, len(object_prompts))
    )
    if not reuse_icp:
        icp_cmd = [
            args.align_python,
            "-m",
            "robosnap.alignment.run_sam3d_vggt_icp",
            "--scene-dir",
            str(scene_dir),
            "--collection-dir",
            str(scene_dir),
            "--object-ids",
            *[str(idx) for idx in range(len(object_prompts))],
            "--output-subdir",
            "depth/object_point_clouds",
            "--scene-output-name",
            "refined_scene.glb",
            "--seed",
            str(args.seed),
        ]
        runner.run("icp", icp_cmd)
        if not args.dry_run:
            if not icp_outputs_valid(scene_dir, len(object_prompts)):
                raise RuntimeError("ICP did not produce a complete refined scene and report")
            icp_report = read_json(icp_dir / "icp_report.json")
            accepted = [obj["object_id"] for obj in icp_report["objects"] if obj.get("icp_accepted")]
            rejected = [obj["object_id"] for obj in icp_report["objects"] if not obj.get("icp_accepted")]
            write_cache_manifest(
                icp_manifest,
                icp_fingerprint,
                status="complete",
                object_count=len(object_prompts),
                accepted_object_ids=accepted,
                rejected_object_ids=rejected,
            )
    else:
        runner.report["stages"].append({"stage": "icp", "skipped": "matching input manifest"})
    if runner.should_stop_after("icp"):
        runner.write_report()
        return 0

    background_ply = scene_dir / "background.ply"
    complete_background = scene_dir / "complete_background.png"
    if not complete_background.exists():
        raise FileNotFoundError(complete_background)

    lyra_root = args.lyra_dir.expanduser().resolve()
    lyra2_dir = lyra_root / "Lyra-2"
    lyra_output = scene_dir / "background" / "lyra2_gs"
    lyra_video_dir = scene_dir / "background" / "lyra2_video"
    lyra_cameras = lyra_output / "cameras.npz"
    lyra_trajectory = lyra_output / "gs_trajectory.mp4"
    lyra_video_manifest = lyra_video_dir / "cache_manifest.json"
    background_manifest = scene_dir / "background" / "cache_manifest.json"
    external_video = args.background_video.expanduser().resolve() if args.background_video else None
    if external_video and not external_video.is_file():
        raise FileNotFoundError(external_video)

    video_metadata = {
        "stage": "lyra2_video",
        "prompt": args.lyra_prompt,
        "frames_in": args.lyra_frames_in,
        "frames_out": args.lyra_frames_out,
        "zoom_in_strength": args.lyra_zoom_in_strength,
        "zoom_out_strength": args.lyra_zoom_out_strength,
        "resolution": args.lyra_resolution,
        "use_dmd": args.lyra_use_dmd,
        "offload": args.lyra_offload,
    }
    fingerprint_inputs = [complete_background]
    if external_video:
        fingerprint_inputs.append(external_video)
        video_metadata["external_video"] = str(external_video)
    video_fingerprint = fingerprint_files(fingerprint_inputs, video_metadata)
    background_metadata = {**video_metadata, "recon_da3_max_frames": args.lyra_recon_da3_max_frames}
    background_fingerprint = fingerprint_files(fingerprint_inputs, background_metadata)
    expected_background_status = "vggt_fallback" if args.skip_lyra else "lyra2_video_da3_gs"
    background_ready = (
        args.skip_existing
        and background_ply.exists()
        and (args.skip_lyra or (lyra_cameras.is_file() and lyra_trajectory.is_file()))
        and cache_manifest_valid(
            background_manifest,
            background_fingerprint,
            expected_status=expected_background_status,
        )
    )
    lyra_video: Path | None = external_video
    if background_ready:
        runner.report["stages"].append({"stage": "background", "skipped": "matching input manifest"})
    elif args.skip_lyra:
        fallback_status = create_background_fallback(
            scene_dir,
            vggt_dir,
            scene_dir / "foreground_mask.png",
            background_ply,
            args.max_background_points,
            args.seed,
        )
        write_cache_manifest(
            background_manifest,
            background_fingerprint,
            status=fallback_status["status"],
            output_ply=str(background_ply),
        )
        background_ready = True
    else:
        if not (lyra2_dir / "lyra_2" / "_src" / "inference" / "lyra2_zoomgs_inference.py").is_file():
            raise FileNotFoundError(f"Lyra-2 source is unavailable: {lyra2_dir}")

        if lyra_video is None:
            lyra_video = lyra_video_dir / "exploration.mp4"
            reuse_video = (
                args.skip_existing
                and lyra_video.is_file()
                and cache_manifest_valid(
                    lyra_video_manifest,
                    video_fingerprint,
                    expected_status="lyra2_video",
                )
            )
            if reuse_video:
                runner.report["stages"].append(
                    {"stage": "background_video", "skipped": "matching Lyra-2 video manifest"}
                )
            else:
                video_cmd = [
                    sys.executable,
                    "-m",
                    "robosnap.background.run_lyra2_video_generation",
                    "--input-image",
                    str(complete_background),
                    "--prompt",
                    args.lyra_prompt,
                    "--output-dir",
                    str(lyra_video_dir),
                    "--lyra-dir",
                    str(lyra_root),
                    "--python",
                    args.lyra_python,
                    "--checkpoint-root",
                    str(args.lyra2_checkpoint_root),
                    "--num-frames-zoom-in",
                    str(args.lyra_frames_in),
                    "--num-frames-zoom-out",
                    str(args.lyra_frames_out),
                    "--zoom-in-strength",
                    str(args.lyra_zoom_in_strength),
                    "--zoom-out-strength",
                    str(args.lyra_zoom_out_strength),
                    "--resolution",
                    args.lyra_resolution,
                    "--seed",
                    str(args.seed),
                    "--force",
                ]
                if not args.lyra_use_dmd:
                    video_cmd.append("--no-use-dmd")
                if not args.lyra_offload:
                    video_cmd.extend(
                        ["--no-offload", "--no-offload-when-prompt", "--no-offload-da3-diffusion"]
                    )
                runner.run("background_video", video_cmd)
                if not args.dry_run:
                    if not lyra_video.is_file() or lyra_video.stat().st_size == 0:
                        raise RuntimeError(f"Lyra-2 Step 1 did not produce {lyra_video}")
                    write_cache_manifest(
                        lyra_video_manifest,
                        video_fingerprint,
                        status="lyra2_video",
                        output_video=str(lyra_video),
                    )

        args.lyra_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        lyra_cmd = [
            sys.executable,
            "-m",
            "robosnap.background.run_lyra2_gs_recon",
            "--input-video",
            str(lyra_video),
            "--output-dir",
            str(lyra_output),
            "--lyra-dir",
            str(lyra_root),
            "--python",
            args.lyra_python,
            "--checkpoint-dir",
            str(args.lyra_checkpoint_dir),
            "--device",
            args.device,
            "--max-frames",
            "0",
            "--da3-max-frames",
            str(args.lyra_recon_da3_max_frames),
            "--vipe",
            "--force",
            "--require-render",
        ]
        candidate_before = find_lyra_ply(lyra_output)
        candidate_before_state = None
        if candidate_before:
            stat = candidate_before.stat()
            candidate_before_state = (str(candidate_before), stat.st_mtime_ns, stat.st_size)
        runner.run("background", lyra_cmd)
        if args.dry_run:
            background_ready = True
        else:
            candidate = find_lyra_ply(lyra_output)
            cameras_path = lyra_cameras
            trajectory_path = lyra_trajectory
            candidate_updated = False
            if candidate:
                stat = candidate.stat()
                candidate_updated = (str(candidate), stat.st_mtime_ns, stat.st_size) != candidate_before_state
            if not candidate or not candidate_updated:
                raise RuntimeError("Lyra-2 Step 2 did not refresh reconstructed_scene.ply")
            if not cameras_path.is_file() or not trajectory_path.is_file():
                raise RuntimeError("Lyra-2 Step 2 did not produce cameras.npz and gs_trajectory.mp4")

            shutil.copy2(candidate, background_ply)
            status = {
                "status": "lyra2_video_da3_gs",
                "reason": "Lyra-2 generated an exploration video and DA3 reconstructed a rendered Gaussian scene.",
                "input_video": str(lyra_video),
                "points_source": str(candidate),
                "output_ply": str(background_ply),
                "cameras": str(lyra_cameras),
                "gs_trajectory": str(lyra_trajectory),
                "lyra_output_dir": str(lyra_output),
                "input_fingerprint": background_fingerprint,
            }
            (scene_dir / "background_status.json").write_text(
                json.dumps(status, indent=2),
                encoding="utf-8",
            )
            write_cache_manifest(
                background_manifest,
                background_fingerprint,
                status="lyra2_video_da3_gs",
                output_ply=str(background_ply),
                cameras=str(cameras_path),
                input_video=str(lyra_video),
            )
            background_ready = True

    if not background_ready:
        raise RuntimeError("Background stage did not produce a valid output")
    background_status_path = scene_dir / "background_status.json"
    if background_status_path.exists():
        try:
            runner.report["background_status"] = json.loads(background_status_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            runner.report["background_status"] = {"status": "unreadable", "path": str(background_status_path)}
    if runner.should_stop_after("background"):
        runner.write_report()
        return 0

    gravity_fg = scene_dir / "gravity_aligned_foreground.glb"
    gravity_bg = scene_dir / "gravity_aligned_background.ply"
    gravity_json = scene_dir / "gravity_alignment.json"
    gravity_cmd = [
        args.align_python,
        "-m",
        "robosnap.refinement.gravity_align_scene",
        "--scene-dir",
        str(scene_dir),
        "--foreground-glb",
        str(refined_scene),
        "--background-ply",
        str(background_ply),
        "--vggt-dir",
        str(vggt_dir),
        "--plane-mask",
        str(scene_dir / "support_mask.png"),
        "--output-foreground",
        str(gravity_fg),
        "--output-background",
        str(gravity_bg),
        "--transform-json",
        str(gravity_json),
        "--seed",
        str(args.seed),
    ]
    lyra_cameras = lyra_output / "cameras.npz"
    if lyra_cameras.exists():
        gravity_cmd.extend(["--background-cameras", str(lyra_cameras)])
    runner.run("gravity", gravity_cmd)
    if runner.should_stop_after("gravity"):
        runner.write_report()
        return 0

    final_fg = scene_dir / "fully_refined_foreground.glb"
    refinement_cmd = [
        sys.executable,
        "-m",
        "robosnap.refinement.run_sim_ready_refinement",
        "--input-foreground",
        str(gravity_fg),
        "--output-foreground",
        str(final_fg),
        "--refinement-dir",
        str(scene_dir / "refinement"),
        "--scene-dir",
        str(scene_dir),
        "--sf-python",
        args.sim_ready_python,
    ]
    sf_extra_pythonpath = os.environ.get("SF_REAL2SIM_EXTRA_PYTHONPATH")
    if sf_extra_pythonpath:
        refinement_cmd.extend(["--sf-extra-pythonpath", sf_extra_pythonpath])
    runner.run("refinement", refinement_cmd)
    refinement_status_path = scene_dir / "refinement" / "status.json"
    if refinement_status_path.exists():
        try:
            refinement_status = json.loads(refinement_status_path.read_text(encoding="utf-8"))
            runner.report["refinement_status"] = refinement_status
            if refinement_status.get("status") != "sf_ok":
                reason = refinement_status.get("reason", "Sim-ready refinement used a fallback.")
                runner.note("refinement", reason)
                if not args.dry_run:
                    runner.write_report()
                    raise RuntimeError(f"SF-Real2Sim refinement did not succeed: {reason}")
        except json.JSONDecodeError:
            runner.report["refinement_status"] = {"status": "unreadable", "path": str(refinement_status_path)}
            if not args.dry_run:
                runner.write_report()
                raise RuntimeError(f"SF-Real2Sim refinement status is unreadable: {refinement_status_path}")
    if runner.should_stop_after("refinement"):
        runner.write_report()
        return 0

    canonical_render_script = write_render_script(scene_dir)
    preview_status = scene_dir / "layered_preview_status.json"
    if lyra_cameras.exists():
        if not args.dry_run:
            for stale in (scene_dir / "layered_preview.png", scene_dir / "layered_preview.ply", preview_status):
                stale.unlink(missing_ok=True)
        preview_cmd = [
            args.align_python,
            "-m",
            "robosnap.rendering.render_layered_scene",
            "--foreground",
            str(final_fg),
            "--background-ply",
            str(gravity_bg),
            "--camera-npz",
            str(lyra_cameras),
            "--gravity-transform",
            str(gravity_json),
            "--foreground-camera-json",
            str(vggt_dir / "camera.json"),
            "--output-ply",
            str(scene_dir / "layered_preview.ply"),
            "--output-image",
            str(scene_dir / "layered_preview.png"),
            "--status-json",
            str(preview_status),
            "--device",
            args.device,
        ]
        runner.run("preview", preview_cmd)
    else:
        runner.report["stages"].append(
            {"stage": "preview", "skipped": "Gaussian camera data unavailable in explicit --skip-lyra mode"}
        )

    runner.report["final_outputs"] = {
        "gravity_aligned_background": str(gravity_bg),
        "fully_refined_foreground": str(final_fg),
        "render_script": str(canonical_render_script),
        "scene_render_script": str(scene_dir / "render_gravity_aligned_scene.py"),
        "lyra_exploration_video": str(
            external_video or (lyra_video_dir / "exploration.mp4")
        ),
        "lyra_gs_trajectory": str(lyra_output / "gs_trajectory.mp4"),
        "layered_preview": str(scene_dir / "layered_preview.ply"),
        "layered_preview_image": str(scene_dir / "layered_preview.png"),
        "layered_preview_status": str(preview_status),
    }
    runner.write_report()
    print(f"[auto-pipeline] wrote {scene_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
