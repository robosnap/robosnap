#!/usr/bin/env python3
"""Generate a camera-controlled Lyra-2 exploration video from one image."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

def clear_socks_proxy(env: dict[str, str]) -> dict[str, str]:
    if env.get("ROBOSNAP_KEEP_PROXY") == "1":
        return env
    for key in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
        value = env.get(key, "")
        if value.lower().startswith("socks"):
            env.pop(key, None)
    return env


def ensure_lyra2_compat(lyra2_dir: Path) -> None:
    markers = (
        (lyra2_dir / "lyra_2" / "_src" / "models" / "lyra2_model.py", "net = net.to(dtype=self.precision)"),
        (lyra2_dir / "lyra_2" / "_src" / "inference" / "lyra2_zoomgs_inference.py", "swap_da3_diffusion ="),
        (lyra2_dir / "lyra_2" / "_src" / "inference" / "lyra2_zoomgs_inference.py", "merged_lora_layers ="),
        (lyra2_dir / "lyra_2" / "_src" / "inference" / "lyra2_ar_inference.py", "self.local_da3_model.cpu()"),
        (lyra2_dir / "lyra_2" / "_src" / "inference" / "lyra2_ar_inference.py", "def _offload_vae_core_to_cpu"),
        (lyra2_dir / "lyra_2" / "_src" / "inference" / "lyra2_ar_inference.py", "self.model._latest_condition_state_pixels = None"),
        (lyra2_dir / "lyra_2" / "_src" / "networks" / "wan2pt1_lyra2.py", "width * 2 * int(multires)"),
    )
    if all(marker in path.read_text(encoding="utf-8") for path, marker in markers):
        return
    patch_file = Path(__file__).resolve().parents[2] / "third_party" / "patches" / "lyra2-4090-offload.patch"
    if not patch_file.is_file():
        raise FileNotFoundError(f"Lyra-2 compatibility patch not found: {patch_file}")
    cmd = ["patch", "--forward", "--batch", "-p1", "-i", str(patch_file)]
    proc = subprocess.run(cmd, cwd=lyra2_dir, text=True, capture_output=True)
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip()
        raise RuntimeError(f"Failed to apply Lyra-2 compatibility patch: {detail}")
    if not all(marker in path.read_text(encoding="utf-8") for path, marker in markers):
        raise RuntimeError("Lyra-2 compatibility patch did not produce the expected source markers")
    print(f"[lyra-video] applied compatibility patch: {patch_file}")



def derive_conda_prefix(python_executable: str) -> Path | None:
    path = Path(python_executable).expanduser()
    if path.is_absolute() and path.name.startswith("python") and path.parent.name == "bin":
        return path.parents[1]
    return None


def required_checkpoints(root: Path, use_dmd: bool) -> list[Path]:
    files = [
        root / "image_encoder" / "model.pth",
        root / "lora" / "detail_enhancer.safetensors",
        root / "lora" / "realism_boost.safetensors",
        root / "model" / "model" / ".metadata",
        root / "recon" / "model.pt",
        root / "text_encoder" / "encoder.pth",
        root / "text_encoder" / "negative_prompt.pt",
        root / "vae" / "images_mean_std.pt",
        root / "vae" / "vae.pth",
        root / "vae" / "video_mean_std.pt",
    ]
    files.extend(root / "model" / "model" / f"__{idx}_0.distcp" for idx in range(64))
    if use_dmd:
        files.append(root / "lora" / "dmd_distillation.safetensors")
    return files


def replace_symlink(link: Path, target: Path) -> None:
    if link.is_symlink() and link.resolve() == target.resolve():
        return
    if link.exists() or link.is_symlink():
        raise FileExistsError(f"Runtime path already exists and is not the expected symlink: {link}")
    link.symlink_to(target, target_is_directory=True)


def build_env(python_executable: str, lyra2_dir: Path) -> dict[str, str]:
    env = clear_socks_proxy(os.environ.copy())
    prefix = derive_conda_prefix(python_executable)
    if prefix:
        env["CONDA_PREFIX"] = str(prefix)
        env["PATH"] = f"{prefix / 'bin'}{os.pathsep}{env.get('PATH', '')}"
        site_paths = sorted((prefix / "lib").glob("python*/site-packages"))
        site = site_paths[-1] if site_paths else prefix / "lib" / "python3.10" / "site-packages"
        include_paths = [
            prefix / "include",
            prefix / "targets" / "x86_64-linux" / "include",
            site / "nvidia" / "cuda_runtime" / "include",
            site / "nvidia" / "cudnn" / "include",
            site / "nvidia" / "nccl" / "include",
            site / "nvidia" / "nvtx" / "include",
        ]
        lib_paths = [
            prefix / "lib",
            prefix / "lib64",
            site / "torch" / "lib",
            site / "nvidia" / "cuda_runtime" / "lib",
            site / "nvidia" / "cudnn" / "lib",
            site / "nvidia" / "nccl" / "lib",
            site / "nvidia" / "nvtx" / "lib",
        ]
        existing_cpath = env.get("CPATH")
        env["CPATH"] = ":".join(map(str, include_paths)) + (f":{existing_cpath}" if existing_cpath else "")
        existing_ld = env.get("LD_LIBRARY_PATH")
        env["LD_LIBRARY_PATH"] = ":".join(map(str, lib_paths)) + (f":{existing_ld}" if existing_ld else "")
        nvcc = prefix / "bin" / "nvcc"
        if nvcc.is_file():
            env["CUDA_HOME"] = str(prefix)
        for name, compiler in (
            ("CC", prefix / "bin" / "x86_64-conda-linux-gnu-gcc"),
            ("CXX", prefix / "bin" / "x86_64-conda-linux-gnu-g++"),
        ):
            if compiler.is_file():
                env[name] = str(compiler)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{lyra2_dir}{os.pathsep}{existing}" if existing else str(lyra2_dir)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUTF8"] = "1"
    env["LANG"] = "C.UTF-8"
    env["LC_ALL"] = "C.UTF-8"
    env.setdefault("MAX_JOBS", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return env


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Run Lyra-2 Step 1 image-to-exploration-video generation.")
    parser.add_argument("--input-image", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lyra-dir", type=Path, default=Path(os.environ.get("LYRA_DIR", root / "third_party" / "lyra")))
    parser.add_argument("--python", default=os.environ.get("PY_LYRA", sys.executable))
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path(os.environ.get("LYRA2_CHECKPOINT_ROOT", root / "checkpoints" / "lyra2")),
    )
    parser.add_argument("--num-frames-zoom-in", type=int, default=81)
    parser.add_argument("--num-frames-zoom-out", type=int, default=81)
    parser.add_argument("--zoom-in-strength", type=float, default=0.35)
    parser.add_argument("--zoom-out-strength", type=float, default=0.65)
    parser.add_argument("--resolution", default="480,832", help="Generation resolution as H,W.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--use-dmd", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--offload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--offload-when-prompt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--offload-da3-diffusion", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--merge-lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_image = args.input_image.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    lyra2_dir = args.lyra_dir.expanduser().resolve() / "Lyra-2"
    checkpoint_root = args.checkpoint_root.expanduser().resolve()
    if not input_image.is_file():
        raise FileNotFoundError(input_image)
    if not (lyra2_dir / "lyra_2" / "_src" / "inference" / "lyra2_zoomgs_inference.py").is_file():
        raise FileNotFoundError(f"Lyra-2 Step 1 entrypoint not found under {lyra2_dir}")
    ensure_lyra2_compat(lyra2_dir)

    missing = [path for path in required_checkpoints(checkpoint_root, args.use_dmd) if not path.is_file()]
    if missing:
        preview = "\n".join(str(path) for path in missing[:8])
        raise FileNotFoundError(f"Lyra-2 checkpoint is incomplete ({len(missing)} missing):\n{preview}")

    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir = output_dir / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    replace_symlink(runtime_dir / "lyra_2", lyra2_dir / "lyra_2")
    replace_symlink(runtime_dir / "checkpoints", checkpoint_root)

    expected = output_dir / "videos" / f"{input_image.stem}.mp4"
    canonical = output_dir / "exploration.mp4"
    if args.force:
        expected.unlink(missing_ok=True)
        canonical.unlink(missing_ok=True)

    cmd = [
        args.python,
        "-m",
        "lyra_2._src.inference.lyra2_zoomgs_inference",
        "--input_image_path",
        str(input_image),
        "--sample_id",
        "0",
        "--prompt",
        args.prompt,
        "--experiment",
        "lyra2",
        "--checkpoint_dir",
        str(checkpoint_root / "model"),
        "--output_path",
        str(output_dir),
        "--num_frames_zoom_in",
        str(args.num_frames_zoom_in),
        "--num_frames_zoom_out",
        str(args.num_frames_zoom_out),
        "--zoom_in_strength",
        str(args.zoom_in_strength),
        "--zoom_out_strength",
        str(args.zoom_out_strength),
        "--seed",
        str(args.seed),
        "--resolution",
        args.resolution,
        "--da3_model_path_custom",
        str(checkpoint_root / "recon" / "model.pt"),
    ]
    if args.use_dmd:
        cmd.append("--use_dmd")
    if args.offload:
        cmd.append("--offload")
    if args.offload_when_prompt:
        cmd.append("--offload_when_prompt")
    if args.offload_da3_diffusion:
        cmd.append("--offload_da3_diffusion")
    if args.merge_lora:
        cmd.append("--merge_lora")

    print("[lyra-video] " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=runtime_dir, env=build_env(args.python, lyra2_dir))
    if proc.returncode != 0:
        return proc.returncode
    if not expected.is_file() or expected.stat().st_size == 0:
        print(f"[FAIL] Lyra-2 did not produce {expected}", file=sys.stderr)
        return 1
    shutil.copy2(expected, canonical)
    status = {
        "status": "lyra2_video",
        "input_image": str(input_image),
        "output_video": str(canonical),
        "prompt": args.prompt,
        "use_dmd": args.use_dmd,
        "frames": [args.num_frames_zoom_in, args.num_frames_zoom_out],
    }
    (output_dir / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(f"[lyra-video] wrote {canonical}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
