#!/usr/bin/env python3
"""Run Lyra-2 VIPE+DA3 Gaussian reconstruction from the RoboSnap layout."""

from __future__ import annotations

import argparse
import os
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


def derive_conda_prefix(python_executable: str | None) -> str | None:
    if not python_executable:
        return None
    path = Path(python_executable).expanduser()
    if path.is_absolute() and path.name.startswith("python") and path.parent.name == "bin":
        return str(path.parents[1])
    return None


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value in (None, "") else int(value)


def env_float(name: str) -> float | None:
    value = os.environ.get(name)
    return None if value in (None, "") else float(value)


def build_env(args: argparse.Namespace, lyra2_dir: Path) -> dict[str, str]:
    env = clear_socks_proxy(os.environ.copy())
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUTF8"] = "1"
    env["LANG"] = "C.UTF-8"
    env["LC_ALL"] = "C.UTF-8"
    env.setdefault("MAX_JOBS", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{lyra2_dir}{os.pathsep}{existing}" if existing else str(lyra2_dir)
    env.setdefault("HF_HOME", str(Path(args.checkpoint_dir) / "hf_cache"))
    env.setdefault("TORCH_HOME", str(Path(args.checkpoint_dir) / "torch_cache"))
    prefix = args.conda_prefix or derive_conda_prefix(args.python)
    if prefix:
        env["CONDA_PREFIX"] = prefix
        prefix_path = Path(prefix)
        env["PATH"] = f"{prefix_path / 'bin'}{os.pathsep}{env.get('PATH', '')}"
        sites = sorted((prefix_path / "lib").glob("python*/site-packages"))
        site = sites[-1] if sites else prefix_path / "lib" / "python3.10" / "site-packages"
        include_paths = [
            prefix_path / "include",
            prefix_path / "targets" / "x86_64-linux" / "include",
            site / "nvidia" / "cuda_runtime" / "include",
            site / "nvidia" / "cudnn" / "include",
            site / "nvidia" / "nccl" / "include",
            site / "nvidia" / "nvtx" / "include",
        ]
        existing_cpath = env.get("CPATH")
        env["CPATH"] = ":".join(map(str, include_paths)) + (f":{existing_cpath}" if existing_cpath else "")
        lib_paths = [
            prefix_path / "lib",
            prefix_path / "lib64",
            site / "torch" / "lib",
            site / "nvidia" / "cuda_runtime" / "lib",
            site / "nvidia" / "cudnn" / "lib",
            site / "nvidia" / "nccl" / "lib",
            site / "nvidia" / "nvtx" / "lib",
        ]
        existing_ld = env.get("LD_LIBRARY_PATH")
        env["LD_LIBRARY_PATH"] = ":".join(str(p) for p in lib_paths) + (f":{existing_ld}" if existing_ld else "")
        nvcc = prefix_path / "bin" / "nvcc"
        if nvcc.is_file():
            env["CUDA_HOME"] = str(prefix_path)
        for name, compiler in (
            ("CC", prefix_path / "bin" / "x86_64-conda-linux-gnu-gcc"),
            ("CXX", prefix_path / "bin" / "x86_64-conda-linux-gnu-g++"),
        ):
            if compiler.is_file():
                env[name] = str(compiler)
    return env


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Run Lyra-2 Step 2: video to Gaussian Splatting reconstruction.")
    parser.add_argument("--input-video", type=Path, required=True, help="Input exploration/background video.")
    parser.add_argument("--output-dir", type=Path, help="Output directory. Defaults next to the video as <stem>_gs_ours.")
    parser.add_argument("--lyra-dir", type=Path, default=Path(os.environ.get("LYRA_DIR", root / "third_party" / "lyra")))
    parser.add_argument("--python", default=os.environ.get("PY_LYRA", os.environ.get("PY_LYRA2", sys.executable)))
    parser.add_argument("--checkpoint-dir", default=os.environ.get("LYRA_CHECKPOINT_DIR", root / "checkpoints" / "lyra"))
    parser.add_argument("--conda-prefix", default=os.environ.get("PY_LYRA_CONDA_PREFIX", os.environ.get("PY_LYRA2_CONDA_PREFIX")))
    parser.add_argument("--da3-model-path", default=os.environ.get("LYRA_DA3_MODEL_PATH"))
    parser.add_argument("--device", default=os.environ.get("LYRA_DEVICE"))
    parser.add_argument("--max-frames", type=int, default=env_int("LYRA_MAX_FRAMES", 0))
    parser.add_argument("--da3-max-frames", type=int, default=env_int("LYRA_DA3_MAX_FRAMES", 128))
    parser.add_argument("--max-resolution", type=int, default=env_int("LYRA_MAX_RESOLUTION", 0))
    parser.add_argument("--gs-down-ratio", type=int, default=env_int("LYRA_GS_DOWN_RATIO", 2))
    parser.add_argument("--render-fps", type=float, default=env_float("LYRA_RENDER_FPS"))
    parser.add_argument("--force", action="store_true", default=os.environ.get("LYRA_FORCE", "0") == "1")
    parser.add_argument("--no-vipe", action="store_true", default=os.environ.get("LYRA_NO_VIPE", "0") == "1")
    parser.add_argument("--vipe", dest="no_vipe", action="store_false")
    parser.add_argument("--require-render", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print command without running the heavy model.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.input_video = args.input_video.expanduser().resolve()
    if args.output_dir:
        args.output_dir = args.output_dir.expanduser().resolve()
    args.lyra_dir = args.lyra_dir.expanduser().resolve()
    args.checkpoint_dir = str(Path(args.checkpoint_dir).expanduser().resolve())
    lyra_dir = args.lyra_dir
    lyra2_dir = lyra_dir / "Lyra-2"
    module_path = lyra2_dir / "lyra_2" / "_src" / "inference" / "vipe_da3_gs_recon.py"
    if not module_path.exists():
        print(f"[FAIL] Lyra-2 GS recon entrypoint not found: {module_path}", file=sys.stderr)
        return 2
    if not args.input_video.exists():
        print(f"[FAIL] Input video not found: {args.input_video}", file=sys.stderr)
        return 2

    cmd = [
        args.python,
        "-m",
        "lyra_2._src.inference.vipe_da3_gs_recon",
        "--input_video_path",
        str(args.input_video),
        "--da3_max_frames",
        str(args.da3_max_frames),
        "--gs_down_ratio",
        str(args.gs_down_ratio),
        "--da3_model_path_custom",
        str(args.da3_model_path) if args.da3_model_path else "",
    ]
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        cmd.extend(["--output_dir", str(args.output_dir)])
    if args.device:
        cmd.extend(["--device", args.device])
    if args.max_frames:
        cmd.extend(["--max_frames", str(args.max_frames)])
    if args.max_resolution:
        cmd.extend(["--max_resolution", str(args.max_resolution)])
    if args.render_fps is not None:
        cmd.extend(["--render_fps", str(args.render_fps)])
    if args.force:
        cmd.append("--force")
    if args.no_vipe:
        cmd.append("--no_vipe")

    print("[INFO] Running Lyra-2 GS reconstruction:")
    print(" ".join(cmd))
    print(f"[INFO] cwd={lyra2_dir}")
    if args.dry_run:
        return 0
    ply_path = args.output_dir / "reconstructed_scene.ply" if args.output_dir else None
    before_ply = None
    if ply_path and ply_path.exists():
        stat = ply_path.stat()
        before_ply = (stat.st_mtime_ns, stat.st_size)
    render_path = args.output_dir / "gs_trajectory.mp4" if args.output_dir else None
    before_render = None
    if render_path and render_path.exists():
        stat = render_path.stat()
        before_render = (stat.st_mtime_ns, stat.st_size)
    proc = subprocess.run(cmd, cwd=str(lyra2_dir), env=build_env(args, lyra2_dir))
    render_updated = False
    if render_path and render_path.exists():
        stat = render_path.stat()
        render_updated = stat.st_size > 0 and (stat.st_mtime_ns, stat.st_size) != before_render
    if args.require_render and not render_updated:
        print(f"[FAIL] Lyra-2 did not refresh required GS render: {render_path}", file=sys.stderr)
        return proc.returncode or 1
    if proc.returncode != 0 and ply_path and ply_path.exists():
        stat = ply_path.stat()
        after_ply = (stat.st_mtime_ns, stat.st_size)
        if stat.st_size > 0 and after_ply != before_ply and not args.require_render:
            print(
                f"[WARN] Lyra-2 returned {proc.returncode} after updating {ply_path}. "
                "Treating background PLY generation as successful; optional GS render may have failed.",
                file=sys.stderr,
            )
            return 0
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
