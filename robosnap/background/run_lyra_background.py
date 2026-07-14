#!/usr/bin/env python3
"""Run the legacy VIPE compatibility background path from the RoboSnap layout."""

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


def build_env(args: argparse.Namespace, vipe_dir: Path) -> dict[str, str]:
    env = clear_socks_proxy(os.environ.copy())
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{vipe_dir}{os.pathsep}{existing}" if existing else str(vipe_dir)
    env.setdefault("HF_HOME", str(Path(args.checkpoint_dir) / "hf_cache"))
    env.setdefault("TORCH_HOME", str(Path(args.checkpoint_dir) / "torch_cache"))
    prefix = args.conda_prefix or derive_conda_prefix(args.python)
    if prefix:
        env["CONDA_PREFIX"] = prefix
    return env


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Run the legacy VIPE compatibility pipeline for background geometry/depth artifacts.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path, help="Input video file.")
    source.add_argument("--image-dir", type=Path, help="Directory containing extracted RGB frames.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for VIPE compatibility artifacts.")
    parser.add_argument("--pipeline", default="lyra", help="VIPE pipeline config name. Default: lyra.")
    parser.add_argument("--vipe-dir", type=Path, default=Path(os.environ.get("VIPE_DIR", root / "third_party" / "vipe")))
    parser.add_argument("--python", default=os.environ.get("PY_VIPE", os.environ.get("PY_LYRA", sys.executable)))
    parser.add_argument("--checkpoint-dir", default=os.environ.get("CHECKPOINT_DIR", root / "checkpoints"))
    parser.add_argument("--conda-prefix", default=os.environ.get("PY_VIPE_CONDA_PREFIX", os.environ.get("PY_LYRA_CONDA_PREFIX")))
    parser.add_argument("--visualize", action="store_true", help="Ask VIPE to save visualization videos.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    vipe_dir = args.vipe_dir.expanduser().resolve()
    vipe_cli = vipe_dir / "vipe" / "cli" / "main.py"
    pipeline_config = vipe_dir / "configs" / "pipeline" / f"{args.pipeline}.yaml"

    missing = []
    if not vipe_cli.exists():
        missing.append(f"VIPE CLI not found: {vipe_cli}")
    if not pipeline_config.exists():
        missing.append(f"VIPE pipeline config not found: {pipeline_config}")
    input_path = args.video or args.image_dir
    if input_path is None or not input_path.exists():
        missing.append(f"Input path not found: {input_path}")
    if missing:
        for item in missing:
            print(f"[FAIL] {item}", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [args.python, str(vipe_cli), "infer"]
    if args.image_dir:
        cmd.extend(["--image-dir", str(args.image_dir)])
    else:
        cmd.append(str(args.video))
    cmd.extend(["--output", str(args.output_dir), "--pipeline", args.pipeline])
    if args.visualize:
        cmd.append("--visualize")

    print("[INFO] Running VIPE compatibility background path:")
    print(" ".join(cmd))
    proc = subprocess.run(cmd, env=build_env(args, vipe_dir), cwd=str(vipe_dir))
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
