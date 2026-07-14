#!/usr/bin/env python3
"""Download or materialize RoboSnap checkpoint files into the release layout.

This helper supports:
1. Downloading checkpoints from configured Hugging Face repositories.
2. Preparing the expected local checkpoint layout from existing local weights.

The default checkpoint paths are:

    SAM3_CKPT=${CHECKPOINT_DIR}/sam3/sam3.pt
    SAM3D_CONFIG=${CHECKPOINT_DIR}/sam-3d-objects/pipeline.yaml
    ARTICULATE_CKPT=${CHECKPOINT_DIR}/articulate/articulate.safetensors
    SONATA_CACHE_DIR=${CHECKPOINT_DIR}/sonata
    HF_HOME=${CHECKPOINT_DIR}/hf_cache
    TORCH_HOME=${CHECKPOINT_DIR}/torch_cache

For existing local checkpoints, use:
    scripts/gui/bash/copy_checkpoints_from_local.sh
"""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FileDownload:
    name: str
    repo_id: str | None
    filename: str | None
    destination: Path
    required: bool = True


@dataclass(frozen=True)
class SnapshotDownload:
    name: str
    repo_id: str | None
    destination: Path
    allow_patterns: list[str] | None = None
    required: bool = False
    cache_only: bool = False


def import_hf():
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required for downloads. Install with: pip install huggingface_hub"
        ) from exc
    return hf_hub_download, snapshot_download


def parse_patterns(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def copy_or_link(src: Path, dst: Path, mode: str) -> None:
    print(f"[checkpoint] {mode}: {src} -> {dst}")
    ensure_parent(dst)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)
    else:
        raise ValueError(f"Unsupported materialize mode: {mode}")


def download_file(item: FileDownload, cache_dir: Path, mode: str) -> None:
    if not item.repo_id or not item.filename:
        if item.required:
            raise SystemExit(f"Missing repo/file for required checkpoint: {item.name}")
        print(f"[checkpoint] skip optional file {item.name}: repo or filename not configured")
        return
    print(f"[checkpoint] file {item.name}: {item.repo_id}/{item.filename} -> {item.destination}")
    hf_hub_download, _ = import_hf()
    local = Path(hf_hub_download(repo_id=item.repo_id, filename=item.filename, cache_dir=str(cache_dir)))
    copy_or_link(local, item.destination, mode)


def download_snapshot(item: SnapshotDownload, cache_dir: Path, mode: str) -> None:
    if not item.repo_id:
        if item.required:
            raise SystemExit(f"Missing repo for required checkpoint snapshot: {item.name}")
        print(f"[checkpoint] skip optional snapshot {item.name}: repo not configured")
        return
    if item.cache_only:
        print(f"[checkpoint] snapshot {item.name}: {item.repo_id} -> HF cache {cache_dir}")
    else:
        print(f"[checkpoint] snapshot {item.name}: {item.repo_id} -> {item.destination}")
    if item.allow_patterns:
        print(f"[checkpoint]   allow_patterns={item.allow_patterns}")
    _, snapshot_download = import_hf()
    local = Path(snapshot_download(repo_id=item.repo_id, cache_dir=str(cache_dir), allow_patterns=item.allow_patterns))
    if item.cache_only:
        print(f"[checkpoint] cached snapshot {item.name}: {local}")
        return
    if mode == "symlink":
        if item.destination.exists() or item.destination.is_symlink():
            if item.destination.is_symlink() or item.destination.is_file():
                item.destination.unlink()
            else:
                raise SystemExit(f"Refusing to replace existing directory with symlink: {item.destination}")
        item.destination.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(local, item.destination)
    elif mode == "copy":
        item.destination.mkdir(parents=True, exist_ok=True)
        shutil.copytree(local, item.destination, dirs_exist_ok=True)
    else:
        raise ValueError(f"Unsupported materialize mode: {mode}")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Download RoboSnap checkpoints into checkpoints/.")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path(os.environ.get("CHECKPOINT_DIR", root / "checkpoints")))
    parser.add_argument("--cache-dir", type=Path, default=Path(os.environ.get("HF_HOME", root / "checkpoints" / "hf_cache")))
    parser.add_argument("--materialize-mode", choices=["copy", "symlink"], default=os.environ.get("MATERIALIZE_MODE", "copy"))
    parser.add_argument("--sam3-repo", default=os.environ.get("SAM3_HF_REPO", "facebook/sam3"))
    parser.add_argument("--sam3-file", default=os.environ.get("SAM3_HF_FILE", "sam3.pt"))
    parser.add_argument("--articulate-repo", dest="articulate_repo", default=os.environ.get("ARTICULATE_HF_REPO", os.environ.get("P3SAM_HF_REPO", "tencent/Hunyuan3D-Part")))
    parser.add_argument("--p3sam-repo", dest="articulate_repo", help=argparse.SUPPRESS, default=argparse.SUPPRESS)
    parser.add_argument("--articulate-file", dest="articulate_file", default=os.environ.get("ARTICULATE_HF_FILE", os.environ.get("P3SAM_HF_FILE", "p3sam/p3sam.safetensors")))
    parser.add_argument("--p3sam-file", dest="articulate_file", help=argparse.SUPPRESS, default=argparse.SUPPRESS)
    parser.add_argument("--sonata-repo", default=os.environ.get("SONATA_HF_REPO", "facebook/sonata"))
    parser.add_argument("--sonata-file", default=os.environ.get("SONATA_HF_FILE", "sonata.pth"))

    parser.add_argument("--sam3d-repo", default=os.environ.get("SAM3D_HF_REPO"), help="Optional repo containing the SAM3D checkpoint bundle.")
    parser.add_argument("--sam3d-patterns", default=os.environ.get("SAM3D_HF_PATTERNS", "*.yaml,*.yml,*.ckpt,*.pth"))
    parser.add_argument("--moge-repo", default=os.environ.get("MOGE_HF_REPO", "Ruicheng/moge-vitl"))
    parser.add_argument("--skip-optional", action="store_true", help="Skip optional SAM3D/MoGe snapshots.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    mode = args.materialize_mode

    file_items = [
        FileDownload("SAM3", args.sam3_repo, args.sam3_file, checkpoint_dir / "sam3" / "sam3.pt"),
        FileDownload("Articulate", args.articulate_repo, args.articulate_file, checkpoint_dir / "articulate" / "articulate.safetensors"),
        FileDownload("Sonata", args.sonata_repo, args.sonata_file, checkpoint_dir / "sonata" / "sonata.pth"),
    ]
    snapshot_items = [] if args.skip_optional else [
        SnapshotDownload("SAM3D Objects", args.sam3d_repo, checkpoint_dir / "sam-3d-objects", parse_patterns(args.sam3d_patterns), required=False),
        SnapshotDownload("MoGe", args.moge_repo, checkpoint_dir / "hf_cache", required=False, cache_only=True),
    ]

    print(f"[checkpoint] checkpoint_dir={checkpoint_dir}")
    print(f"[checkpoint] cache_dir={cache_dir}")
    print(f"[checkpoint] materialize_mode={mode}")
    for item in file_items:
        download_file(item, cache_dir, mode)
    for item in snapshot_items:
        download_snapshot(item, cache_dir, mode)

    print("[checkpoint] done")
    print("[checkpoint] update configs/gui.env, then start with: bash scripts/run_gui.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
