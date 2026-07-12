#!/usr/bin/env python3
"""Download automatic-pipeline checkpoints into the RoboSnap runtime layout."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Download automatic-pipeline model checkpoints.")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(os.environ.get("CHECKPOINT_DIR", root / "checkpoints")),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.environ.get("HF_HOME", root / "checkpoints" / "hf_cache")),
    )
    parser.add_argument("--core", action="store_true", help="Download SAM3, gated SAM3D, and cache VGGT.")
    parser.add_argument("--sam3", action="store_true")
    parser.add_argument("--sam3d", action="store_true")
    parser.add_argument("--vggt", action="store_true")
    parser.add_argument("--lyra", action="store_true", help="Download the approximately 91 GB Lyra-2 bundle.")
    parser.add_argument(
        "--accept-lyra-license",
        action="store_true",
        help="Confirm acceptance of the NVIDIA Lyra-2 model license.",
    )
    parser.add_argument("--copy", action="store_true", help="Copy snapshot trees instead of linking them.")
    parser.add_argument("--force", action="store_true", help="Replace an existing model link or directory.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def get_hf_api():
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:
        raise SystemExit("Install huggingface_hub in the calling environment first.") from exc
    return hf_hub_download, snapshot_download


def remove_destination(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def materialize_tree(source: Path, destination: Path, *, copy: bool, force: bool) -> None:
    if destination.exists() or destination.is_symlink():
        if not force:
            print(f"[models] keep existing {destination}")
            return
        remove_destination(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copytree(source, destination)
    else:
        destination.symlink_to(source.resolve(), target_is_directory=True)


def download_file(repo_id: str, filename: str, destination: Path, cache_dir: Path, args: argparse.Namespace) -> None:
    print(f"[models] {repo_id}/{filename} -> {destination}")
    if args.dry_run:
        return
    if destination.is_file() and not args.force:
        print(f"[models] keep existing {destination}")
        return
    hf_hub_download, _ = get_hf_api()
    source = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=str(cache_dir),
            token=os.environ.get("HF_TOKEN"),
        )
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        remove_destination(destination)
    if args.copy:
        shutil.copy2(source, destination)
    else:
        destination.symlink_to(source.resolve())


def download_snapshot(
    repo_id: str,
    cache_dir: Path,
    args: argparse.Namespace,
    *,
    allow_patterns: list[str] | None = None,
) -> Path | None:
    print(f"[models] snapshot {repo_id} -> cache {cache_dir}")
    if allow_patterns:
        print(f"[models] allow_patterns={','.join(allow_patterns)}")
    if args.dry_run:
        return None
    _, snapshot_download = get_hf_api()
    return Path(
        snapshot_download(
            repo_id=repo_id,
            cache_dir=str(cache_dir),
            allow_patterns=allow_patterns,
            token=os.environ.get("HF_TOKEN"),
            max_workers=1,
        )
    )


def main() -> int:
    args = parse_args()
    if args.core:
        args.sam3 = args.sam3d = args.vggt = True
    if not any((args.sam3, args.sam3d, args.vggt, args.lyra)):
        raise SystemExit("Select --core, --sam3, --sam3d, --vggt, or --lyra.")
    if args.lyra and not args.accept_lyra_license:
        raise SystemExit("--lyra requires --accept-lyra-license.")

    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    print(f"[models] checkpoint_dir={checkpoint_dir}")
    print(f"[models] cache_dir={cache_dir}")
    if not args.dry_run:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)

    if args.sam3:
        download_file(
            "facebook/sam3",
            "sam3.pt",
            checkpoint_dir / "sam3" / "sam3.pt",
            cache_dir,
            args,
        )

    if args.sam3d:
        snapshot = download_snapshot(
            "facebook/sam-3d-objects",
            cache_dir,
            args,
            allow_patterns=["checkpoints/**"],
        )
        if snapshot is not None:
            source = snapshot / "checkpoints"
            if not (source / "pipeline.yaml").is_file():
                raise RuntimeError(f"SAM3D snapshot is missing checkpoints/pipeline.yaml: {snapshot}")
            destination = checkpoint_dir / "sam-3d-objects"
            materialize_tree(
                source,
                destination,
                copy=args.copy,
                force=args.force,
            )
            if not (destination / "pipeline.yaml").is_file():
                raise RuntimeError(f"Existing SAM3D destination is incomplete: {destination}")

    if args.vggt:
        download_snapshot("facebook/VGGT-1B", cache_dir, args)

    if args.lyra:
        print("[models] Lyra-2 requires approximately 91 GB before Hugging Face cache overhead.")
        snapshot = download_snapshot(
            "nvidia/Lyra-2.0",
            cache_dir,
            args,
            allow_patterns=["checkpoints/**"],
        )
        if snapshot is not None:
            source = snapshot / "checkpoints"
            if not (source / "recon" / "model.pt").is_file():
                raise RuntimeError(f"Lyra snapshot is missing checkpoints/recon/model.pt: {snapshot}")
            destination = checkpoint_dir / "lyra2" / "checkpoints"
            materialize_tree(
                source,
                destination,
                copy=args.copy,
                force=args.force,
            )
            if not (destination / "recon" / "model.pt").is_file():
                raise RuntimeError(f"Existing Lyra destination is incomplete: {destination}")

    print("[models] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
