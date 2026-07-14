"""Canonical paths for automatic-pipeline scene outputs."""

from pathlib import Path


def reconstruction_dir(scene_dir: Path) -> Path:
    return scene_dir / "reconstruction"


def mesh_dir(scene_dir: Path) -> Path:
    return reconstruction_dir(scene_dir) / "meshes"


def vggt_dir(scene_dir: Path) -> Path:
    return reconstruction_dir(scene_dir) / "vggt"
