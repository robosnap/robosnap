from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image


def filter_small_components(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    if count <= 2:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    min_area = max(8, int(np.ceil(float(areas.max()) * 0.001)))
    keep = np.flatnonzero(areas >= min_area) + 1
    return np.isin(labels, keep)


def normalized_bbox_xyxy(obj: dict[str, Any]) -> tuple[float, float, float, float] | None:
    value = obj.get("bbox_xyxy")
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    bbox = np.asarray(value, dtype=np.float64)
    if not np.isfinite(bbox).all() or np.any(bbox < 0.0) or np.any(bbox > 1.0):
        return None
    x1, y1, x2, y2 = (float(item) for item in bbox)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def mask_matches_bbox(mask_path: Path, bbox: tuple[float, float, float, float]) -> bool:
    rgba = np.asarray(Image.open(mask_path).convert("RGBA"))
    mask = rgba[:, :, 3] > 0
    if not mask.any():
        return False
    height, width = mask.shape
    x1, y1, x2, y2 = bbox
    px1 = max(0, min(width - 1, int(np.floor(x1 * width))))
    py1 = max(0, min(height - 1, int(np.floor(y1 * height))))
    px2 = max(px1 + 1, min(width, int(np.ceil(x2 * width))))
    py2 = max(py1 + 1, min(height, int(np.ceil(y2 * height))))
    inside = int(mask[py1:py2, px1:px2].sum())
    ys, xs = np.nonzero(mask)
    center_inside = px1 <= float(xs.mean()) < px2 and py1 <= float(ys.mean()) < py2
    return center_inside and inside / int(mask.sum()) >= 0.2


def reject_mismatched_masks(objects: list[dict[str, Any]], out_dir: Path) -> None:
    for obj in objects:
        bbox = normalized_bbox_xyxy(obj)
        mask_path = out_dir / f"{int(obj['id'])}.png"
        if bbox is not None and mask_path.exists() and not mask_matches_bbox(mask_path, bbox):
            mask_path.unlink()


def _pythonpath_env(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{root}{os.pathsep}{existing}" if existing else str(root)
    return env


def run_geometric_retry(
    *,
    image: Path,
    objects: list[dict[str, Any]],
    out_dir: Path,
    python: str,
    sam3_dir: Path,
    checkpoint: Path,
) -> None:
    boxes = []
    for obj in objects:
        bbox = normalized_bbox_xyxy(obj)
        if bbox is None:
            raise ValueError(f"Missing bbox_xyxy for object {obj.get('id')}")
        x1, y1, x2, y2 = bbox
        boxes.append([(x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1])
    cmd = [
        python,
        str(sam3_dir / "inference_image.py"),
        "--image",
        str(image),
        "--prompt_mode",
        "1",
        "--text_prompts",
        "visual",
        "--geo_prompts",
        json.dumps(boxes),
        "--checkpoint",
        str(checkpoint),
        "--out_dir",
        str(out_dir),
    ]
    print("[auto-segment] " + " ".join(shlex.quote(part) for part in cmd))
    proc = subprocess.run(cmd, cwd=str(sam3_dir), env=_pythonpath_env(sam3_dir))
    if proc.returncode != 0:
        raise RuntimeError(f"SAM3 geometric retry failed with code {proc.returncode}")


def recover_sam3_masks(
    *,
    image: Path,
    objects: list[dict[str, Any]],
    out_dir: Path,
    python: str,
    sam3_dir: Path,
    checkpoint: Path,
    dry_run: bool,
    run_text: Callable[..., None],
) -> None:
    if dry_run:
        return
    reject_mismatched_masks(objects, out_dir)
    missing = [obj for obj in objects if not (out_dir / f"{int(obj['id'])}.png").exists()]
    if missing:
        retry_dir = out_dir / ".short_prompt_retry"
        shutil.rmtree(retry_dir, ignore_errors=True)
        run_text(
            image=image,
            prompts=[str(obj.get("fallback_prompt") or obj["name"]) for obj in missing],
            out_dir=retry_dir,
            python=python,
            sam3_dir=sam3_dir,
            checkpoint=checkpoint,
            dry_run=False,
        )
        for retry_id, obj in enumerate(missing):
            source = retry_dir / f"{retry_id}.png"
            if source.exists():
                shutil.move(str(source), str(out_dir / f"{int(obj['id'])}.png"))
        shutil.rmtree(retry_dir, ignore_errors=True)

    reject_mismatched_masks(objects, out_dir)
    geometric = [
        obj
        for obj in objects
        if not (out_dir / f"{int(obj['id'])}.png").exists()
        and normalized_bbox_xyxy(obj) is not None
    ]
    if not geometric:
        return
    retry_dir = out_dir / ".geometric_retry"
    shutil.rmtree(retry_dir, ignore_errors=True)
    run_geometric_retry(
        image=image,
        objects=geometric,
        out_dir=retry_dir,
        python=python,
        sam3_dir=sam3_dir,
        checkpoint=checkpoint,
    )
    for retry_id, obj in enumerate(geometric):
        source = retry_dir / f"{retry_id}.png"
        if source.exists():
            shutil.move(str(source), str(out_dir / f"{int(obj['id'])}.png"))
    shutil.rmtree(retry_dir, ignore_errors=True)
    reject_mismatched_masks(objects, out_dir)


def fill_binary_holes(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(np.uint8), 1, mode="constant", constant_values=0)
    inverse = (1 - padded) * 255
    flood = inverse.copy()
    flood_mask = np.zeros((flood.shape[0] + 2, flood.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 128)
    holes = flood == 255
    filled = padded.astype(bool) | holes
    return filled[1:-1, 1:-1]
