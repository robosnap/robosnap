#!/usr/bin/env python3
"""Prepare a single-image automatic segmentation workspace.

This module owns the lightweight, releasable API surface around VLM object
listing, SAM3 text-prompt segmentation, foreground-mask union, and optional
background inpainting. Heavy models are launched through subprocesses so users
can keep their own Python environments.
"""

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
from typing import Any

import cv2
import numpy as np
from PIL import Image

from robosnap.preprocess.sam3_mask_retry import (
    fill_binary_holes,
    filter_small_components,
    recover_sam3_masks,
)


DEFAULT_VLM_PROMPT = """Inspect the image and return a JSON object with an objects list.
List every separate foreground asset needed to reconstruct the interactive scene, including the complete support furniture, every item on or against it, partially occluded or image-border-truncated objects, and near dividers or occluders that intersect the workspace.
Do not list walls, floor, ceiling, windows, distant people, or distant room furniture.
Use one entry per physical instance with name and a location-aware prompt suitable for text-prompted segmentation.
For each entry also return a concise fallback_prompt containing only the object category and relative position.
Also return bbox_xyxy as normalized [x_min, y_min, x_max, y_max] coordinates for each object.
Return JSON only."""

DEFAULT_INPAINT_PROMPT = """You are an excellent image inpainter and currently are here to help me inpaint the masked image where only the background is reserved and the interactive area is removed.
Please remove all the black area from these scenes, which is not included in the background.
Keep the room structure unchanged.
Preserve walls, floor, ceiling, windows and lighting to recover a full background image.
Do not change camera perspective.
Return an empty background with consistent geometry and textures. I stress again that I am asking you to inpaint the background, {!not!} refill the mask area.
Make additional careful checks in order not to delete additional background objects.
If there are any pixels from the interactive area that were not completely removed, please remove them as well, leaving only the background.
Special reminder: remove the desktop area as well since I generate the area separately. I want the background to be empty to hold the desk."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_prompt_text(value: str) -> str:
    if "\n" not in value:
        try:
            path = Path(value).expanduser()
            if path.is_file():
                return path.read_text(encoding="utf-8").strip()
        except OSError:
            pass
    return value.strip()


def parse_comma_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def normalize_object_item(item: Any, index: int) -> dict[str, Any]:
    if isinstance(item, str):
        prompt = item.strip()
        return {"id": index, "name": prompt, "prompt": prompt}
    if isinstance(item, dict):
        name = str(item.get("name") or item.get("label") or item.get("prompt") or f"object_{index}").strip()
        prompt = str(item.get("prompt") or item.get("segmentation_prompt") or item.get("description") or name).strip()
        out = dict(item)
        out.update({"id": index, "name": name, "prompt": prompt})
        return out
    raise ValueError(f"Unsupported object item at index {index}: {type(item)!r}")


def normalize_objects(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict):
        if "objects" in data:
            data = data["objects"]
        elif "prompts" in data:
            data = data["prompts"]
        else:
            raise ValueError("VLM JSON must contain an 'objects' or 'prompts' field.")
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of objects/prompts, got {type(data)!r}")
    objects = [normalize_object_item(item, i) for i, item in enumerate(data)]
    return [obj for obj in objects if obj["prompt"]]


def expand_command_template(template: str, **values: str) -> list[str]:
    return [token.format(**values) for token in shlex.split(template)]


def load_objects(args: argparse.Namespace, output_dir: Path) -> list[dict[str, Any]]:
    if args.objects:
        return normalize_objects(parse_comma_list(args.objects))

    if args.object_file:
        path = Path(args.object_file)
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix.lower() == ".json":
            return normalize_objects(json.loads(path.read_text(encoding="utf-8")))
        return normalize_objects([line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])

    if args.vlm_command:
        output_json = output_dir / "vlm_objects.json"
        output_txt = output_dir / "object.txt"
        prompt_path = output_dir / "vlm_prompt.txt"
        prompt_path.write_text(load_prompt_text(args.vlm_prompt), encoding="utf-8")
        command = expand_command_template(
            args.vlm_command,
            image=str(args.image),
            prompt=str(prompt_path),
            output_json=str(output_json),
            output_txt=str(output_txt),
        )
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                "VLM command failed with code "
                f"{result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
        if output_json.exists():
            return normalize_objects(json.loads(output_json.read_text(encoding="utf-8")))
        stdout = result.stdout.strip()
        if stdout:
            try:
                return normalize_objects(json.loads(stdout))
            except json.JSONDecodeError:
                return normalize_objects([line.strip() for line in stdout.splitlines() if line.strip()])
        if output_txt.exists():
            return normalize_objects([line.strip() for line in output_txt.read_text(encoding="utf-8").splitlines() if line.strip()])
        raise RuntimeError("VLM command succeeded but produced no object list.")

    raise RuntimeError("No object source configured. Pass --objects, --object-file, or --vlm-command.")


def write_object_files(objects: list[dict[str, Any]], output_dir: Path) -> None:
    lines = [obj["prompt"] for obj in objects]
    (output_dir / "object.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    (output_dir / "object_metadata.json").write_text(json.dumps({"objects": objects}, indent=2), encoding="utf-8")


def compact_object_masks(mask_dir: Path, objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only prompts whose numeric SAM3 mask exists, then renumber 0..N-1."""
    kept: list[dict[str, Any]] = []
    tmp_moves: list[tuple[Path, Path]] = []
    for obj in objects:
        new_id = len(kept)
        old_id = int(obj["id"])
        src = mask_dir / f"{old_id}.png"
        if not src.exists():
            continue
        tmp = mask_dir / f".compact_{new_id}.png"
        tmp_moves.append((tmp, mask_dir / f"{new_id}.png"))
        if src != tmp:
            shutil.move(str(src), str(tmp))
        updated = dict(obj)
        updated["id"] = new_id
        updated["source_id"] = old_id
        kept.append(updated)
    for tmp, dst in tmp_moves:
        shutil.move(str(tmp), str(dst))
    return kept


def build_env_with_pythonpath(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{root}{os.pathsep}{existing}" if existing else str(root)
    return env


def run_sam3(
    *,
    image: Path,
    prompts: list[str],
    out_dir: Path,
    python: str,
    sam3_dir: Path,
    checkpoint: Path,
) -> None:
    if not prompts:
        return
    script = sam3_dir / "inference_image.py"
    missing = [p for p in (script, checkpoint, image) if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing SAM3 input(s): " + ", ".join(str(p) for p in missing))
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        python,
        str(script),
        "--image",
        str(image),
        "--prompt_mode",
        "0",
        "--text_prompts",
        # SAM3 uses commas as its prompt-list delimiter; keep one mask per VLM object.
        ",".join(prompt.replace(",", ";") for prompt in prompts),
        "--checkpoint",
        str(checkpoint),
        "--out_dir",
        str(out_dir),
    ]
    print("[auto-segment] " + " ".join(shlex.quote(part) for part in cmd))
    proc = subprocess.run(cmd, cwd=str(sam3_dir), env=build_env_with_pythonpath(sam3_dir))
    if proc.returncode != 0:
        raise RuntimeError(f"SAM3 failed with code {proc.returncode}")


def load_alpha_mask(mask_path: Path, size_hw: tuple[int, int]) -> np.ndarray:
    image = Image.open(mask_path).convert("RGBA")
    if image.size != (size_hw[1], size_hw[0]):
        image = image.resize((size_hw[1], size_hw[0]), Image.Resampling.NEAREST)
    arr = np.asarray(image)
    alpha = arr[:, :, 3]
    if alpha.max() > alpha.min():
        return alpha > 0
    return np.any(arr[:, :, :3] > 0, axis=2)


def union_numeric_masks(mask_dir: Path, count: int, size_hw: tuple[int, int]) -> np.ndarray:
    union = np.zeros(size_hw, dtype=bool)
    for idx in range(count):
        path = mask_dir / f"{idx}.png"
        if path.exists():
            union |= filter_small_components(load_alpha_mask(path, size_hw))
    return union


def save_mask_products(
    *,
    image_path: Path,
    sam3d_dir: Path,
    support_dir: Path,
    object_count: int,
    support_count: int,
    output_dir: Path,
    inpaint_dilation: int,
    inpaint_extra_mask: Path | None,
) -> tuple[dict[str, str], dict[str, Any]]:
    image = Image.open(image_path).convert("RGB")
    rgb = np.asarray(image)
    size_hw = rgb.shape[:2]
    object_union = union_numeric_masks(sam3d_dir, object_count, size_hw)
    support_union = union_numeric_masks(support_dir, support_count, size_hw)
    raw_foreground = object_union | support_union
    foreground = fill_binary_holes(raw_foreground)
    foreground_hole_pixels = int(np.logical_and(foreground, ~raw_foreground).sum())

    def save_rgba(path: Path, alpha_mask: np.ndarray) -> None:
        alpha = (alpha_mask.astype(np.uint8) * 255)
        Image.fromarray(np.dstack([rgb, alpha])).save(path)

    foreground_path = output_dir / "foreground_mask.png"
    support_path = output_dir / "support_mask.png"
    background_rgba_path = output_dir / "background_rgba.png"
    object_dilated_mask_path = output_dir / "inpaint_mask_object_dilated.png"
    inpaint_mask_path = output_dir / "inpaint_mask.png"
    save_rgba(foreground_path, foreground)
    save_rgba(support_path, support_union)
    save_rgba(background_rgba_path, ~foreground)
    inpaint = foreground
    if inpaint_dilation > 0:
        kernel_size = 2 * inpaint_dilation + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        inpaint = cv2.dilate(foreground.astype(np.uint8), kernel, iterations=1) > 0
    object_dilated_pixels = int(inpaint.sum())
    Image.fromarray(inpaint.astype(np.uint8) * 255).save(object_dilated_mask_path)

    extra_mask_sha256 = None
    if inpaint_extra_mask is not None:
        if not inpaint_extra_mask.exists():
            raise FileNotFoundError(inpaint_extra_mask)
        inpaint |= load_alpha_mask(inpaint_extra_mask, size_hw)
        extra_mask_sha256 = sha256_file(inpaint_extra_mask)

    Image.fromarray(inpaint.astype(np.uint8) * 255).save(inpaint_mask_path)
    paths = {
        "foreground_mask": str(foreground_path),
        "support_mask": str(support_path),
        "background_rgba": str(background_rgba_path),
        "object_dilated_mask": str(object_dilated_mask_path),
        "inpaint_mask": str(inpaint_mask_path),
    }
    stats = {
        "foreground_pixels": int(foreground.sum()),
        "support_pixels": int(support_union.sum()),
        "object_dilated_pixels": object_dilated_pixels,
        "inpaint_pixels": int(inpaint.sum()),
        "foreground_hole_fill_pixels": foreground_hole_pixels,
        "inpaint_region_policy": "instance-union+enclosed-hole-fill",
        "extra_inpaint_mask": str(inpaint_extra_mask) if inpaint_extra_mask else None,
        "extra_inpaint_mask_sha256": extra_mask_sha256,
    }
    return paths, stats


def run_inpaint(args: argparse.Namespace, output_dir: Path, paths: dict[str, str]) -> None:
    prompt_path = output_dir / "inpaint_prompt.txt"
    prompt_path.write_text(load_prompt_text(args.inpaint_prompt), encoding="utf-8")
    output_path = output_dir / "complete_background.png"

    if args.inpaint_command:
        provider_status_path = output_dir / "inpaint_provider_status.json"
        command = expand_command_template(
            args.inpaint_command,
            image=str(args.image),
            mask=paths["inpaint_mask"],
            output=str(output_path),
            prompt=str(prompt_path),
            status=str(provider_status_path),
        )
        print("[auto-segment] " + shlex.join(command))
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                "Inpaint command failed with code "
                f"{result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
        if not output_path.exists():
            raise RuntimeError(f"Inpaint command did not create {output_path}")
        output_image = Image.open(output_path)
        source_image = Image.open(args.image)
        if output_image.size != source_image.size:
            raise RuntimeError(
                f"Inpaint output size {output_image.size} does not match source image size {source_image.size}: {output_path}"
            )
        provider_status = {}
        if provider_status_path.exists():
            try:
                provider_status = json.loads(provider_status_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                provider_status = {"status": "unreadable", "path": str(provider_status_path)}
        status = {
            "status": "external_ok",
            "image": str(args.image),
            "mask": paths["inpaint_mask"],
            "output": str(output_path),
            "prompt_file": str(prompt_path),
            "provider_status_file": str(provider_status_path),
            "provider_status": provider_status,
            "output_mode": provider_status.get("output_mode", "provider_defined"),
            "sha256": {
                "image": sha256_file(args.image),
                "mask": sha256_file(Path(paths["inpaint_mask"])),
                "output": sha256_file(output_path),
            },
        }
        (output_dir / "complete_background_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        return

    raise RuntimeError(
        "No --inpaint-command was provided. Configure an image-edit API/provider; "
        "the pipeline will not treat an unchanged source image as a completed background."
    )


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Create image.png, object.txt, SAM3 masks, and background inpaint inputs.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--objects", help="Comma-separated object prompts.")
    parser.add_argument("--object-file", type=Path, help="Text or JSON object list.")
    parser.add_argument("--vlm-command", help="Command template. Placeholders: {image}, {prompt}, {output_json}, {output_txt}.")
    parser.add_argument("--vlm-prompt", default=DEFAULT_VLM_PROMPT, help="Prompt text or path passed to the VLM command.")
    parser.add_argument("--support-prompts", default="table, desk, tabletop, tabletop surface")
    parser.add_argument("--sam3-python", default=os.environ.get("PY_SAM3", sys.executable))
    parser.add_argument("--sam3-dir", type=Path, default=Path(os.environ.get("SAM3_DIR", root / "third_party" / "sam3")))
    parser.add_argument("--sam3-checkpoint", type=Path, default=Path(os.environ.get("SAM3_CKPT", root / "checkpoints" / "sam3" / "sam3.pt")))
    parser.add_argument("--inpaint-command", help="Command template. Placeholders: {image}, {mask}, {output}, {prompt}, {status}.")
    parser.add_argument("--inpaint-prompt", default=DEFAULT_INPAINT_PROMPT)
    parser.add_argument("--inpaint-dilation", type=int, default=7)
    parser.add_argument("--inpaint-extra-mask", type=Path, help="Optional binary semantic mask merged into the edit region.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.image = args.image.expanduser().resolve()
    if args.inpaint_extra_mask is not None:
        args.inpaint_extra_mask = args.inpaint_extra_mask.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    sam3d_dir = output_dir / "sam3d"
    support_dir = output_dir / "support_masks"
    output_dir.mkdir(parents=True, exist_ok=True)
    sam3d_dir.mkdir(parents=True, exist_ok=True)

    if not args.image.exists():
        raise FileNotFoundError(args.image)
    shutil.copy2(args.image, output_dir / "image.png")

    objects = load_objects(args, output_dir)
    write_object_files(objects, output_dir)
    prompts = [obj["prompt"] for obj in objects]
    support_prompts = parse_comma_list(args.support_prompts)

    run_sam3(
        image=args.image,
        prompts=prompts,
        out_dir=sam3d_dir,
        python=args.sam3_python,
        sam3_dir=args.sam3_dir.expanduser().resolve(),
        checkpoint=args.sam3_checkpoint.expanduser().resolve(),
    )
    recover_sam3_masks(
        image=args.image,
        objects=objects,
        out_dir=sam3d_dir,
        python=args.sam3_python,
        sam3_dir=args.sam3_dir.expanduser().resolve(),
        checkpoint=args.sam3_checkpoint.expanduser().resolve(),
        run_text=run_sam3,
    )
    objects = compact_object_masks(sam3d_dir, objects)
    if not objects:
        raise RuntimeError("SAM3 did not produce any object masks.")
    write_object_files(objects, output_dir)
    prompts = [obj["prompt"] for obj in objects]
    run_sam3(
        image=args.image,
        prompts=support_prompts,
        out_dir=support_dir,
        python=args.sam3_python,
        sam3_dir=args.sam3_dir.expanduser().resolve(),
        checkpoint=args.sam3_checkpoint.expanduser().resolve(),
    )

    paths, mask_stats = save_mask_products(
        image_path=args.image,
        sam3d_dir=sam3d_dir,
        support_dir=support_dir,
        object_count=len(prompts),
        support_count=len(support_prompts),
        output_dir=output_dir,
        inpaint_dilation=args.inpaint_dilation,
        inpaint_extra_mask=args.inpaint_extra_mask,
    )
    mask_status = {
        "status": "ready",
        "objects": prompts,
        "object_count": len(prompts),
        "inpaint_dilation": args.inpaint_dilation,
        **mask_stats,
    }
    (output_dir / "mask_status.json").write_text(json.dumps(mask_status, indent=2), encoding="utf-8")
    run_inpaint(args, output_dir, paths)

    print(f"[auto-segment] wrote workspace: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
