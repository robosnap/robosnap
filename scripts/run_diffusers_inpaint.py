#!/usr/bin/env python3
"""Run a diffusers inpainting model on an image/mask pair.

The output keeps all unmasked pixels from the original image. This makes the
script suitable for RoboSnap background completion, where only the segmented
interactive foreground should be replaced.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


DEFAULT_NEGATIVE = (
    "people, person, human, desk, table, monitor, laptop, computer, cable, "
    "charger, bottle, tissue pack, backpack, box, text, logo, watermark, "
    "distorted geometry, warped lines, blurry, low quality, artifacts"
)


def parse_size(value: str) -> tuple[int, int]:
    if "x" not in value:
        raise argparse.ArgumentTypeError("size must be WIDTHxHEIGHT")
    width, height = value.lower().split("x", 1)
    return int(width), int(height)


def load_prompt(path_or_text: str) -> str:
    path = Path(path_or_text)
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return path_or_text.strip()


def composite_original(original: Image.Image, generated: Image.Image, mask: Image.Image) -> Image.Image:
    mask_l = mask.convert("L")
    if mask_l.size != original.size:
        mask_l = mask_l.resize(original.size, Image.Resampling.NEAREST)
    generated = generated.resize(original.size, Image.Resampling.BICUBIC)
    return Image.composite(generated.convert("RGB"), original.convert("RGB"), mask_l)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", required=True, help="Prompt text or path to a prompt file.")
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE)
    parser.add_argument("--model", default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1")
    parser.add_argument("--cache-dir", type=Path, default=Path("/cpfs/shared/aigc/zhangshujie/hf_cache"))
    parser.add_argument("--size", type=parse_size, default=(1024, 576))
    parser.add_argument("--steps", type=int, default=35)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--strength", type=float, default=0.98)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mask-blur", type=float, default=2.0)
    parser.add_argument("--disable-safety-checker", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--status", type=Path)
    args = parser.parse_args()

    compat_path = os.environ.get("ROBOSNAP_DIFFUSERS_PATH")
    if compat_path:
        sys.path.insert(0, compat_path)
    try:
        import diffusers
        import torch
        from diffusers import AutoPipelineForInpainting
    except ImportError as exc:
        raise RuntimeError("Install the optional inpaint dependencies with: pip install -e '.[inpaint]'") from exc

    prompt = load_prompt(args.prompt)
    image = Image.open(args.image).convert("RGB")
    mask = Image.open(args.mask).convert("L")
    if args.mask_blur > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(args.mask_blur))

    target_w, target_h = args.size
    image_small = image.resize((target_w, target_h), Image.Resampling.LANCZOS)
    mask_small = mask.resize((target_w, target_h), Image.Resampling.NEAREST)

    dtype = torch.float16 if args.device.startswith("cuda") and torch.cuda.is_available() else torch.float32
    load_kwargs = {
        "torch_dtype": dtype,
        "cache_dir": str(args.cache_dir),
        "use_safetensors": True,
        "local_files_only": args.local_files_only,
    }
    if args.disable_safety_checker:
        load_kwargs.update(safety_checker=None, requires_safety_checker=False)
    if dtype == torch.float16:
        load_kwargs["variant"] = "fp16"
    try:
        pipe = AutoPipelineForInpainting.from_pretrained(args.model, **load_kwargs)
    except OSError:
        if "variant" not in load_kwargs:
            raise
        load_kwargs.pop("variant")
        pipe = AutoPipelineForInpainting.from_pretrained(args.model, **load_kwargs)
    pipe = pipe.to(args.device)
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()

    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    result = pipe(
        prompt=prompt,
        negative_prompt=args.negative_prompt,
        image=image_small,
        mask_image=mask_small,
        guidance_scale=args.guidance_scale,
        strength=args.strength,
        num_inference_steps=args.steps,
        generator=generator,
    ).images[0]

    output = composite_original(image, result, mask)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.save(args.output)

    status_path = args.status or args.output.with_name(args.output.stem + "_status.json")
    status = {
        "status": "diffusers_inpaint_complete",
        "model": args.model,
        "diffusers_version": diffusers.__version__,
        "cache_dir": str(args.cache_dir),
        "compat_path": compat_path,
        "image": str(args.image),
        "mask": str(args.mask),
        "output": str(args.output),
        "prompt": prompt,
        "negative_prompt": args.negative_prompt,
        "size": [target_w, target_h],
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "strength": args.strength,
        "seed": args.seed,
        "masked_pixels": int((np.asarray(mask) > 0).sum()),
    }
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
