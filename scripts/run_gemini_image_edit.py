#!/usr/bin/env python3
"""Run semantic Gemini image editing for RoboSnap background completion.

The request contains the source image, a black-hole masked scene, and a binary
mask. Gemini returns a complete empty background. Strict mask compositing is
available only as an explicit compatibility option.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter


DEFAULT_MODEL = "gemini-3.1-flash-image"
DEFAULT_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_PROMPT = """You are an excellent image inpainter and currently are here to help me inpaint the masked image where only the background is reserved and the interactive area is removed.
Please remove all the black area from these scenes, which is not included in the background.
Keep the room structure unchanged.
Preserve walls, floor, ceiling, windows and lighting to recover a full background image.
Do not change camera perspective.
Return an empty background with consistent geometry and textures. I stress again that I am asking you to inpaint the background, {!not!} refill the mask area.
Make additional careful checks in order not to delete additional background objects.
If there are any pixels from the interactive area that were not completely removed, please remove them as well, leaving only the background.
Special reminder: remove the desktop area as well since I generate the area separately. I want the background to be empty to hold the desk."""


def load_prompt(value: str | None) -> str:
    if not value:
        return DEFAULT_PROMPT
    path = Path(value)
    if path.exists():
        text = path.read_text(encoding="utf-8").strip()
    else:
        text = value.strip()
    return text or DEFAULT_PROMPT


def pil_to_png_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def image_to_png_b64(path: Path) -> str:
    return pil_to_png_b64(Image.open(path).convert("RGB"))


def binary_mask(mask_path: Path, size: tuple[int, int]) -> Image.Image:
    mask = Image.open(mask_path).convert("L")
    if mask.size != size:
        mask = mask.resize(size, Image.Resampling.NEAREST)
    return mask.point(lambda p: 255 if p > 0 else 0)


def mask_to_png_b64(mask_path: Path, size: tuple[int, int]) -> str:
    return pil_to_png_b64(binary_mask(mask_path, size))


def masked_scene_to_png_b64(image_path: Path, mask_path: Path) -> str:
    source = Image.open(image_path).convert("RGB")
    mask = binary_mask(mask_path, source.size)
    black = Image.new("RGB", source.size, (0, 0, 0))
    return pil_to_png_b64(Image.composite(black, source, mask))


def build_request(image_path: Path, mask_path: Path, prompt: str, args: argparse.Namespace) -> dict[str, Any]:
    image_size = Image.open(image_path).size
    full_prompt = (
        prompt.strip()
        + "\n\nInput image roles:\n"
        + "1. The first image is the original source photograph.\n"
        + "2. The second image is the masked scene; black pixels mark removed or uncertain interactive content.\n"
        + "3. The third image is the binary guidance mask; white marks known foreground.\n"
        + "The mask is guidance, not a strict edit boundary. Remove any residual interactive-area pixels "
        + "that segmentation missed, including the complete desk/desktop when present. "
        + "Preserve the room background, camera intrinsics, perspective, lighting, and aspect ratio."
    )
    response_format: dict[str, Any] = {"type": "image", "mime_type": "image/png"}
    if args.aspect_ratio:
        response_format["aspect_ratio"] = args.aspect_ratio
    if args.image_size:
        response_format["image_size"] = args.image_size
    return {
        "model": args.model,
        "input": [
            {"type": "text", "text": full_prompt},
            {"type": "image", "mime_type": "image/png", "data": image_to_png_b64(image_path)},
            {"type": "image", "mime_type": "image/png", "data": masked_scene_to_png_b64(image_path, mask_path)},
            {"type": "image", "mime_type": "image/png", "data": mask_to_png_b64(mask_path, image_size)},
        ],
        "response_format": response_format,
        "store": False,
    }


def request_gemini(payload: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing Gemini API key. Set GEMINI_API_KEY or GOOGLE_API_KEY.")
    url = f"{args.api_base.rstrip('/')}/interactions"
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
    last_error: Exception | None = None
    for attempt in range(args.retries + 1):
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=args.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"Gemini HTTP {exc.code}: {body}")
        except urllib.error.URLError as exc:
            last_error = RuntimeError(f"Gemini request failed: {exc}")
        if attempt < args.retries:
            time.sleep(args.retry_delay * (attempt + 1))
    assert last_error is not None
    raise last_error


def extract_image_bytes(response: dict[str, Any]) -> tuple[bytes, str]:
    output_image = response.get("output_image") or response.get("outputImage")
    if isinstance(output_image, dict) and output_image.get("data"):
        mime_type = output_image.get("mime_type") or output_image.get("mimeType") or "image/png"
        return base64.b64decode(output_image["data"]), mime_type

    for step in reversed(response.get("steps", [])):
        if not isinstance(step, dict) or step.get("type") != "model_output":
            continue
        for block in reversed(step.get("content", [])):
            if isinstance(block, dict) and block.get("type") == "image" and block.get("data"):
                mime_type = block.get("mime_type") or block.get("mimeType") or "image/png"
                return base64.b64decode(block["data"]), mime_type

    outputs = response.get("output") or response.get("outputs") or []
    for item in outputs if isinstance(outputs, list) else []:
        if isinstance(item, dict) and item.get("type") == "image" and item.get("data"):
            mime_type = item.get("mime_type") or item.get("mimeType") or "image/png"
            return base64.b64decode(item["data"]), mime_type

    for candidate in response.get("candidates", []):
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            inline = part.get("inlineData") or part.get("inline_data")
            if inline and inline.get("data"):
                mime_type = inline.get("mimeType") or inline.get("mime_type") or "image/png"
                return base64.b64decode(inline["data"]), mime_type
    raise RuntimeError("Gemini response did not contain an image payload.")


def generated_image(generated_bytes: bytes, size: tuple[int, int]) -> Image.Image:
    image = Image.open(io.BytesIO(generated_bytes)).convert("RGB")
    if image.size != size:
        image = image.resize(size, Image.Resampling.BICUBIC)
    return image


def write_semantic_output(original_path: Path, generated_bytes: bytes, output_path: Path) -> None:
    size = Image.open(original_path).size
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generated_image(generated_bytes, size).save(output_path)


def composite_masked(original_path: Path, generated_bytes: bytes, mask_path: Path, output_path: Path, mask_blur: float) -> None:
    original = Image.open(original_path).convert("RGB")
    generated = generated_image(generated_bytes, original.size)
    mask = Image.open(mask_path).convert("L")
    if mask.size != original.size:
        mask = mask.resize(original.size, Image.Resampling.NEAREST)
    mask = mask.point(lambda p: 255 if p > 0 else 0)
    if mask_blur > 0:
        binary = np.asarray(mask) > 0
        alpha = np.asarray(mask.filter(ImageFilter.GaussianBlur(mask_blur))).copy()
        alpha[~binary] = 0
        mask = Image.fromarray(alpha)
    output = Image.composite(generated, original, mask)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(output_path)


def write_status(path: Path, status: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", help="Prompt text or path to a prompt file.")
    parser.add_argument("--model", default=os.environ.get("GEMINI_IMAGE_MODEL", DEFAULT_MODEL))
    parser.add_argument("--api-base", default=os.environ.get("GEMINI_API_BASE", DEFAULT_API_BASE))
    parser.add_argument("--api-key", default=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    parser.add_argument("--aspect-ratio", help="Optional Gemini response_format aspect_ratio, e.g. 16:9.")
    parser.add_argument("--image-size", help="Optional Gemini response_format image_size, e.g. 1K or 2K.")
    parser.add_argument("--mask-blur", type=float, default=1.5)
    parser.add_argument(
        "--strict-mask-composite",
        action="store_true",
        help="Copy black-mask pixels from the source instead of accepting Gemini's semantic full-frame edit.",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-delay", type=float, default=3.0)
    parser.add_argument("--raw-output", type=Path, help="Optional path for the uncomposited generated image.")
    parser.add_argument("--status", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prompt = load_prompt(args.prompt)
    payload = build_request(args.image, args.mask, prompt, args)
    response = request_gemini(payload, args)
    image_bytes, mime_type = extract_image_bytes(response)

    if args.raw_output:
        suffix = mimetypes.guess_extension(mime_type) or ".png"
        raw_path = args.raw_output
        if raw_path.suffix == "":
            raw_path = raw_path.with_suffix(suffix)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(image_bytes)
    else:
        raw_path = None

    if args.strict_mask_composite:
        composite_masked(args.image, image_bytes, args.mask, args.output, args.mask_blur)
        output_mode = "strict_mask_composite"
    else:
        write_semantic_output(args.image, image_bytes, args.output)
        output_mode = "semantic_full_frame"
    status_path = args.status or args.output.with_name(args.output.stem + "_status.json")
    mask = Image.open(args.mask).convert("L")
    status = {
        "status": "gemini_image_edit_complete",
        "model": args.model,
        "api_base": args.api_base,
        "image": str(args.image),
        "mask": str(args.mask),
        "output": str(args.output),
        "raw_output": str(raw_path) if raw_path else None,
        "prompt": prompt,
        "aspect_ratio": args.aspect_ratio,
        "image_size": args.image_size,
        "output_mode": output_mode,
        "strict_mask_composite": bool(args.strict_mask_composite),
        "mask_blur": args.mask_blur if args.strict_mask_composite else None,
        "masked_pixels": int((np.asarray(mask) > 0).sum()),
        "response_mime_type": mime_type,
    }
    write_status(status_path, status)
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[gemini-image-edit] ERROR: {exc}", file=sys.stderr)
        raise
