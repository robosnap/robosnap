from __future__ import annotations

import base64
import importlib.util
import io
from argparse import Namespace
from pathlib import Path

import numpy as np
from PIL import Image


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_gemini_image_edit.py"
SPEC = importlib.util.spec_from_file_location("run_gemini_image_edit", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
gemini = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gemini)


def png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def decode_payload_image(item: dict) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(item["data"]))).convert("RGB"))


def test_request_contains_source_masked_scene_and_binary_mask(tmp_path) -> None:
    source = np.full((4, 6, 3), (120, 80, 40), dtype=np.uint8)
    mask = np.zeros((4, 6), dtype=np.uint8)
    mask[1:3, 2:5] = 255
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    Image.fromarray(source).save(image_path)
    Image.fromarray(mask).save(mask_path)
    args = Namespace(model="gemini-3.1-flash-image", aspect_ratio=None, image_size=None)

    payload = gemini.build_request(image_path, mask_path, gemini.DEFAULT_PROMPT, args)

    assert len(payload["input"]) == 4
    masked_scene = decode_payload_image(payload["input"][2])
    binary = decode_payload_image(payload["input"][3])
    assert np.all(masked_scene[1:3, 2:5] == 0)
    assert np.array_equal(masked_scene[0, 0], source[0, 0])
    assert np.all(binary[1:3, 2:5] == 255)
    assert "not a strict edit boundary" in payload["input"][0]["text"]


def test_semantic_output_can_change_pixels_outside_mask(tmp_path) -> None:
    source_path = tmp_path / "source.png"
    mask_path = tmp_path / "mask.png"
    semantic_path = tmp_path / "semantic.png"
    strict_path = tmp_path / "strict.png"
    Image.new("RGB", (6, 4), (255, 0, 0)).save(source_path)
    mask = np.zeros((4, 6), dtype=np.uint8)
    mask[:, 2:4] = 255
    Image.fromarray(mask).save(mask_path)
    generated = png_bytes(Image.new("RGB", (6, 4), (0, 0, 255)))

    gemini.write_semantic_output(source_path, generated, semantic_path)
    gemini.composite_masked(source_path, generated, mask_path, strict_path, mask_blur=0)

    semantic = np.asarray(Image.open(semantic_path))
    strict = np.asarray(Image.open(strict_path))
    assert np.all(semantic == (0, 0, 255))
    assert np.all(strict[:, :2] == (255, 0, 0))
    assert np.all(strict[:, 2:4] == (0, 0, 255))
