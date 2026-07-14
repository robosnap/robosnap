from __future__ import annotations

import numpy as np
from PIL import Image

from robosnap.preprocess.auto_segment_image import load_prompt_text
from robosnap.preprocess.sam3_mask_retry import (
    fill_binary_holes,
    mask_matches_bbox,
    reject_mismatched_masks,
)


def test_load_prompt_text_accepts_long_inline_prompt() -> None:
    prompt = "remove foreground and preserve background\n" * 100
    assert load_prompt_text(prompt) == prompt.strip()


def test_fill_binary_holes_preserves_exterior_background() -> None:
    mask = np.zeros((20, 30), dtype=bool)
    mask[4:16, 5:25] = True
    mask[8:12, 10:15] = False
    mask[10:16, 20:24] = False
    filled = fill_binary_holes(mask)
    assert filled[8:12, 10:15].all()
    assert not filled[10:16, 20:24].all()
    assert not filled[0].any()


def test_bbox_gate_rejects_spatially_wrong_mask(tmp_path) -> None:
    rgba = np.zeros((100, 200, 4), dtype=np.uint8)
    rgba[5:15, 5:15, 3] = 255
    mask_path = tmp_path / "0.png"
    Image.fromarray(rgba).save(mask_path)
    bbox = (0.70, 0.60, 0.85, 0.80)
    assert not mask_matches_bbox(mask_path, bbox)
    reject_mismatched_masks([{"id": 0, "bbox_xyxy": list(bbox)}], tmp_path)
    assert not mask_path.exists()
