from __future__ import annotations

import numpy as np
from PIL import Image

from robosnap.preprocess.sam3_mask_retry import mask_matches_bbox


def test_bbox_gate_rejects_mask_that_only_grazes_target_box(tmp_path) -> None:
    rgba = np.zeros((100, 200, 4), dtype=np.uint8)
    rgba[35:65, 50:90, 3] = 255
    mask_path = tmp_path / "mask.png"
    Image.fromarray(rgba).save(mask_path)

    assert not mask_matches_bbox(mask_path, (0.42, 0.40, 0.55, 0.60))


def test_bbox_gate_accepts_centered_mask(tmp_path) -> None:
    rgba = np.zeros((100, 200, 4), dtype=np.uint8)
    rgba[35:65, 80:120, 3] = 255
    mask_path = tmp_path / "mask.png"
    Image.fromarray(rgba).save(mask_path)

    assert mask_matches_bbox(mask_path, (0.35, 0.25, 0.65, 0.75))
