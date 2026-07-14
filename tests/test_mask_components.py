from __future__ import annotations

import numpy as np

from robosnap.preprocess.sam3_mask_retry import filter_small_components


def test_filter_small_components_removes_relative_noise() -> None:
    mask = np.zeros((200, 300), dtype=bool)
    mask[20:180, 30:270] = True
    mask[2:5, 2:5] = True
    mask[185:195, 280:295] = True
    filtered = filter_small_components(mask)
    assert filtered[20:180, 30:270].all()
    assert not filtered[2:5, 2:5].any()
    assert filtered[185:195, 280:295].all()
