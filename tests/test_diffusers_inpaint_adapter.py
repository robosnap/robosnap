import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_diffusers_inpaint.py"
SPEC = importlib.util.spec_from_file_location("run_diffusers_inpaint", SCRIPT)
assert SPEC and SPEC.loader
diffusers_inpaint = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diffusers_inpaint)


def test_fit_size_preserves_landscape_aspect_ratio():
    assert diffusers_inpaint.fit_size((1280, 720)) == (1024, 576)


def test_fit_size_preserves_portrait_aspect_ratio():
    assert diffusers_inpaint.fit_size((720, 1280)) == (576, 1024)


def test_fit_size_does_not_upscale_small_images():
    assert diffusers_inpaint.fit_size((640, 480)) == (640, 480)


def test_mask_blur_does_not_expand_outside_binary_mask():
    array = np.zeros((9, 9), dtype=np.uint8)
    array[3:6, 3:6] = 255
    prepared = np.asarray(diffusers_inpaint.prepare_mask(Image.fromarray(array), blur=2.0))
    assert np.all(prepared[array == 0] == 0)
    assert np.any(prepared[array > 0] > 0)
