import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_gemini_vlm.py"
SPEC = importlib.util.spec_from_file_location("run_gemini_vlm", SCRIPT)
assert SPEC and SPEC.loader
gemini_vlm = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gemini_vlm)


def test_extracts_interactions_api_text():
    payload = {
        "steps": [
            {
                "type": "model_output",
                "content": [{"type": "text", "text": '{"objects": []}'}],
            }
        ]
    }
    assert gemini_vlm.extract_output_text(payload) == '{"objects": []}'


def test_normalizes_structured_object_output():
    payload = {
        "objects": [
            {
                "name": "cup",
                "prompt": "blue cup on the right side of the table",
                "fallback_prompt": "blue cup",
                "bbox_xyxy": [0.1, 0.2, 0.3, 0.5],
            }
        ]
    }
    parsed = gemini_vlm.parse_objects(json.dumps(payload))
    assert parsed == payload


@pytest.mark.parametrize(
    "bbox",
    ([0.4, 0.2, 0.3, 0.5], [-0.1, 0.2, 0.3, 0.5], [0.1, 0.2, 0.3]),
)
def test_rejects_invalid_normalized_bbox(bbox):
    payload = {
        "objects": [
            {
                "name": "cup",
                "prompt": "cup",
                "fallback_prompt": "cup",
                "bbox_xyxy": bbox,
            }
        ]
    }
    with pytest.raises(ValueError):
        gemini_vlm.parse_objects(json.dumps(payload))
