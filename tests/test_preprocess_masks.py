from __future__ import annotations

import json
import sys
from argparse import Namespace

import numpy as np
import pytest
from PIL import Image

from robosnap.pipeline.auto_layered_scene import preprocess_cache_valid
from robosnap.preprocess.auto_segment_image import load_objects, run_inpaint, save_mask_products, sha256_file


def write_rgba_mask(path, mask: np.ndarray) -> None:
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[:, :, :3] = 127
    rgba[:, :, 3] = mask.astype(np.uint8) * 255
    Image.fromarray(rgba).save(path)


def test_vlm_command_receives_prompt_file(tmp_path) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (4, 4), "white").save(image_path)
    provider = tmp_path / "provider.py"
    provider.write_text(
        "import argparse, json\n"
        "from pathlib import Path\n"
        "p=argparse.ArgumentParser()\n"
        "p.add_argument('--prompt')\n"
        "p.add_argument('--output')\n"
        "a=p.parse_args()\n"
        "assert 'support furniture' in Path(a.prompt).read_text()\n"
        "Path(a.output).write_text(json.dumps({'objects':[{'name':'desk','prompt':'complete desk'}]}))\n",
        encoding="utf-8",
    )
    args = Namespace(
        objects=None,
        object_file=None,
        image=image_path,
        vlm_prompt="include the complete support furniture",
        vlm_command=f"{sys.executable} {provider} --prompt {{prompt}} --output {{output_json}}",
    )

    objects = load_objects(args, tmp_path)
    assert objects == [{"id": 0, "name": "desk", "prompt": "complete desk"}]
    assert (tmp_path / "vlm_prompt.txt").read_text() == args.vlm_prompt


def test_mask_builder_does_not_expand_to_image_region(tmp_path) -> None:
    height, width = 100, 200
    image_path = tmp_path / "image.png"
    Image.fromarray(np.full((height, width, 3), 127, dtype=np.uint8)).save(image_path)

    sam3d_dir = tmp_path / "sam3d"
    support_dir = tmp_path / "support_masks"
    sam3d_dir.mkdir()
    support_dir.mkdir()

    object_mask = np.zeros((height, width), dtype=bool)
    object_mask[20:30, 80:120] = True
    support_mask = np.zeros((height, width), dtype=bool)
    support_mask[60:90, 20:180] = True
    write_rgba_mask(sam3d_dir / "0.png", object_mask)
    write_rgba_mask(support_dir / "0.png", support_mask)

    _, stats = save_mask_products(
        image_path=image_path,
        sam3d_dir=sam3d_dir,
        support_dir=support_dir,
        object_count=1,
        support_count=1,
        output_dir=tmp_path,
        inpaint_dilation=0,
        inpaint_extra_mask=None,
    )

    final_mask = np.asarray(Image.open(tmp_path / "inpaint_mask.png")) > 0
    expected = object_mask | support_mask
    assert np.array_equal(final_mask, expected)
    assert not final_mask.any(axis=1).all()
    assert stats["inpaint_region_policy"] == "instance-union"
    assert "semantic_lower_region" not in stats


def test_extra_semantic_mask_is_merged(tmp_path) -> None:
    height, width = 40, 60
    image_path = tmp_path / "image.png"
    Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8)).save(image_path)
    sam3d_dir = tmp_path / "sam3d"
    support_dir = tmp_path / "support_masks"
    sam3d_dir.mkdir()
    support_dir.mkdir()

    object_mask = np.zeros((height, width), dtype=bool)
    object_mask[5:10, 5:10] = True
    extra_mask = np.zeros((height, width), dtype=bool)
    extra_mask[25:35, 30:50] = True
    write_rgba_mask(sam3d_dir / "0.png", object_mask)
    extra_path = tmp_path / "extra.png"
    Image.fromarray(extra_mask.astype(np.uint8) * 255).save(extra_path)

    save_mask_products(
        image_path=image_path,
        sam3d_dir=sam3d_dir,
        support_dir=support_dir,
        object_count=1,
        support_count=0,
        output_dir=tmp_path,
        inpaint_dilation=0,
        inpaint_extra_mask=extra_path,
    )
    final_mask = np.asarray(Image.open(tmp_path / "inpaint_mask.png")) > 0
    assert final_mask[5:10, 5:10].all()
    assert final_mask[25:35, 30:50].all()



def test_inpaint_provider_status_is_preserved(tmp_path) -> None:
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "inpaint_mask.png"
    Image.new("RGB", (8, 8), "white").save(image_path)
    Image.new("L", (8, 8), 255).save(mask_path)
    provider = tmp_path / "provider.py"
    provider.write_text(
        "import argparse, json, shutil\n"
        "from pathlib import Path\n"
        "p=argparse.ArgumentParser()\n"
        "p.add_argument('--image'); p.add_argument('--output'); p.add_argument('--status')\n"
        "a, _ = p.parse_known_args()\n"
        "shutil.copy2(a.image, a.output)\n"
        "Path(a.status).write_text(json.dumps({'status':'ok','output_mode':'semantic_full_frame'}))\n",
        encoding="utf-8",
    )
    args = Namespace(
        image=image_path,
        inpaint_prompt="remove the interactive foreground",
        inpaint_command=(
            f"{sys.executable} {provider} --image {{image}} --mask {{mask}} "
            "--prompt {prompt} --output {output} --status {status}"
        ),
        dry_run=False,
    )

    run_inpaint(args, tmp_path, {"inpaint_mask": str(mask_path)})

    status = json.loads((tmp_path / "complete_background_status.json").read_text())
    assert status["status"] == "external_ok"
    assert status["output_mode"] == "semantic_full_frame"
    assert status["provider_status"]["status"] == "ok"

def test_inpainting_is_fail_closed_without_provider(tmp_path) -> None:
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "inpaint_mask.png"
    Image.new("RGB", (8, 8), "white").save(image_path)
    Image.new("L", (8, 8), 255).save(mask_path)
    args = Namespace(
        image=image_path,
        inpaint_prompt="remove the masked foreground",
        inpaint_command=None,
        dry_run=False,
    )

    with pytest.raises(RuntimeError, match="No --inpaint-command"):
        run_inpaint(args, tmp_path, {"inpaint_mask": str(mask_path)})
    assert not (tmp_path / "complete_background.png").exists()


def test_preprocess_cache_rejects_unedited_background(tmp_path) -> None:
    (tmp_path / "sam3d").mkdir()
    (tmp_path / "object.txt").write_text("object\n", encoding="utf-8")
    Image.new("RGBA", (4, 4), (255, 255, 255, 255)).save(tmp_path / "sam3d" / "0.png")
    Image.new("RGB", (4, 4), "white").save(tmp_path / "image.png")
    Image.new("L", (4, 4), 255).save(tmp_path / "inpaint_mask.png")
    Image.new("RGB", (4, 4), "white").save(tmp_path / "complete_background.png")
    paths = {
        "image": tmp_path / "image.png",
        "mask": tmp_path / "inpaint_mask.png",
        "output": tmp_path / "complete_background.png",
    }
    status = {
        "status": "not_run",
        "sha256": {name: sha256_file(path) for name, path in paths.items()},
    }
    (tmp_path / "complete_background_status.json").write_text(json.dumps(status), encoding="utf-8")
    assert not preprocess_cache_valid(tmp_path)

    status["status"] = "external_ok"
    (tmp_path / "complete_background_status.json").write_text(json.dumps(status), encoding="utf-8")
    assert preprocess_cache_valid(tmp_path)
