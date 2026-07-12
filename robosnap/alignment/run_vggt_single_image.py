# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROBOSNAP_ROOT = Path(
    os.environ.get("ROBOSNAP_ROOT", Path(__file__).resolve().parents[2])
).expanduser().resolve()
VGGT_DIR = Path(os.environ.get("VGGT_DIR", ROBOSNAP_ROOT / "third_party" / "vggt")).expanduser().resolve()
if str(VGGT_DIR) not in sys.path:
    sys.path.insert(0, str(VGGT_DIR))

from demo_single_image import load_model, save_depth_preview, save_pcd, save_ply, select_points
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


DEFAULT_SCENES = [
    "scene01",
    "scene02",
    "scene03",
    "scene05",
    "scene07",
    "scene09",
    "scene10",
    "scene12_backup",
    "scene14_backup",
    "scene15_backup",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run VGGT single-image depth for scene folders.")
    parser.add_argument(
        "--scene-root",
        required=True,
        help="Root containing sceneXX folders.",
    )
    parser.add_argument("--scenes", nargs="*", default=DEFAULT_SCENES, help="Scene folder names to process.")
    parser.add_argument("--image-name", default="image.png", help="Image filename inside each scene.")
    parser.add_argument("--output-name", default="vggt_single_image", help="Subfolder under sam3d+fpose.")
    parser.add_argument("--model", default="facebook/VGGT-1B", help="Hugging Face model id.")
    parser.add_argument("--checkpoint", default=None, help="Optional local model.pt checkpoint path.")
    parser.add_argument("--mode", choices=["pad", "crop"], default="pad", help="Image preprocessing mode.")
    parser.add_argument("--device", default="cuda:0", help="Device, e.g. cuda:0 or cpu.")
    parser.add_argument("--conf-threshold", type=float, default=None, help="Optional depth confidence threshold.")
    parser.add_argument("--stride", type=int, default=2, help="Subsample exported PLY/PCD by image-space stride.")
    parser.add_argument("--max-points", type=int, default=300000, help="Maximum points exported to PLY/PCD.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for deterministic point subsampling.")
    parser.add_argument("--include-padding", action="store_true", help="Include padded pixels in exported PLY/PCD.")
    return parser.parse_args()


def preprocess_transform_for_original(original_size_wh, preprocessed_size_hw, mode):
    original_w, original_h = original_size_wh
    target_h, target_w = preprocessed_size_hw

    if mode == "pad":
        if original_w >= original_h:
            new_w = target_w
            new_h = round(original_h * (new_w / original_w) / 14) * 14
        else:
            new_h = target_h
            new_w = round(original_w * (new_h / original_h) / 14) * 14
        pad_left = (target_w - new_w) // 2
        pad_top = (target_h - new_h) // 2
        sx = new_w / original_w
        sy = new_h / original_h
    else:
        new_w = target_w
        new_h = round(original_h * (new_w / original_w) / 14) * 14
        start_y = max((new_h - target_h) // 2, 0)
        pad_left = 0
        pad_top = -start_y
        sx = new_w / original_w
        sy = new_h / original_h

    return {
        "scale_x": float(sx),
        "scale_y": float(sy),
        "offset_x": float(pad_left),
        "offset_y": float(pad_top),
        "resized_width": int(new_w),
        "resized_height": int(new_h),
    }


def intrinsics_to_original_pixels(intrinsic_preprocessed, transform):
    sx = transform["scale_x"]
    sy = transform["scale_y"]
    ox = transform["offset_x"]
    oy = transform["offset_y"]

    intrinsic_original = intrinsic_preprocessed.copy()
    intrinsic_original[0, 0] /= sx
    intrinsic_original[1, 1] /= sy
    intrinsic_original[0, 2] = (intrinsic_preprocessed[0, 2] - ox) / sx
    intrinsic_original[1, 2] = (intrinsic_preprocessed[1, 2] - oy) / sy
    return intrinsic_original


def make_content_mask(preprocessed_size_hw, transform):
    height, width = preprocessed_size_hw
    mask = np.zeros((height, width), dtype=bool)
    x0 = int(round(transform["offset_x"]))
    y0 = int(round(transform["offset_y"]))
    x1 = x0 + int(transform["resized_width"])
    y1 = y0 + int(transform["resized_height"])
    mask[max(y0, 0) : min(y1, height), max(x0, 0) : min(x1, width)] = True
    return mask


def run_one_scene(model, image_path, output_dir, args, device, dtype):
    output_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as original_image:
        original_size_wh = original_image.size

    images = load_and_preprocess_images([str(image_path)], mode=args.mode).to(device)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype, enabled=device.type == "cuda"):
            predictions = model(images)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])

    depth = predictions["depth"].detach().cpu().numpy()[0, 0, :, :, 0]
    depth_conf = predictions["depth_conf"].detach().cpu().numpy()[0, 0]
    extrinsic_np = extrinsic.detach().cpu().numpy()[0, 0]
    intrinsic_np = intrinsic.detach().cpu().numpy()[0, 0]
    image_np = (images.detach().cpu().numpy()[0].transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)

    transform = preprocess_transform_for_original(original_size_wh, depth.shape, args.mode)
    content_mask = make_content_mask(depth.shape, transform)
    intrinsic_original = intrinsics_to_original_pixels(intrinsic_np, transform)
    points_world = unproject_depth_map_to_point_map(
        depth[None, :, :, None],
        extrinsic_np[None],
        intrinsic_np[None],
    )[0]
    valid_mask = None if args.include_padding else content_mask
    points_export, colors_export = select_points(points_world, depth, depth_conf, image_np, args, valid_mask=valid_mask)

    np.save(output_dir / "depth.npy", depth)
    np.save(output_dir / "depth_conf.npy", depth_conf)
    np.save(output_dir / "points_world.npy", points_world)
    np.save(output_dir / "content_mask.npy", content_mask)
    np.save(output_dir / "intrinsic_preprocessed.npy", intrinsic_np)
    np.save(output_dir / "intrinsic_original_pixels.npy", intrinsic_original)
    np.save(output_dir / "extrinsic.npy", extrinsic_np)
    np.savetxt(output_dir / "intrinsic_preprocessed.txt", intrinsic_np, fmt="%.10g")
    np.savetxt(output_dir / "intrinsic_original_pixels.txt", intrinsic_original, fmt="%.10g")
    np.savetxt(output_dir / "extrinsic.txt", extrinsic_np, fmt="%.10g")
    np.savez(
        output_dir / "geometry.npz",
        depth=depth,
        depth_conf=depth_conf,
        intrinsic_preprocessed=intrinsic_np,
        intrinsic_original_pixels=intrinsic_original,
        extrinsic=extrinsic_np,
        points_world=points_world,
        content_mask=content_mask,
        image_preprocessed=image_np,
    )

    Image.fromarray(image_np).save(output_dir / "image_preprocessed.png")
    save_depth_preview(depth, output_dir / "depth_vis.png")
    save_ply(output_dir / "point_cloud.ply", points_export, colors_export)
    save_pcd(output_dir / "point_cloud.pcd", points_export, colors_export)

    metadata = {
        "input_image": str(image_path),
        "original_size_wh": list(original_size_wh),
        "preprocessed_size_hw": list(depth.shape),
        "preprocess_mode": args.mode,
        "preprocess_transform_original_to_preprocessed": transform,
        "model": args.model,
        "checkpoint": args.checkpoint,
        "exported_points": int(len(points_export)),
        "include_padding": bool(args.include_padding),
        "conf_threshold": args.conf_threshold,
        "stride": int(max(args.stride, 1)),
        "coordinate_convention": "OpenCV: x right, y down, z forward",
        "extrinsic_convention": "camera-from-world [R|t], shape 3x4",
        "intrinsic_preprocessed_convention": "pixel intrinsics for image_preprocessed.png",
        "intrinsic_original_pixels_convention": "same VGGT camera converted back to original image pixel coordinates",
    }
    with open(output_dir / "camera.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                **metadata,
                "intrinsic_preprocessed": intrinsic_np.tolist(),
                "intrinsic_original_pixels": intrinsic_original.tolist(),
                "extrinsic": extrinsic_np.tolist(),
            },
            f,
            indent=2,
        )

    return output_dir, len(points_export)


def main():
    args = parse_args()
    scene_root = Path(args.scene_root)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    else:
        dtype = torch.float32

    model = load_model(args, device)

    for scene in args.scenes:
        scene_dir = scene_root / scene
        image_path = scene_dir / args.image_name
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image: {image_path}")

        output_dir = scene_dir / "sam3d+fpose" / args.output_name
        written_dir, point_count = run_one_scene(model, image_path, output_dir, args, device, dtype)
        print(f"{scene}: wrote {written_dir} ({point_count} points)")


if __name__ == "__main__":
    main()
