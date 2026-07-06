#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
from pathlib import Path
import uuid
import imageio
import numpy as np

from inference import Inference, ready_gaussian_for_video_rendering, load_image, load_masks, display_image, make_scene, render_video, interactive_visualizer


# ============================================================
# 1. Imports and Model Loading
# ============================================================

PATH = os.getcwd()
TAG = "hf"
config_path = os.environ.get("SAM3D_CONFIG", "checkpoints/hf/pipeline.yaml")
inference = Inference(config_path, compile=False)


# ============================================================
# 2. Load input image to lift to 3D (multiple objects)
# ============================================================

IMAGE_PATH = os.environ.get("SAM3D_DEMO_IMAGE", str(Path("notebook/images/demo_multi_object/image.png")))
IMAGE_NAME = os.path.basename(os.path.dirname(IMAGE_PATH))
OUTPUT_DIR = Path(os.environ.get("SAM3D_DEMO_OUTPUT_DIR", os.path.dirname(IMAGE_PATH)))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

image = load_image(IMAGE_PATH)
masks = load_masks(os.path.dirname(IMAGE_PATH), extension=".png")
display_image(image, masks)


# ============================================================
# 3. Generate Gaussian Splats
# ============================================================

outputs = [inference(image, mask, seed=42) for mask in masks]


# ============================================================
# 4. Visualize Gaussian Splat of the Scene
# ============================================================

# 4a. Animated Gif

scene_gs = make_scene(*outputs)
# export posed gaussian splatting (as point cloud)
scene_gs.save_ply(str(OUTPUT_DIR / f"{IMAGE_NAME}_posed.ply"))

scene_gs = ready_gaussian_for_video_rendering(scene_gs)
# export gaussian splatting (as point cloud)
scene_gs.save_ply(str(OUTPUT_DIR / f"{IMAGE_NAME}.ply"))

video = render_video(
    scene_gs,
    r=1,
    fov=60,
    resolution=512,
)["color"]

# save video as gif
imageio.mimsave(
    str(OUTPUT_DIR / f"{IMAGE_NAME}.gif"),
    video,
    format="GIF",
    duration=1000 / 30,  # default assuming 30fps from the input MP4
    loop=0,  # 0 means loop indefinitely
)

# # notebook display
# ImageDisplay(url=f"gaussians/multi/{IMAGE_NAME}.gif?cache_invalidator={uuid.uuid4()}",)


# 4b. Interactive Visualizer
# (might take a while to load - black screen)
# interactive_visualizer(f"{PATH}/gaussians/multi/{IMAGE_NAME}.ply")
