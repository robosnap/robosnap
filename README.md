# RoboSnap

<p align="center">
  <b>One-Shot Real-to-Sim Scene Generation for Generalizable Robot Learning and Evaluation</b>
</p>

<p align="center">
  <a href="https://robosnap.github.io"><img src="https://img.shields.io/badge/Project-Page-4F6F52" alt="Project Page"></a>
  <a href="robosnap/paper.pdf"><img src="https://img.shields.io/badge/Paper-PDF-B85C38" alt="Paper"></a>
  <a href="#quick-start-docker"><img src="https://img.shields.io/badge/GUI-Quick_Start-345995" alt="GUI Quick Start"></a>
</p>

RoboSnap reconstructs real-world scenes as simulation-ready assets from a short video or image sequence. This first public release is GUI-first: interactive segmentation, mask workspace management, mask-to-3D asset generation, scene composition, and articulated-object refinement.

<p align="center">
  <img src="assets/gui_preview.png" alt="RoboSnap GUI preview" width="760">
</p>

## Release Plan

- [x] GUI tool release
- [ ] Auto pipeline, including sim-ready scene preparation
- [ ] Real-robot deployment tutorial
- [ ] Evaluation code
- [ ] DROID-Sim dataset release

## Quick Start: Docker

Docker is the recommended path for release testing because the GUI uses several heavy third-party stacks.

Prerequisites on the host:

```bash
docker version
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

Clone and enter the repo:

```bash
git clone https://github.com/robosnap/robosnap.git
cd robosnap
```

Build the image:

```bash
docker build -t robosnap-gui:local .
```

Run a dry launch first. This checks path resolution and prints the GUI command without starting Gradio:

```bash
docker run --gpus all --rm -it \
  -e DRY_RUN=1 \
  -v "$(pwd)/checkpoints:/workspace/robosnap/checkpoints" \
  -v "$(pwd)/outputs:/workspace/robosnap/outputs" \
  robosnap-gui:local
```

Start the GUI:

```bash
docker run --gpus all --rm -it \
  --ipc=host --shm-size=16g \
  -p 7897:7897 \
  -v "$(pwd)/checkpoints:/workspace/robosnap/checkpoints" \
  -v "$(pwd)/outputs:/workspace/robosnap/outputs" \
  robosnap-gui:local
```

Open:

```text
http://127.0.0.1:7897
```

The default input video is `examples/video.mp4`. The default output workspace is `outputs/example/multi_mask`.

## Checkpoints

Model weights are not committed. `checkpoints/` is the default local mount point and is git-ignored except for `.gitkeep`.

Expected local layout:

```text
checkpoints/
  sam3/
    sam3.pt
  sam-3d-objects/
    pipeline.yaml
    ... files referenced by pipeline.yaml ...
  articulate/
    articulate.safetensors
  sonata/
    sonata.pth
  hf_cache/
  torch_cache/
```

If you already have local weights, materialize them into this layout:

```bash
LOCAL_SAM3_CKPT=/path/to/sam3.pt \
LOCAL_SAM3D_CHECKPOINT_DIR=/path/to/sam3d/checkpoints \
LOCAL_ARTICULATE_CKPT=/path/to/p3sam.safetensors \
LOCAL_SONATA_CKPT=/path/to/sonata.pth \
MATERIALIZE_MODE=symlink \
bash scripts/gui/bash/copy_checkpoints_from_local.sh
```

To preview or run Hugging Face downloads, use the checkpoint helper. Private or unreleased checkpoint repos must be passed explicitly:

```bash
python3 scripts/gui/python/download_checkpoints.py --dry-run --skip-optional
python3 scripts/gui/python/download_checkpoints.py --sam3d-repo <your-sam3d-checkpoint-repo>
```

You can also mount an external checkpoint directory into Docker:

```bash
docker run --gpus all --rm -it \
  -p 7897:7897 \
  -v /path/to/checkpoints:/workspace/robosnap/checkpoints \
  -v "$(pwd)/outputs:/workspace/robosnap/outputs" \
  robosnap-gui:local
```

## Configuration

RoboSnap uses one path prefix: `ROBOSNAP_ROOT`. In Docker it is `/workspace/robosnap`; in a native clone it defaults to the cloned repo root.

`configs/gui.env` is optional. Create it only when you need to override defaults:

```bash
cp configs/gui.env.example configs/gui.env
$EDITOR configs/gui.env
```

Common overrides:

```bash
ROBOSNAP_ROOT=/path/to/robosnap
CHECKPOINT_DIR=${ROBOSNAP_ROOT}/checkpoints
VIDEO=${ROBOSNAP_ROOT}/examples/video.mp4
OUT_DIR=${ROBOSNAP_ROOT}/outputs/example/multi_mask
PORT=7897
MAX_FRAMES=20
```

`VIDEO`, `OUT_DIR`, and `CHECKPOINT_DIR` may be absolute paths. Relative paths are resolved under `ROBOSNAP_ROOT`.

## Native Install

Use this path when Docker is unavailable or when you want to debug the repo directly. The launcher supports three Python runtimes: GUI/video segmentation, mask-to-3D asset generation, and the Articulate Tool.

Install the native conda environments with the helper script:

```bash
bash scripts/install_native_envs.sh --dry-run
bash scripts/install_native_envs.sh -y
```

The script creates `robosnap-gui`, `robosnap-asset`, and `robosnap-articulate`, then writes `configs/gui.env` so `bash scripts/run_gui.sh` uses the new envs. Useful options:

```bash
bash scripts/install_native_envs.sh --help
bash scripts/install_native_envs.sh -y --skip-asset --skip-articulate
bash scripts/install_native_envs.sh -y --force-env
```

After installation:

```bash
bash scripts/run_gui.sh
```

## GUI Workflow

1. Load the configured video or upload a video/image in the GUI.
2. Add positive and negative prompt points.
3. Confirm and preview the propagated object mask.
4. Save masks for each object.
5. Generate GLB assets from the saved mask workspace.
6. Compose generated assets into a scene preview.
7. Select articulated objects and launch the Articulate Tool on child ports.
8. Save joint JSON/USD files and final scene artifacts.

## Outputs

The GUI writes a workspace centered on `multi_mask/` and `single_mask/`:

```text
outputs/example/
  multi_mask/
    video.mp4
    segmented_video.mp4
    background/
      000000.png
    object_name_a/
      all_mask/
      top20_mask/
      object_name_a.glb
      object_name_a_joints.json
      object_name_a_joints.usd
  single_mask/
    image.png
    0.png
    0.glb
    scene_composed.glb
```

Key files:

- `all_mask/`: propagated masks for the object.
- `top*_mask/`: selected masks used for 3D generation.
- `{object}.glb`: generated object asset.
- `{object}_joints.json`: GUI-authored joint metadata.
- `{object}_joints.usd`: USD joint export when enabled.
- `single_mask/scene_composed.glb`: composed scene preview.

## Remote Access

`scripts/run_gui.sh` starts Gradio on the machine where it runs. It does not create SSH tunnels.

For SSH local forwarding:

```sshconfig
Host robosnap-gui
  HostName <remote-host>
  User <user>
  Port <ssh-port>
  IdentityFile ~/.ssh/id_ed25519
  LocalForward 7897 127.0.0.1:7897
  LocalForward 8180 127.0.0.1:8180
  ExitOnForwardFailure yes
```

Start the GUI on the remote machine and open `http://127.0.0.1:7897` locally. Forward additional Articulate Tool ports as needed, starting from `ARTICULATE_BASE_PORT`.

For a temporary Gradio public link:

```bash
SHARE=1 PUBLIC_DEMO=1 bash scripts/run_gui.sh
```

Public demo mode narrows Gradio file access to the input file, output workspace, download directory, and explicit allowed roots. Uploaded files are stored in the configured Gradio temp directory; keep it under `outputs/` and keep `checkpoints/`, `data/`, `.git/`, and private SSH directories blocked.

The public Gradio link only proxies the main GUI port. Articulate child ports need a separate tunnel, reverse proxy, or `ARTICULATE_PUBLIC_URL_TEMPLATE`.

## Mask-to-Assets CLI

The GUI can generate assets interactively. To rerun asset generation from an existing mask workspace:

```bash
bash scripts/gui/bash/run_mask_to_assets.sh --mode multi --input-path outputs/example/multi_mask
bash scripts/gui/bash/run_mask_to_assets.sh --mode single --mask-dir outputs/example/single_mask
DRY_RUN=1 bash scripts/gui/bash/run_mask_to_assets.sh --mode auto --input-path outputs/example/multi_mask
```



## License

The RoboSnap source code is released under the Apache License 2.0.
See the [LICENSE](LICENSE) file for details.

## Third-Party Code and Models

RoboSnap includes adapted third-party components under `third_party/`.
Each component remains subject to its original license and attribution
requirements. Please refer to the corresponding subdirectory for details.

Model checkpoints are not included in this repository and may be subject
to their respective licenses from the original authors.
