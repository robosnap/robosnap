# RoboSnap

<p align="center">
  <b>One-Shot Real-to-Sim Scene Generation for Generalizable Robot Learning and Evaluation</b>
</p>

<p align="center">
  <a href="https://robosnap.github.io"><img src="https://img.shields.io/badge/Project-Page-4F6F52" alt="Project Page"></a>
  <a href="robosnap/paper.pdf"><img src="https://img.shields.io/badge/Paper-PDF-B85C38" alt="Paper"></a>
  <a href="#quick-start-gui"><img src="https://img.shields.io/badge/GUI-Quick_Start-345995" alt="GUI Quick Start"></a>
</p>


RoboSnap reconstructs real-world scenes as reusable simulation-ready environments. The first public release is GUI-first: interactive video/image segmentation, mask management, mask-to-3D asset generation, scene composition, and articulated-object refinement through a separate Articulate Tool page.



## Quick Start: GUI

Prepare the repository and start from the built-in defaults:

```bash
git clone <repo-url> robosnap
cd robosnap
bash scripts/run_gui.sh
```

The launcher uses `ROBOSNAP_ROOT` as the single path prefix. It defaults to the repository root, so outputs default to `${ROBOSNAP_ROOT}/outputs/...`. Create `configs/gui.env` only when you need local overrides such as separate Python runtimes, checkpoint locations, input video, output path, or port:

```bash
cp configs/gui.env.example configs/gui.env
$EDITOR configs/gui.env
```

```bash
# configs/gui.env
# ROBOSNAP_ROOT=/path/to/robosnap  # optional; defaults to the cloned repo
VIDEO=${ROBOSNAP_ROOT}/examples/video.mp4
OUT_DIR=${ROBOSNAP_ROOT}/outputs/example/multi_mask
PORT=7897
```

Absolute paths work as written. Relative `VIDEO`, `OUT_DIR`, and `CHECKPOINT_DIR` values are resolved under `ROBOSNAP_ROOT`.

Preview the checkpoint download plan:

```bash
python3 scripts/gui/python/download_checkpoints.py --dry-run --skip-optional
```

Open:

```text
http://127.0.0.1:7897
```

GUI Preview:

<p align="center">
  <img src="assets/gui_preview.png" alt="RoboSnap GUI preview" width="600">
</p>

## Environment

RoboSnap can run in one Python environment only if that environment can import every dependency. In practice, the dependency stacks are heavy, so the launcher supports three runtimes:

```text
PY_SAM3        main GUI and video segmentation runtime
PY_ASSET       mask-to-3D asset generation runtime
PY_ARTICULATE  Articulate Tool runtime launched on per-object ports
```

Set these in `configs/gui.env`:

```bash
PY_SAM3=/path/to/robosnap-gui/bin/python
PY_ASSET=/path/to/robosnap-asset/bin/python
PY_ARTICULATE=/path/to/robosnap-articulate/bin/python
```

Typical setup outline:

```bash
# Runtime 1: GUI + video segmentation
conda create -n robosnap-gui python=3.10
conda activate robosnap-gui
pip install -U pip
# Install torch/torchvision matching your CUDA first.
pip install -e .
pip install -e third_party/sam3

# Runtime 2: asset generation
conda create -n robosnap-asset python=3.11
conda activate robosnap-asset
pip install -U pip
# Install torch/torchvision matching your CUDA first.
pip install -e "third_party/sam-3d-objects[inference,p3d]"

# Runtime 3: Articulate Tool
conda create -n robosnap-articulate python=3.10
conda activate robosnap-articulate
pip install -U pip
# Install torch/torchvision matching your CUDA first.
pip install viser fpsample trimesh numba gradio scikit-learn websockets
cd third_party/Hunyuan3D-Part/P3-SAM/utils/chamfer3D
python setup.py install
cd ../../../../..
```

Exact CUDA wheels depend on your driver and PyTorch matrix. If optional extras fail, install the relevant requirements from the vendored third-party snapshots manually.

## Checkpoints

Model weights are not committed. `checkpoints/` is kept only as an empty default download directory in source control; its contents are git-ignored. Keep weights there locally or point `configs/gui.env` to external paths.

Expected local layout:

```text
checkpoints/
  sam3/
    sam3.pt
  sam-3d-objects/
    pipeline.yaml
    ss_generator.yaml
    ss_generator.ckpt
    slat_generator.yaml
    slat_generator.ckpt
    ss_decoder.yaml
    ss_decoder.ckpt
    slat_decoder_gs.yaml
    slat_decoder_gs.ckpt
    slat_decoder_gs_4.yaml
    slat_decoder_gs_4.ckpt
    slat_decoder_mesh.yaml
    slat_decoder_mesh.ckpt
    dinov2_vitb14_pretrain.pth
  articulate/
    articulate.safetensors
  sonata/
    sonata.pth
  hf_cache/
  torch_cache/
```

Configure these paths in `configs/gui.env`:

```bash
SAM3_CKPT=${CHECKPOINT_DIR}/sam3/sam3.pt
SAM3D_CONFIG=${CHECKPOINT_DIR}/sam-3d-objects/pipeline.yaml
ARTICULATE_CKPT=${CHECKPOINT_DIR}/articulate/articulate.safetensors
SONATA_CACHE_DIR=${CHECKPOINT_DIR}/sonata
HF_HOME=${CHECKPOINT_DIR}/hf_cache
TORCH_HOME=${CHECKPOINT_DIR}/torch_cache
```

Download helper:

```bash
python3 scripts/gui/python/download_checkpoints.py --dry-run --skip-optional
python3 scripts/gui/python/download_checkpoints.py --sam3d-repo <your-sam3d-checkpoint-repo>
```

If the weights already exist locally, materialize them into the expected layout with:

```bash
LOCAL_SAM3_CKPT=/path/to/sam3.pt LOCAL_SAM3D_CHECKPOINT_DIR=/path/to/sam-3d-objects/checkpoints LOCAL_ARTICULATE_CKPT=/path/to/articulate.safetensors LOCAL_SONATA_CKPT=/path/to/sonata.pth MATERIALIZE_MODE=symlink bash scripts/gui/bash/copy_checkpoints_from_local.sh
```

## Temporary Public Demo on DSW

For a short public demo through a Gradio share link, keep the demo isolated from local checkpoints and private datasets:

```bash
# configs/gui.env
VIDEO=${ROBOSNAP_ROOT}/examples/video.mp4
OUT_DIR=${ROBOSNAP_ROOT}/outputs/public_demo/multi_mask
PORT=7897
SHARE=1
PUBLIC_DEMO=1
GRADIO_ALLOWED_ROOTS=
GRADIO_BLOCKED_ROOTS=/cpfs/user/zhangshujie/ikea:${ROBOSNAP_ROOT}/checkpoints:/cpfs/user/zhangshujie/.ssh
GRADIO_MAX_FILE_SIZE=200mb
GRADIO_DELETE_CACHE_FREQUENCY=3600
GRADIO_DELETE_CACHE_AGE=21600
```

Then start the GUI:

```bash
bash scripts/run_gui.sh
```

The terminal prints a `gradio.live` URL when `SHARE=1`. The main segmentation GUI, mask saving, asset generation controls, and result zip download run through that URL. Click `Prepare Results Download` after saving masks to create a downloadable zip under `outputs/public_demo/downloads/`.

`PUBLIC_DEMO=1` narrows Gradio file access to the input video, output workspace, download directory, and explicit `GRADIO_ALLOWED_ROOTS`. Do not add the repository root, checkpoint directory, private datasets, or SSH directories to `GRADIO_ALLOWED_ROOTS`.

The Articulate Tool opens per-object child ports. A Gradio share link only proxies the main GUI port, so public Articulate access needs a DSW port proxy or reverse proxy configured through `ARTICULATE_PUBLIC_URL_TEMPLATE`.

## Remote GUI Access

`scripts/run_gui.sh` starts Gradio on the remote machine; it does not create SSH tunnels. If the GUI runs remotely and the browser runs locally, keep a separate SSH session open with local forwards:

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

Then start the GUI on the remote machine and open `http://127.0.0.1:7897` locally. Forward additional Articulate Tool ports as needed, starting from `ARTICULATE_BASE_PORT`.

Use `RemoteForward` only when a remote port should forward back to a service running on the SSH client side.

## GUI Workflow

1. Load the configured video or image.
2. Click positive/negative points and confirm a mask.
3. Preview the propagated single-object video.
4. Save masks for each object.
5. End segmentation and scan the saved object folders.
6. Generate GLB assets from the mask workspace.
7. Compose generated assets into a scene preview.
8. Select articulated objects and launch the Articulate Tool on separate ports.
9. Save joint JSON/USD files and final scene artifacts.

## Expected Outputs

The GUI writes a case workspace centered on `multi_mask/` and `single_mask/`. A typical completed case looks like this:

```text
case/
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
    object_name_b/
      all_mask/
      top20_mask/
      object_name_b.glb
  single_mask/
    image.png
    0.png
    0.glb
    1.png
    1.glb
    scene_composed.glb
```

Key files:

- `all_mask/`: propagated masks for the object.
- `top*_mask/`: selected high-quality masks used for 3D generation.
- `{object}.glb`: generated object asset.
- `{object}_joints.json`: GUI-authored joint metadata.
- `{object}_joints.usd`: USD joint export when enabled.
- `single_mask/scene_composed.glb`: composed scene preview.

## Mask-to-Assets CLI

The GUI can generate assets interactively. To rerun the asset-generation stage from an existing mask workspace:

```bash
bash scripts/gui/bash/run_mask_to_assets.sh --mode multi --input-path outputs/example/multi_mask
bash scripts/gui/bash/run_mask_to_assets.sh --mode single --mask-dir outputs/example/single_mask
DRY_RUN=1 bash scripts/gui/bash/run_mask_to_assets.sh --mode auto --input-path outputs/example/multi_mask
```

## Manual Release Checklist

Before publishing the GUI release, check these on a configured machine with real checkpoints:

- [ ] `bash scripts/run_gui.sh` starts without a traceback.
- [ ] The first frame from `examples/video.mp4` is visible.
- [ ] Prompt points are smaller and do not resize the image unexpectedly.
- [ ] Positive and negative points update the mask preview.
- [ ] `Save Masks & Next` creates object folders under `multi_mask/`.
- [ ] Each object has `all_mask/` and `top*_mask/`.
- [ ] Background/non-object masks are not listed as articulated objects.
- [ ] End segmentation moves to the object/mesh stage.
- [ ] Asset generation creates object GLBs and `single_mask/scene_composed.glb`.
- [ ] Existing GLBs are detected on rerun instead of always regenerating.
- [ ] The Articulate Tool opens on `ARTICULATE_BASE_PORT` or the next free port.
- [ ] A second articulated object uses a different port.
- [ ] Joint JSON/USD files are written beside the object GLB.
- [ ] No generated files are written outside `OUT_DIR`, `OUT_DIR/..`, or configured cache/checkpoint directories.

Keep release-review artifacts out of source control: generated `outputs/`, terminal logs, screenshots, checkpoints, caches, GLBs, USD files, and local env files.

## Release Roadmap

- [x] GUI tool: segmentation, mask organization, asset generation, scene composition, and articulated-object annotation/refinement.
- [ ] Auto pipeline, including sim-ready scene construction.
- [ ] Real-robot deployment tutorial.
- [ ] Evaluation code.
- [ ] DROID-sim dataset release and download instructions.


## License and Third-party Notes

First-party RoboSnap code is released under Apache-2.0; see `LICENSE` and `NOTICE`. The top-level project license does not relicense anything under `third_party/`.

GUI-critical third-party snapshots currently include:

| Component | Release path | License boundary |
| --- | --- | --- |
| sam3 | `third_party/sam3` | SAM License / upstream terms; not Apache-2.0 RoboSnap code. |
| sam-3d-objects | `third_party/sam-3d-objects` | SAM License / upstream terms plus nested third-party license files. |
| Hunyuan3D-Part / P3-SAM | `third_party/Hunyuan3D-Part` | Tencent Hunyuan 3D-Part Community License Agreement; review redistribution and acceptable-use terms before release. |

Users should use the bundled adapted snapshots or published RoboSnap forks/submodules at fixed commits. Unmodified upstream clones are not assumed to work with the RoboSnap GUI scripts. Model weights, datasets, generated assets, and local outputs remain under their own terms and are not covered by the first-party Apache-2.0 license.
