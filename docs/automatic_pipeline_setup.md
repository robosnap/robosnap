# Automatic Pipeline Setup

## Conda

Requirements: Ubuntu 22.04, NVIDIA driver compatible with CUDA 12.8, Conda, Git, and Patch.

~~~bash
bash scripts/install_auto_pipeline.sh -y
~~~

The installer creates:

| Environment | Runtime |
| --- | --- |
| robosnap-sam3 | SAM3 segmentation |
| robosnap-asset | SAM3D, VGGT, ICP, gravity alignment, rendering |
| robosnap-lyra | Lyra-2 video generation and Gaussian reconstruction |
| robosnap-sim | RoboSnap SF-Real2Sim refinement |

It also fetches pinned VGGT and Lyra sources and writes configs/auto_pipeline.env.

Authenticate with Hugging Face before downloading gated SAM3D weights:

~~~bash
hf auth login
bash scripts/install_auto_pipeline.sh -y --skip-sam3 --skip-asset --skip-lyra --skip-sim \
  --download-checkpoints
~~~

Lyra-2 weights are about 91 GB and use the NVIDIA model license. Download them explicitly:

~~~bash
bash scripts/install_auto_pipeline.sh -y --skip-sam3 --skip-asset --skip-lyra --skip-sim \
  --download-lyra --accept-lyra-license
~~~

Use HF_HOME to place the cache on a persistent disk. The model script links cached snapshots into checkpoints/ by default; pass --copy-checkpoints when links are unsuitable.

Configure OBJECT_FILE or VLM_COMMAND, set the image-edit API key, then run:

~~~bash
bash scripts/run_auto_pipeline.sh
~~~

## Docker

Build the complete four-environment image:

~~~bash
docker build -f docker/Dockerfile.auto -t robosnap-auto:local .
~~~

Weights stay outside the image. Download them into the mounted checkpoints/ directory, then run:

~~~bash
docker compose -f docker-compose.auto.yml run --rm pipeline \
  --object-file /workspace/robosnap/examples/object.txt
~~~

The default low-memory Lyra profile is 352x624. SAM3D needs at least 32 GB VRAM; Lyra is validated with offload on 48 GB and is safer on an 80 GB GPU.
