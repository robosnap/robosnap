# Automatic Pipeline Setup

## Conda

Requirements: Ubuntu 22.04, an NVIDIA driver compatible with CUDA 12.8, Conda, and Git.

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

The four runtimes are isolated for their incompatible CUDA and PyTorch requirements. The installer fetches pinned VGGT and Lyra sources and writes `configs/auto_pipeline.env`; users still run the pipeline through one command.

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

Set `HF_HOME` before installation to use a persistent cache. Downloads are linked into `checkpoints/` by default; pass `--copy-checkpoints` when links are unsuitable.

The default provider uses one Gemini key for VLM object discovery and semantic background editing:

~~~bash
export GEMINI_API_KEY=<your-api-key>
bash scripts/run_auto_pipeline.sh
~~~

Run another image in its chosen output directory:

~~~bash
INPUT_IMAGE=/path/to/image.png OUTPUT_DIR=/path/to/result \
  bash scripts/run_auto_pipeline.sh
~~~

To bypass automatic object discovery, set `OBJECT_FILE` to a text file with one segmentation prompt per line or a JSON file with an `objects` list. A custom `VLM_COMMAND` must write:

~~~json
{"objects":[{"name":"table","prompt":"complete table","fallback_prompt":"table","bbox_xyxy":[0.1,0.3,0.9,1.0],"support_parent_id":-1,"support_relation":"none"},{"name":"cup","prompt":"blue cup on the table","fallback_prompt":"blue cup","bbox_xyxy":[0.4,0.4,0.5,0.7],"support_parent_id":0,"support_relation":"on"}]}
~~~

`bbox_xyxy` uses normalized image coordinates. `support_parent_id` references an earlier object or is `-1`; `support_relation` is `on`, `inside`, or `none`. `INPAINT_COMMAND` receives `{image}`, `{mask}`, `{prompt}`, `{output}`, and `{status}` placeholders. See `configs/auto_pipeline.env.example` for both command templates.

## Docker

Build the complete four-environment image:

~~~bash
docker build -f docker/Dockerfile.auto -t robosnap-auto:local .
~~~

Weights stay outside the image. Download them into the mounted `checkpoints/` directory:

~~~bash
docker compose -f docker-compose.auto.yml run --rm \
  --entrypoint /opt/conda/envs/robosnap-asset/bin/python pipeline \
  scripts/download_auto_checkpoints.py --core
docker compose -f docker-compose.auto.yml run --rm \
  --entrypoint /opt/conda/envs/robosnap-asset/bin/python pipeline \
  scripts/download_auto_checkpoints.py --lyra --accept-lyra-license
~~~

Run the example:

~~~bash
export GEMINI_API_KEY=<your-api-key>
docker compose -f docker-compose.auto.yml run --rm pipeline \
  --image /workspace/robosnap/examples/test1.png \
  --output-dir /workspace/robosnap/outputs/automatic
~~~

The default low-memory Lyra profile is 352x624. SAM3D needs at least 32 GB VRAM; Lyra is validated with offload on 48 GB and is safer on an 80 GB GPU.
