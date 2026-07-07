# Docker Notes

Build from the repository root:

```bash
docker build -t robosnap-gui:local .
```

Run the GUI with checkpoints and outputs mounted under the repository prefix:

```bash
docker run --gpus all --rm -it \
  --ipc=host --shm-size=16g \
  -p 7897:7897 \
  -v "$(pwd)/checkpoints:/workspace/robosnap/checkpoints" \
  -v "$(pwd)/outputs:/workspace/robosnap/outputs" \
  robosnap-gui:local
```

The container sets:

```bash
ROBOSNAP_ROOT=/workspace/robosnap
PY_SAM3=/opt/conda/envs/robosnap-gui/bin/python
PY_ASSET=/opt/conda/envs/robosnap-asset/bin/python
PY_ARTICULATE=/opt/conda/envs/robosnap-articulate/bin/python
```

Weights are not baked into the image. Put them under `checkpoints/` on the host or mount an external checkpoint directory into `/workspace/robosnap/checkpoints`.
