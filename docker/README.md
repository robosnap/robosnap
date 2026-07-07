# RoboSnap Docker Notes

Build from the repository root:

```bash
docker build -t robosnap-gui .
```

Run the GUI with checkpoints and outputs mounted under the repository prefix:

```bash
docker run --gpus all --rm -it \
  -p 7897:7897 \
  -v "$(pwd)/checkpoints:/workspace/robosnap/checkpoints" \
  -v "$(pwd)/outputs:/workspace/robosnap/outputs" \
  robosnap-gui
```

The container sets:

```bash
ROBOSNAP_ROOT=/workspace/robosnap
PY_SAM3=/opt/conda/envs/robosnap-gui/bin/python
PY_ASSET=/opt/conda/envs/robosnap-asset/bin/python
PY_ARTICULATE=/opt/conda/envs/robosnap-articulate/bin/python
```

Weights are not baked into the image. Put them under `checkpoints/` on the host
and mount that directory into `/workspace/robosnap/checkpoints`.
