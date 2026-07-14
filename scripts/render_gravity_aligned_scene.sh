#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROBOSNAP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY_RENDER="${PY_RENDER:-${PY_ALIGN:-${PY_ASSET:-python}}}"

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="$(dirname "${PY_RENDER}"):${PATH}"
export PYTHONUTF8=1
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MAX_JOBS="${MAX_JOBS:-1}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  exec "${PY_RENDER}" -m robosnap.rendering.render_layered_scene --help
fi

if [[ $# -gt 0 ]]; then
  SCENE_DIR="$1"
  shift
else
  SCENE_DIR="${SCENE_DIR:-${ROOT}/outputs/automatic}"
fi
FOREGROUND="${FOREGROUND:-${SCENE_DIR}/fully_refined_foreground.glb}"
BACKGROUND_PLY="${BACKGROUND_PLY:-${SCENE_DIR}/gravity_aligned_background.ply}"
OUTPUT_PLY="${OUTPUT_PLY:-${SCENE_DIR}/layered_preview.ply}"
OUTPUT_IMAGE="${OUTPUT_IMAGE:-${SCENE_DIR}/layered_preview.png}"
STATUS_JSON="${STATUS_JSON:-${SCENE_DIR}/layered_preview_status.json}"
CAMERA_NPZ="${CAMERA_NPZ:-${SCENE_DIR}/background/lyra2_gs/cameras.npz}"
GRAVITY_TRANSFORM="${GRAVITY_TRANSFORM:-${SCENE_DIR}/gravity_alignment.json}"
FOREGROUND_CAMERA_JSON="${FOREGROUND_CAMERA_JSON:-${SCENE_DIR}/reconstruction/vggt/camera.json}"
FOREGROUND_SAMPLES="${FOREGROUND_SAMPLES:-100000}"
RENDER_DEVICE="${RENDER_DEVICE:-${ROBOSNAP_DEVICE:-cuda:0}}"

exec "${PY_RENDER}" -m robosnap.rendering.render_layered_scene \
  --foreground "${FOREGROUND}" \
  --background-ply "${BACKGROUND_PLY}" \
  --camera-npz "${CAMERA_NPZ}" \
  --gravity-transform "${GRAVITY_TRANSFORM}" \
  --foreground-camera-json "${FOREGROUND_CAMERA_JSON}" \
  --output-ply "${OUTPUT_PLY}" \
  --output-image "${OUTPUT_IMAGE}" \
  --status-json "${STATUS_JSON}" \
  --foreground-samples "${FOREGROUND_SAMPLES}" \
  --device "${RENDER_DEVICE}" \
  "$@"
