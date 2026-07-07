#!/usr/bin/env bash
set -euo pipefail

export ROBOSNAP_ROOT="${ROBOSNAP_ROOT:-/workspace/robosnap}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ROBOSNAP_ROOT}/checkpoints}"
export HF_HOME="${HF_HOME:-${CHECKPOINT_DIR}/hf_cache}"
export TORCH_HOME="${TORCH_HOME:-${CHECKPOINT_DIR}/torch_cache}"

export PY_SAM3="${PY_SAM3:-/opt/conda/envs/robosnap-gui/bin/python}"
export PY_ASSET="${PY_ASSET:-/opt/conda/envs/robosnap-asset/bin/python}"
export PY_ARTICULATE="${PY_ARTICULATE:-/opt/conda/envs/robosnap-articulate/bin/python}"

export PORT="${PORT:-7897}"
export SHARE="${SHARE:-0}"
export PUBLIC_DEMO="${PUBLIC_DEMO:-0}"
export ROBOSNAP_KEEP_PROXY="${ROBOSNAP_KEEP_PROXY:-0}"

mkdir -p "${CHECKPOINT_DIR}" "${ROBOSNAP_ROOT}/outputs"

cd "${ROBOSNAP_ROOT}"
exec "$@"
