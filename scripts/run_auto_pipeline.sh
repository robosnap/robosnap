#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROBOSNAP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
AUTO_ENV="${ROBOSNAP_AUTO_ENV_FILE:-${ROOT}/configs/auto_pipeline.env}"
export ROBOSNAP_ROOT="${ROBOSNAP_ROOT:-${ROOT}}"

if [[ -f "${AUTO_ENV}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${AUTO_ENV}"
  set +a
fi

export PYTHONPATH="${ROBOSNAP_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PY_AUTO="${PY_AUTO:-python}"
INPUT_IMAGE="${INPUT_IMAGE:-${ROBOSNAP_ROOT}/examples/test1.png}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROBOSNAP_ROOT}/outputs/automatic}"
DEVICE="${DEVICE:-cuda:0}"

args=(
  -m robosnap.pipeline.auto_layered_scene
  --image "${INPUT_IMAGE}"
  --output-dir "${OUTPUT_DIR}"
  --device "${DEVICE}"
)

[[ -n "${OBJECT_PROMPTS:-}" ]] && args+=(--objects "${OBJECT_PROMPTS}")
[[ -n "${OBJECT_FILE:-}" ]] && args+=(--object-file "${OBJECT_FILE}")
[[ -n "${VLM_COMMAND:-}" ]] && args+=(--vlm-command "${VLM_COMMAND}")
[[ -n "${VLM_PROMPT:-}" ]] && args+=(--vlm-prompt "${VLM_PROMPT}")
[[ -n "${INPAINT_COMMAND:-}" ]] && args+=(--inpaint-command "${INPAINT_COMMAND}")
[[ -n "${INPAINT_PROMPT:-}" ]] && args+=(--inpaint-prompt "${INPAINT_PROMPT}")
[[ -n "${INPAINT_DILATION:-}" ]] && args+=(--inpaint-dilation "${INPAINT_DILATION}")
[[ -n "${INPAINT_EXTRA_MASK:-}" ]] && args+=(--inpaint-extra-mask "${INPAINT_EXTRA_MASK}")
[[ -n "${BACKGROUND_VIDEO:-}" ]] && args+=(--background-video "${BACKGROUND_VIDEO}")
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && args+=(--cuda-visible-devices "${CUDA_VISIBLE_DEVICES}")
[[ "${SKIP_EXISTING:-0}" == "1" ]] && args+=(--skip-existing)

exec "${PY_AUTO}" "${args[@]}" "$@"
