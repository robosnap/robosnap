#!/usr/bin/env bash
set -euo pipefail

# Detailed GUI launcher. Users should configure paths in configs/gui.env
# or environment variables, not by editing this file. The stable public command
# remains: bash scripts/run_gui.sh
#
# Configure input data, checkpoints, and optional runtime overrides in
# configs/gui.env or shell environment variables.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ENV_FILE="${ROBOSNAP_ENV_FILE:-${ROOT}/configs/gui.env}"
export ROBOSNAP_ROOT="${ROBOSNAP_ROOT:-${ROOT}}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ROBOSNAP_ROOT}/checkpoints}"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
else
  echo "No env file found at ${ENV_FILE}; using shell environment only." >&2
fi

export HF_HOME="${HF_HOME:-${CHECKPOINT_DIR}/hf_cache}"
export TORCH_HOME="${TORCH_HOME:-${CHECKPOINT_DIR}/torch_cache}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost"

if [[ "${ROBOSNAP_KEEP_PROXY:-0}" != "1" ]]; then
  for proxy_var in ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy; do
    proxy_value="${!proxy_var:-}"
    case "${proxy_value}" in
      socks*://*|SOCKS*://*)
        unset "${proxy_var}"
        ;;
    esac
  done
fi

derive_conda_prefix() {
  local py="$1"
  if [[ "${py}" = /* && "${py}" == */bin/python* ]]; then
    dirname "$(dirname "${py}")"
  fi
}

PY_SAM3="${PY_SAM3:-python}"
PY_ASSET="${PY_ASSET:-python}"
PY_ARTICULATE="${PY_ARTICULATE:-${PY_P3SAM:-python}}"
PY_P3SAM="${PY_P3SAM:-${PY_ARTICULATE}}"
export PY_ASSET_CONDA_PREFIX="${PY_ASSET_CONDA_PREFIX:-$(derive_conda_prefix "${PY_ASSET}")}"
export PY_ARTICULATE_CONDA_PREFIX="${PY_ARTICULATE_CONDA_PREFIX:-${PY_P3SAM_CONDA_PREFIX:-$(derive_conda_prefix "${PY_ARTICULATE}")}}"
export PY_P3SAM_CONDA_PREFIX="${PY_P3SAM_CONDA_PREFIX:-${PY_ARTICULATE_CONDA_PREFIX}}"
PORT="${PORT:-7897}"
MAX_FRAMES="${MAX_FRAMES:-20}"
DRY_RUN="${DRY_RUN:-0}"
DEFAULT_VIDEO="${ROBOSNAP_ROOT}/examples/video.mp4"
DEFAULT_OUT_DIR="${ROBOSNAP_ROOT}/outputs/example/multi_mask"
SAM3_DIR="${SAM3_DIR:-${ROBOSNAP_ROOT}/third_party/sam3}"
SAM3_CKPT="${SAM3_CKPT:-${CHECKPOINT_DIR}/sam3/sam3.pt}"
SAM3D_DIR="${SAM3D_DIR:-${ROBOSNAP_ROOT}/third_party/sam-3d-objects}"
SAM3D_CONFIG="${SAM3D_CONFIG:-${CHECKPOINT_DIR}/sam-3d-objects/pipeline.yaml}"
ARTICULATE_APP="${ARTICULATE_APP:-${P3SAM_APP:-${ROBOSNAP_ROOT}/third_party/Hunyuan3D-Part/P3-SAM/demo/app.py}}"
ARTICULATE_CKPT="${ARTICULATE_CKPT:-${P3SAM_CKPT:-${CHECKPOINT_DIR}/articulate/articulate.safetensors}}"
ARTICULATE_BASE_PORT="${ARTICULATE_BASE_PORT:-${P3SAM_BASE_PORT:-8180}}"
ARTICULATE_PUBLIC_URL_TEMPLATE="${ARTICULATE_PUBLIC_URL_TEMPLATE:-${P3SAM_PUBLIC_URL_TEMPLATE:-}}"

if [[ -z "${VIDEO:-}" ]]; then
  VIDEO="${DEFAULT_VIDEO}"
fi

if [[ -z "${OUT_DIR:-}" ]]; then
  if [[ "${VIDEO}" == "${DEFAULT_VIDEO}" ]]; then
    OUT_DIR="${DEFAULT_OUT_DIR}"
  else
    video_parent="$(dirname "${VIDEO}")"
    if [[ "$(basename "${video_parent}")" == "multi_mask" ]]; then
      OUT_DIR="${video_parent}"
    else
      OUT_DIR="${video_parent}/multi_mask"
    fi
  fi
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${OUT_DIR}"
fi

prepare_legacy_video_workspace() {
  local task_name base_dir source_video target_video
  task_name="$(basename "${OUT_DIR}")"
  if [[ "${task_name}" != "multi_mask" ]]; then
    return 0
  fi
  base_dir="$(dirname "${OUT_DIR}")"
  source_video="${base_dir}/video.mp4"
  target_video="${OUT_DIR}/video.mp4"

  # Preserve the original run/video2glb.sh workspace convention.
  if [[ -f "${base_dir}/video_wt_tag" && ! -f "${target_video}" && -f "${source_video}" ]]; then
    mv "${source_video}" "${target_video}"
    echo "Moved tagged input video to ${target_video}"
  fi

  if [[ "${VIDEO}" == "${source_video}" && -f "${target_video}" ]]; then
    VIDEO="${target_video}"
  fi
}

if [[ "${DRY_RUN}" != "1" ]]; then
  prepare_legacy_video_workspace
fi

if [[ ! -f "${VIDEO}" ]]; then
  echo "VIDEO does not exist: ${VIDEO}" >&2
  exit 2
fi

export PYTHONPATH="${SAM3_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

cmd=(
  "${PY_SAM3}" "${ROBOSNAP_ROOT}/robosnap/gui/RoboSnapGUI/inference_interactive_videoseg.py"
  --video "${VIDEO}"
  --out_dir "${OUT_DIR}"
  --ckpt "${SAM3_CKPT}"
  --port "${PORT}"
  --max_frames "${MAX_FRAMES}"
  --asset-python "${PY_ASSET}"
  --asset-dir "${SAM3D_DIR}"
  --asset-config "${SAM3D_CONFIG}"
  --articulate-python "${PY_ARTICULATE}"
  --articulate-app "${ARTICULATE_APP}"
  --articulate-ckpt "${ARTICULATE_CKPT}"
  --articulate-base-port "${ARTICULATE_BASE_PORT}"
)

if [[ -n "${ARTICULATE_PUBLIC_URL_TEMPLATE:-}" ]]; then
  cmd+=(--articulate-public-url-template "${ARTICULATE_PUBLIC_URL_TEMPLATE}")
fi
if [[ -n "${GRADIO_ALLOWED_ROOTS:-}" ]]; then
  IFS=":" read -r -a allowed_roots <<< "${GRADIO_ALLOWED_ROOTS}"
  for allowed_root in "${allowed_roots[@]}"; do
    if [[ -n "${allowed_root}" ]]; then
      cmd+=(--allowed-root "${allowed_root}")
    fi
  done
fi
if [[ "${SHARE:-0}" == "1" ]]; then
  cmd+=(--share)
fi
if [[ "${DEBUG:-0}" == "1" ]]; then
  cmd+=(--debug)
fi

echo "Starting RoboSnap GUI on port ${PORT}"
echo "Input video: ${VIDEO}"
echo "Output directory: ${OUT_DIR}"
if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'DRY RUN:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  exit 0
fi
exec "${cmd[@]}"
