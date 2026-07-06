#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROBOSNAP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
ENV_FILE="${ROBOSNAP_ENV_FILE:-${ROOT}/configs/gui.env}"
MODE="auto"
INPUT_PATH=""
MASK_DIR=""
IMAGE_PATH=""
NUM_MASKS=""
DRY_RUN="${DRY_RUN:-0}"

usage() {
  cat <<USAGE
Usage: bash scripts/gui/bash/run_mask_to_assets.sh [options]

Run the RoboSnap mask-to-assets stage outside the interactive GUI.

Modes:
  multi   Run SAM3D run_inference.py on a multi_mask/ directory.
  single  Run SAM3D image2glb.py on image.png plus numbered masks in single_mask/.
  auto    Detect multi vs single from the input directory. Default.

Options:
  --mode auto|multi|single
  --input-path PATH      Input directory for auto/multi. Defaults to OUT_DIR from gui.env.
  --mask-dir PATH        single_mask directory for single mode.
  --image PATH           Image for single mode. Defaults to MASK_DIR/image.png.
  --num-masks N          Number of masks for image2glb.py. Auto-detected if omitted.
  --dry-run              Print the command without running it.
  -h, --help             Show this help.

Examples:
  bash scripts/gui/bash/run_mask_to_assets.sh --mode multi --input-path outputs/case/multi_mask
  bash scripts/gui/bash/run_mask_to_assets.sh --mode single --mask-dir outputs/case/single_mask
  DRY_RUN=1 bash scripts/gui/bash/run_mask_to_assets.sh --mode auto --input-path outputs/case/multi_mask
USAGE
}

log() {
  printf '[mask2asset] %s\n' "$*"
}

fail() {
  printf '[mask2asset] ERROR: %s\n' "$*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --input-path)
      INPUT_PATH="$2"
      shift 2
      ;;
    --mask-dir)
      MASK_DIR="$2"
      shift 2
      ;;
    --image)
      IMAGE_PATH="$2"
      shift 2
      ;;
    --num-masks)
      NUM_MASKS="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "unknown argument: $1"
      ;;
  esac
done

case "${MODE}" in
  auto|multi|single) ;;
  *) fail "--mode must be auto, multi, or single; got ${MODE}" ;;
esac

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

clear_socks_proxy() {
  if [[ "${ROBOSNAP_KEEP_PROXY:-0}" == "1" ]]; then
    return
  fi
  local proxy_var proxy_value
  for proxy_var in ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy; do
    proxy_value="${!proxy_var:-}"
    case "${proxy_value}" in
      socks*://*|SOCKS*://*) unset "${proxy_var}" ;;
    esac
  done
}

derive_conda_prefix() {
  local py="$1"
  if [[ "${py}" = /* && "${py}" == */bin/python* ]]; then
    dirname "$(dirname "${py}")"
  fi
}

abs_path() {
  local value="$1"
  if [[ -z "${value}" ]]; then
    return 0
  fi
  if [[ "${value}" = /* ]]; then
    printf '%s\n' "${value}"
  else
    printf '%s\n' "$(pwd)/${value}"
  fi
}

has_multi_mask_layout() {
  local dir="$1"
  find "${dir}" -mindepth 2 -maxdepth 2 -type d -name 'top*_mask' -print -quit 2>/dev/null | grep -q .
}

has_single_mask_layout() {
  local dir="$1"
  [[ -f "${dir}/image.png" ]] || return 1
  find "${dir}" -maxdepth 1 -type f \( -name '[0-9]*.png' -o -name '[0-9]*.glb' \) -print -quit 2>/dev/null | grep -q .
}

infer_num_masks() {
  local dir="$1"
  python3 - "${dir}" <<'PYCOUNT'
import re
import sys
from pathlib import Path
root = Path(sys.argv[1])
indices = set()
for path in root.iterdir() if root.exists() else []:
    if not path.is_file():
        continue
    match = re.fullmatch(r"(\d+)\.(png|glb)", path.name)
    if match:
        indices.add(int(match.group(1)))
if not indices:
    raise SystemExit("no numeric mask/glb files found")
expected = list(range(max(indices) + 1))
missing = [idx for idx in expected if idx not in indices]
if missing:
    raise SystemExit(f"non-contiguous mask indices: missing {missing[:10]}")
print(max(indices) + 1)
PYCOUNT
}

run_cmd() {
  log "$(printf '%q ' "$@")"
  if [[ "${DRY_RUN}" != "1" ]]; then
    "$@"
  fi
}

require_file_unless_dry_run() {
  local path="$1"
  local label="$2"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  [[ -f "${path}" ]] || fail "${label} does not exist: ${path}"
}

clear_socks_proxy

export ROBOSNAP_ROOT="${ROBOSNAP_ROOT:-${ROOT}}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ROBOSNAP_ROOT}/checkpoints}"
PY_ASSET="${PY_ASSET:-python}"
PY_ASSET_CONDA_PREFIX="${PY_ASSET_CONDA_PREFIX:-$(derive_conda_prefix "${PY_ASSET}")}"
SAM3D_DIR="${SAM3D_DIR:-${ROBOSNAP_ROOT}/third_party/sam-3d-objects}"
SAM3D_CONFIG="${SAM3D_CONFIG:-${CHECKPOINT_DIR}/sam-3d-objects/pipeline.yaml}"
HF_HOME="${HF_HOME:-${CHECKPOINT_DIR}/hf_cache}"
TORCH_HOME="${TORCH_HOME:-${CHECKPOINT_DIR}/torch_cache}"
export HF_HOME TORCH_HOME PY_ASSET_CONDA_PREFIX
export PYTHONPATH="${SAM3D_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -z "${INPUT_PATH}" && -n "${OUT_DIR:-}" ]]; then
  INPUT_PATH="${OUT_DIR}"
fi
if [[ -n "${INPUT_PATH}" ]]; then
  INPUT_PATH="$(abs_path "${INPUT_PATH}")"
fi
if [[ -n "${MASK_DIR}" ]]; then
  MASK_DIR="$(abs_path "${MASK_DIR}")"
fi
if [[ -n "${IMAGE_PATH}" ]]; then
  IMAGE_PATH="$(abs_path "${IMAGE_PATH}")"
fi

if [[ "${MODE}" == "auto" ]]; then
  probe="${INPUT_PATH:-${MASK_DIR}}"
  [[ -n "${probe}" ]] || fail "auto mode needs --input-path or --mask-dir"
  [[ -d "${probe}" ]] || fail "input directory does not exist: ${probe}"
  if has_multi_mask_layout "${probe}"; then
    MODE="multi"
    INPUT_PATH="${probe}"
  elif has_single_mask_layout "${probe}"; then
    MODE="single"
    MASK_DIR="${probe}"
  else
    fail "could not detect multi_mask or single_mask layout in ${probe}"
  fi
fi

case "${MODE}" in
  multi)
    [[ -n "${INPUT_PATH}" ]] || fail "multi mode requires --input-path"
    [[ -d "${INPUT_PATH}" ]] || fail "input path does not exist: ${INPUT_PATH}"
    require_file_unless_dry_run "${SAM3D_CONFIG}" "SAM3D_CONFIG"
    run_cmd env \
      CONDA_PREFIX="${PY_ASSET_CONDA_PREFIX}" \
      HF_HOME="${HF_HOME}" \
      TORCH_HOME="${TORCH_HOME}" \
      PYTHONPATH="${PYTHONPATH}" \
      "${PY_ASSET}" "${SAM3D_DIR}/sam3d_objects/run_inference.py" \
        --input_path "${INPUT_PATH}" \
        --compose_scene \
        --config "${SAM3D_CONFIG}"
    ;;
  single)
    if [[ -z "${MASK_DIR}" ]]; then
      [[ -n "${INPUT_PATH}" ]] || fail "single mode requires --mask-dir or --input-path"
      MASK_DIR="${INPUT_PATH}"
    fi
    [[ -d "${MASK_DIR}" ]] || fail "mask dir does not exist: ${MASK_DIR}"
    IMAGE_PATH="${IMAGE_PATH:-${MASK_DIR}/image.png}"
    [[ -f "${IMAGE_PATH}" ]] || fail "image does not exist: ${IMAGE_PATH}"
    require_file_unless_dry_run "${SAM3D_CONFIG}" "SAM3D_CONFIG"
    if [[ -z "${NUM_MASKS}" ]]; then
      NUM_MASKS="$(infer_num_masks "${MASK_DIR}")"
    fi
    run_cmd env \
      CONDA_PREFIX="${PY_ASSET_CONDA_PREFIX}" \
      HF_HOME="${HF_HOME}" \
      TORCH_HOME="${TORCH_HOME}" \
      PYTHONPATH="${PYTHONPATH}" \
      "${PY_ASSET}" "${SAM3D_DIR}/sam3d_objects/image2glb.py" \
        --config "${SAM3D_CONFIG}" \
        --image "${IMAGE_PATH}" \
        --mask_dir "${MASK_DIR}" \
        --num_masks "${NUM_MASKS}" \
        --seed 43 \
        --compose_scene
    ;;
esac
