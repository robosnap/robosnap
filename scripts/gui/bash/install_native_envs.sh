#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROBOSNAP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
CONDA_BIN="${CONDA_EXE:-conda}"
GUI_ENV="${GUI_ENV:-robosnap-gui}"
ASSET_ENV="${ASSET_ENV:-robosnap-asset}"
ARTICULATE_ENV="${ARTICULATE_ENV:-robosnap-articulate}"
GUI_PYTHON="${GUI_PYTHON:-3.12}"
ASSET_PYTHON="${ASSET_PYTHON:-3.11}"
ARTICULATE_PYTHON="${ARTICULATE_PYTHON:-3.10}"
TORCH_CU_INDEX="${TORCH_CU_INDEX:-https://download.pytorch.org/whl/cu121}"
PYG_WHEEL_INDEX="${PYG_WHEEL_INDEX:-https://data.pyg.org/whl/torch-2.4.0+cu121.html}"
EXTRA_INDEX_URLS="${EXTRA_INDEX_URLS:-https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121}"
DRY_RUN="${DRY_RUN:-0}"
YES="${YES:-0}"
INSTALL_GUI=1
INSTALL_ASSET=1
INSTALL_ARTICULATE=1
WRITE_ENV=1
FORCE_ENV=0

usage() {
  cat <<EOF
Usage: bash scripts/install_native_envs.sh [options]

Create the native three-env RoboSnap GUI setup and write configs/gui.env.

Options:
  --dry-run              Print commands without executing them.
  -y, --yes              Do not prompt before creating/installing envs.
  --skip-gui             Skip the GUI/video segmentation env.
  --skip-asset           Skip the mask-to-3D asset env.
  --skip-articulate      Skip the Articulate Tool env.
  --gui-env NAME         Conda env name for GUI runtime. Default: ${GUI_ENV}
  --asset-env NAME       Conda env name for asset runtime. Default: ${ASSET_ENV}
  --articulate-env NAME  Conda env name for Articulate runtime. Default: ${ARTICULATE_ENV}
  --gui-python VERSION   Python version for GUI env. Default: ${GUI_PYTHON}
  --asset-python VERSION Python version for asset env. Default: ${ASSET_PYTHON}
  --art-python VERSION   Python version for Articulate env. Default: ${ARTICULATE_PYTHON}
  --no-write-env         Do not write configs/gui.env after install.
  --force-env            Overwrite configs/gui.env if it already exists.
  -h, --help             Show this help.

Environment overrides:
  CONDA_EXE, TORCH_CU_INDEX, PYG_WHEEL_INDEX, EXTRA_INDEX_URLS, ROBOSNAP_ROOT
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    -y|--yes) YES=1 ;;
    --skip-gui) INSTALL_GUI=0 ;;
    --skip-asset) INSTALL_ASSET=0 ;;
    --skip-articulate) INSTALL_ARTICULATE=0 ;;
    --gui-env) GUI_ENV="$2"; shift ;;
    --asset-env) ASSET_ENV="$2"; shift ;;
    --articulate-env) ARTICULATE_ENV="$2"; shift ;;
    --gui-python) GUI_PYTHON="$2"; shift ;;
    --asset-python) ASSET_PYTHON="$2"; shift ;;
    --art-python|--articulate-python) ARTICULATE_PYTHON="$2"; shift ;;
    --no-write-env) WRITE_ENV=0 ;;
    --force-env) FORCE_ENV=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

log() {
  printf '[install] %s\n' "$*"
}

run() {
  printf '[install]'
  printf ' %q' "$@"
  printf '
'
  if [[ "${DRY_RUN}" != "1" ]]; then
    "$@"
  fi
}

confirm() {
  if [[ "${YES}" == "1" || "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  read -r -p "Create/install RoboSnap conda envs now? [y/N] " reply
  case "${reply}" in
    y|Y|yes|YES) ;;
    *) echo "Aborted." >&2; exit 1 ;;
  esac
}

env_exists() {
  "${CONDA_BIN}" run -n "$1" python --version >/dev/null 2>&1
}

create_env() {
  local env_name="$1"
  local py_version="$2"
  if [[ "${DRY_RUN}" == "1" ]]; then
    run "${CONDA_BIN}" create -y -n "${env_name}" "python=${py_version}"
    return 0
  fi
  if env_exists "${env_name}"; then
    log "conda env exists: ${env_name}"
  else
    run "${CONDA_BIN}" create -y -n "${env_name}" "python=${py_version}"
  fi
}

pip_install() {
  local env_name="$1"
  shift
  run "${CONDA_BIN}" run -n "${env_name}" python -m pip install "$@"
}

install_gui() {
  create_env "${GUI_ENV}" "${GUI_PYTHON}"
  pip_install "${GUI_ENV}" --upgrade pip setuptools wheel
  pip_install "${GUI_ENV}" --index-url "${TORCH_CU_INDEX}" torch torchvision
  pip_install "${GUI_ENV}" -e "${ROOT}"
  pip_install "${GUI_ENV}" -e "${ROOT}/third_party/sam3"
}

install_asset() {
  create_env "${ASSET_ENV}" "${ASSET_PYTHON}"
  pip_install "${ASSET_ENV}" --upgrade pip setuptools wheel packaging ninja
  pip_install "${ASSET_ENV}" --index-url "${TORCH_CU_INDEX}" torch==2.5.1 torchvision==0.20.1
  run env PIP_EXTRA_INDEX_URL="${EXTRA_INDEX_URLS}" "${CONDA_BIN}" run -n "${ASSET_ENV}" python -m pip install flash_attn==2.8.3 --no-build-isolation
  run env PIP_EXTRA_INDEX_URL="${EXTRA_INDEX_URLS}" "${CONDA_BIN}" run -n "${ASSET_ENV}" python -m pip install -e "${ROOT}/third_party/sam-3d-objects[inference,p3d]"
}

install_articulate() {
  create_env "${ARTICULATE_ENV}" "${ARTICULATE_PYTHON}"
  pip_install "${ARTICULATE_ENV}" --upgrade pip setuptools wheel ninja
  pip_install "${ARTICULATE_ENV}" --index-url "${TORCH_CU_INDEX}" torch==2.4.0 torchvision==0.19.0
  pip_install "${ARTICULATE_ENV}" spconv-cu121==2.3.8 torch-scatter -f "${PYG_WHEEL_INDEX}"
  pip_install "${ARTICULATE_ENV}" viser fpsample trimesh numba gradio scikit-learn websockets opencv-python pillow numpy scipy timm addict safetensors huggingface_hub tqdm einops omegaconf diffusers scikit-image pymeshlab easydict
  run bash -lc "cd '${ROOT}/third_party/Hunyuan3D-Part/P3-SAM/utils/chamfer3D' && '${CONDA_BIN}' run -n '${ARTICULATE_ENV}' python setup.py install"
}

conda_python_path() {
  local env_name="$1"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '/path/to/miniconda3/envs/%s/bin/python\n' "${env_name}"
  else
    "${CONDA_BIN}" run -n "${env_name}" python - <<'PY'
import sys
print(sys.executable)
PY
  fi
}

write_gui_env() {
  local env_file="${ROOT}/configs/gui.env"
  if [[ "${WRITE_ENV}" != "1" ]]; then
    return 0
  fi
  if [[ -f "${env_file}" && "${FORCE_ENV}" != "1" ]]; then
    log "keep existing ${env_file}; pass --force-env to overwrite"
    return 0
  fi
  local py_gui py_asset py_art
  py_gui="$(conda_python_path "${GUI_ENV}")"
  py_asset="$(conda_python_path "${ASSET_ENV}")"
  py_art="$(conda_python_path "${ARTICULATE_ENV}")"
  log "write ${env_file}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  cat > "${env_file}" <<EOF
PY_SAM3=${py_gui}
PY_ASSET=${py_asset}
PY_ARTICULATE=${py_art}
VIDEO=\${ROBOSNAP_ROOT}/examples/video.mp4
OUT_DIR=\${ROBOSNAP_ROOT}/outputs/example/multi_mask
PORT=7897
EOF
}

if ! command -v "${CONDA_BIN}" >/dev/null 2>&1; then
  if [[ "${DRY_RUN}" == "1" ]]; then
    log "conda not found; dry-run will still print planned commands"
  else
    echo "conda not found. Set CONDA_EXE=/path/to/conda or activate conda first." >&2
    exit 127
  fi
fi

cd "${ROOT}"
log "repo root: ${ROOT}"
log "conda: ${CONDA_BIN}"
confirm

if [[ "${INSTALL_GUI}" == "1" ]]; then
  install_gui
fi
if [[ "${INSTALL_ASSET}" == "1" ]]; then
  install_asset
fi
if [[ "${INSTALL_ARTICULATE}" == "1" ]]; then
  install_articulate
fi
write_gui_env

log "done"
log "start GUI with: bash scripts/run_gui.sh"
