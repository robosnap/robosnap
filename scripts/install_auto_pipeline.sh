#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROBOSNAP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONDA_BIN="${CONDA_EXE:-conda}"
SAM3_ENV="${SAM3_ENV:-robosnap-sam3}"
ASSET_ENV="${ASSET_ENV:-robosnap-asset}"
LYRA_ENV="${LYRA_ENV:-robosnap-lyra}"
SIM_ENV="${SIM_ENV:-robosnap-sim}"
SAM3_PYTHON="${SAM3_PYTHON:-3.12}"
ASSET_PYTHON="${ASSET_PYTHON:-3.11}"
LYRA_PYTHON="${LYRA_PYTHON:-3.10}"
SIM_PYTHON="${SIM_PYTHON:-3.10}"
TORCH_CU121_INDEX="${TORCH_CU121_INDEX:-https://download.pytorch.org/whl/cu121}"
TORCH_CU126_INDEX="${TORCH_CU126_INDEX:-https://download.pytorch.org/whl/cu126}"
TORCH_CU128_INDEX="${TORCH_CU128_INDEX:-https://download.pytorch.org/whl/cu128}"
PIP_EXTRA_INDEX_URLS="${PIP_EXTRA_INDEX_URLS:-https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121}"
KAOLIN_FIND_LINKS="${KAOLIN_FIND_LINKS:-https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html}"
MAX_JOBS="${MAX_JOBS:-4}"
YES="${YES:-0}"
INSTALL_SAM3=1
INSTALL_ASSET=1
INSTALL_LYRA=1
INSTALL_SIM=1
SETUP_SOURCES=1
WRITE_ENV=1
FORCE_ENV=0
DOWNLOAD_CORE=0
DOWNLOAD_LYRA=0
ACCEPT_LYRA_LICENSE=0
COPY_CHECKPOINTS=0

usage() {
  cat <<EOF
Usage: bash scripts/install_auto_pipeline.sh [options]

Create the four Conda runtimes for the automatic layered-scene pipeline and
write configs/auto_pipeline.env.

Options:
  -y, --yes                 Do not prompt before installation.
  --skip-sam3               Skip the SAM3 runtime.
  --skip-asset              Skip the SAM3D/VGGT/alignment runtime.
  --skip-lyra               Skip the Lyra runtime.
  --skip-sim                Skip the sim-ready runtime.
  --skip-sources            Do not initialize pinned source submodules.
  --no-write-env            Do not write configs/auto_pipeline.env.
  --force-env               Replace configs/auto_pipeline.env.
  --download-checkpoints    Download SAM3, gated SAM3D, and VGGT models.
  --download-lyra           Download the approximately 91 GB Lyra-2 bundle.
  --accept-lyra-license     Confirm acceptance of the NVIDIA Lyra-2 model license.
  --copy-checkpoints        Copy model trees instead of linking the HF cache.
  --sam3-env NAME           Conda env name. Default: ${SAM3_ENV}
  --asset-env NAME          Conda env name. Default: ${ASSET_ENV}
  --lyra-env NAME           Conda env name. Default: ${LYRA_ENV}
  --sim-env NAME            Conda env name. Default: ${SIM_ENV}
  -h, --help                Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -y|--yes) YES=1 ;;
    --skip-sam3) INSTALL_SAM3=0 ;;
    --skip-asset) INSTALL_ASSET=0 ;;
    --skip-lyra) INSTALL_LYRA=0 ;;
    --skip-sim) INSTALL_SIM=0 ;;
    --skip-sources) SETUP_SOURCES=0 ;;
    --no-write-env) WRITE_ENV=0 ;;
    --force-env) FORCE_ENV=1 ;;
    --download-checkpoints) DOWNLOAD_CORE=1 ;;
    --download-lyra) DOWNLOAD_LYRA=1 ;;
    --accept-lyra-license) ACCEPT_LYRA_LICENSE=1 ;;
    --copy-checkpoints) COPY_CHECKPOINTS=1 ;;
    --sam3-env) SAM3_ENV="$2"; shift ;;
    --asset-env) ASSET_ENV="$2"; shift ;;
    --lyra-env) LYRA_ENV="$2"; shift ;;
    --sim-env) SIM_ENV="$2"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

log() {
  printf '[auto-install] %s\n' "$*"
}

run() {
  printf '[auto-install]'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

confirm() {
  if [[ "${YES}" == "1" ]]; then
    return 0
  fi
  read -r -p "Create/update the four automatic-pipeline Conda envs? [y/N] " reply
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
  if env_exists "${env_name}"; then
    log "conda env exists: ${env_name}"
  else
    run "${CONDA_BIN}" create -y -n "${env_name}" "python=${py_version}" pip
  fi
}

conda_prefix() {
  local env_name="$1"
  "${CONDA_BIN}" run -n "${env_name}" python -c 'import sys; print(sys.prefix)'
}

site_packages() {
  local env_name="$1"
  "${CONDA_BIN}" run -n "${env_name}" python -c 'import site; print(site.getsitepackages()[0])'
}

pip_install() {
  local env_name="$1"
  shift
  run "${CONDA_BIN}" run -n "${env_name}" python -m pip install "$@"
}

build_pip_install() {
  local env_name="$1"
  shift
  local prefix site
  prefix="$(conda_prefix "${env_name}")"
  site="$(site_packages "${env_name}")"
  local cpath="${prefix}/include:${site}/nvidia/cudnn/include:${site}/nvidia/nccl/include"
  local ldpath="${prefix}/lib:${prefix}/lib64:${site}/torch/lib:${site}/nvidia/cuda_runtime/lib:${site}/nvidia/cudnn/lib:${site}/nvidia/nccl/lib"
  run env \
    CUDA_HOME="${prefix}" \
    CC="${prefix}/bin/x86_64-conda-linux-gnu-gcc" \
    CXX="${prefix}/bin/x86_64-conda-linux-gnu-g++" \
    CPATH="${cpath}${CPATH:+:${CPATH}}" \
    LD_LIBRARY_PATH="${ldpath}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    MAX_JOBS="${MAX_JOBS}" \
    PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URLS}" \
    "${CONDA_BIN}" run -n "${env_name}" python -m pip install "$@"
}

install_sam3() {
  create_env "${SAM3_ENV}" "${SAM3_PYTHON}"
  pip_install "${SAM3_ENV}" --upgrade pip setuptools wheel
  pip_install "${SAM3_ENV}" --index-url "${TORCH_CU126_INDEX}" torch==2.7.0 torchvision==0.22.0
  pip_install "${SAM3_ENV}" opencv-python==4.11.0.86
  pip_install "${SAM3_ENV}" -e "${ROOT}" --no-deps
  pip_install "${SAM3_ENV}" -e "${ROOT}/third_party/sam3"
}

install_asset() {
  create_env "${ASSET_ENV}" "${ASSET_PYTHON}"
  run "${CONDA_BIN}" install -y -n "${ASSET_ENV}" -c conda-forge \
    cmake ninja packaging libgl ffmpeg gcc=12.4.0 gxx=12.4.0 eigen zlib
  run "${CONDA_BIN}" install -y -n "${ASSET_ENV}" -c nvidia/label/cuda-12.1.1 cuda-toolkit=12.1.1
  pip_install "${ASSET_ENV}" --upgrade pip setuptools wheel packaging ninja
  pip_install "${ASSET_ENV}" --index-url "${TORCH_CU121_INDEX}" torch==2.5.1 torchvision==0.20.1
  build_pip_install "${ASSET_ENV}" flash_attn==2.8.3 --no-build-isolation
  build_pip_install "${ASSET_ENV}" -e "${ROOT}/third_party/sam-3d-objects"
  build_pip_install "${ASSET_ENV}" -e "${ROOT}/third_party/sam-3d-objects[p3d]"
  run env \
    CUDA_HOME="$(conda_prefix "${ASSET_ENV}")" \
    MAX_JOBS="${MAX_JOBS}" \
    PIP_FIND_LINKS="${KAOLIN_FIND_LINKS}" \
    PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URLS}" \
    "${CONDA_BIN}" run -n "${ASSET_ENV}" python -m pip install \
    -e "${ROOT}/third_party/sam-3d-objects[inference]"
  run "${CONDA_BIN}" run -n "${ASSET_ENV}" python "${ROOT}/third_party/sam-3d-objects/patching/hydra"
  pip_install "${ASSET_ENV}" \
    numpy==1.26.4 scipy==1.16.3 open3d==0.18.0 trimesh==4.10.1 \
    opencv-python==4.9.0.80 Pillow==11.3.0 scikit-image==0.23.1 \
    plyfile==1.1.3 pyrender==0.1.45 PyOpenGL==3.1.0 \
    transformers==4.36.0 einops==0.8.1 safetensors==0.7.0
  pip_install "${ASSET_ENV}" -e "${ROOT}/third_party/vggt" --no-deps
  pip_install "${ASSET_ENV}" -e "${ROOT}" --no-deps
}

install_lyra() {
  create_env "${LYRA_ENV}" "${LYRA_PYTHON}"
  run "${CONDA_BIN}" install -y -n "${LYRA_ENV}" -c conda-forge \
    cmake ninja libgl ffmpeg packaging
  run env CONDA_BACKUP_CXX= "${CONDA_BIN}" install -y -n "${LYRA_ENV}" -c conda-forge \
    gcc=13.3.0 gxx=13.3.0 eigen zlib
  run "${CONDA_BIN}" install -y -n "${LYRA_ENV}" -c nvidia/label/cuda-12.8.0 cuda
  pip_install "${LYRA_ENV}" --upgrade pip setuptools wheel
  pip_install "${LYRA_ENV}" --index-url "${TORCH_CU128_INDEX}" torch==2.7.1 torchvision==0.22.1
  build_pip_install "${LYRA_ENV}" --no-deps -r "${ROOT}/third_party/lyra/Lyra-2/requirements.txt"
  build_pip_install "${LYRA_ENV}" "git+https://github.com/microsoft/MoGe.git@07444410f1e33f402353b99d6ccd26bd31e469e8"
  build_pip_install "${LYRA_ENV}" "transformer_engine[pytorch]==2.4.0" --no-build-isolation
  local site
  site="$(site_packages "${LYRA_ENV}")"
  run ln -sfn "${site}/nvidia/cuda_runtime" "${site}/nvidia/cudart"
  build_pip_install "${LYRA_ENV}" --no-build-isolation --no-binary :all: flash-attn==2.6.3
  run env USE_SYSTEM_EIGEN=1 \
    CUDA_HOME="$(conda_prefix "${LYRA_ENV}")" \
    MAX_JOBS="${MAX_JOBS}" \
    "${CONDA_BIN}" run -n "${LYRA_ENV}" python -m pip install --no-build-isolation \
    -e "${ROOT}/third_party/lyra/Lyra-2/lyra_2/_src/inference/vipe"
  build_pip_install "${LYRA_ENV}" --no-build-isolation \
    -e "${ROOT}/third_party/lyra/Lyra-2/lyra_2/_src/inference/depth_anything_3[gs]"
  pip_install "${LYRA_ENV}" -e "${ROOT}" --no-deps
}

install_sim() {
  create_env "${SIM_ENV}" "${SIM_PYTHON}"
  pip_install "${SIM_ENV}" --upgrade pip setuptools wheel
  pip_install "${SIM_ENV}" --index-url "${TORCH_CU121_INDEX}" torch==2.4.0 torchvision==0.19.0
  pip_install "${SIM_ENV}" \
    manifold3d==3.4.0 numpy==1.26.4 open3d==0.19.0 \
    opencv-python==4.11.0.86 rtree==1.4.1 sapien==2.2.2 \
    scikit-image==0.25.2 scipy==1.15.3 tqdm==4.67.1 \
    trimesh==4.11.4 vhacdx==0.0.10
  pip_install "${SIM_ENV}" -e "${ROOT}" --no-deps
}

verify_envs() {
  if [[ "${INSTALL_SAM3}" == "1" ]]; then
    run "${CONDA_BIN}" run -n "${SAM3_ENV}" python -c \
      "import cv2, numpy, sam3, torch; print('sam3', torch.__version__)"
  fi
  if [[ "${INSTALL_ASSET}" == "1" ]]; then
    run env LIDRA_SKIP_INIT=1 VGGT_DIR="${ROOT}/third_party/vggt" "${CONDA_BIN}" run -n "${ASSET_ENV}" python -c \
      "import gsplat, open3d, plyfile, sam3d_objects, torch, trimesh, vggt; print('asset', torch.__version__)"
  fi
  if [[ "${INSTALL_LYRA}" == "1" ]]; then
    run env PYTHONPATH="${ROOT}/third_party/lyra/Lyra-2" "${CONDA_BIN}" run -n "${LYRA_ENV}" python -c \
      "import depth_anything_3.api, flash_attn, moge.model.v1, torch, transformer_engine.pytorch, vipe_ext; print('lyra', torch.__version__)"
  fi
  if [[ "${INSTALL_SIM}" == "1" ]]; then
    run "${CONDA_BIN}" run -n "${SIM_ENV}" python -c \
      "import open3d, sapien.core, torch, trimesh, vhacdx; print('sim', torch.__version__)"
  fi
}

python_for_env() {
  local env_name="$1"
  if env_exists "${env_name}"; then
    "${CONDA_BIN}" run -n "${env_name}" python -c 'import sys; print(sys.executable)'
  else
    printf 'python\n'
  fi
}

prefix_for_env() {
  local env_name="$1"
  if env_exists "${env_name}"; then
    conda_prefix "${env_name}"
  fi
}

shell_quote() {
  printf '%q' "$1"
}

write_auto_env() {
  local env_file="${ROOT}/configs/auto_pipeline.env"
  if [[ "${WRITE_ENV}" != "1" ]]; then
    return 0
  fi
  if [[ -f "${env_file}" && "${FORCE_ENV}" != "1" ]]; then
    log "keep existing ${env_file}; pass --force-env to replace it"
    return 0
  fi

  local py_sam3 py_asset py_lyra py_sim lyra_prefix
  py_sam3="$(python_for_env "${SAM3_ENV}")"
  py_asset="$(python_for_env "${ASSET_ENV}")"
  py_lyra="$(python_for_env "${LYRA_ENV}")"
  py_sim="$(python_for_env "${SIM_ENV}")"
  lyra_prefix="$(prefix_for_env "${LYRA_ENV}")"
  log "write ${env_file}"

  {
    printf 'ROBOSNAP_ROOT=%s\n' "$(shell_quote "${ROOT}")"
    printf 'CHECKPOINT_DIR=${ROBOSNAP_ROOT}/checkpoints\n'
    printf 'HF_HOME=${CHECKPOINT_DIR}/hf_cache\n'
    printf 'TORCH_HOME=${CHECKPOINT_DIR}/torch_cache\n'
    printf 'HF_HUB_DISABLE_XET=1\n'
    printf 'ROBOSNAP_KEEP_PROXY=1\n\n'
    printf 'PY_AUTO=%s\n' "$(shell_quote "${py_asset}")"
    printf 'PY_SAM3=%s\n' "$(shell_quote "${py_sam3}")"
    printf 'PY_ASSET=%s\n' "$(shell_quote "${py_asset}")"
    printf 'PY_VGGT=%s\n' "$(shell_quote "${py_asset}")"
    printf 'PY_ALIGN=%s\n' "$(shell_quote "${py_asset}")"
    printf 'PY_LYRA=%s\n' "$(shell_quote "${py_lyra}")"
    printf 'PY_LYRA_CONDA_PREFIX=%s\n' "$(shell_quote "${lyra_prefix}")"
    printf 'PY_SIM_READY=%s\n\n' "$(shell_quote "${py_sim}")"
    printf 'SAM3_DIR=${ROBOSNAP_ROOT}/third_party/sam3\n'
    printf 'SAM3_CKPT=${CHECKPOINT_DIR}/sam3/sam3.pt\n'
    printf 'SAM3D_DIR=${ROBOSNAP_ROOT}/third_party/sam-3d-objects\n'
    printf 'SAM3D_CONFIG=${CHECKPOINT_DIR}/sam-3d-objects/pipeline.yaml\n'
    printf 'VGGT_DIR=${ROBOSNAP_ROOT}/third_party/vggt\n'
    printf 'VGGT_CKPT=\n'
    printf 'LYRA_DIR=${ROBOSNAP_ROOT}/third_party/lyra\n'
    printf 'LYRA2_CHECKPOINT_ROOT=${CHECKPOINT_DIR}/lyra2/checkpoints\n'
    printf 'LYRA_CHECKPOINT_DIR=${CHECKPOINT_DIR}/lyra\n'
    printf 'LYRA_DA3_MODEL_PATH=${LYRA2_CHECKPOINT_ROOT}/recon/model.pt\n\n'
    printf 'LYRA_RESOLUTION=352,624\nLYRA_USE_DMD=1\nLYRA_OFFLOAD=1\n'
    printf 'LYRA_RECON_DA3_MAX_FRAMES=32\nROBOSNAP_MAX_JOBS=1\n\n'
    printf 'SF_REAL2SIM_USE_CACHED_COLLISIONS=1\n'
    printf '\n'
    printf 'INPUT_IMAGE=${INPUT_IMAGE:-${ROBOSNAP_ROOT}/examples/test1.png}\n'
    printf 'OUTPUT_DIR=${OUTPUT_DIR:-${ROBOSNAP_ROOT}/outputs/automatic}\n'
    printf 'DEVICE=${DEVICE:-cuda:0}\nCUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}\nINPAINT_DILATION=${INPAINT_DILATION:-7}\n\n'
    printf '# Set OBJECT_FILE to bypass automatic object discovery.\n'
    printf '# OBJECT_FILE=/path/to/object.txt\n'
    printf 'GEMINI_TEXT_MODEL=gemini-3.5-flash\n'
    printf 'VLM_COMMAND="%s ${ROBOSNAP_ROOT}/scripts/run_gemini_vlm.py --image {image} --prompt {prompt} --output {output_json}"\n' "${py_asset}"
    printf 'INPAINT_PROMPT=${ROBOSNAP_ROOT}/configs/prompts/background_inpaint.txt\n'
    printf 'GEMINI_IMAGE_MODEL=gemini-3.1-flash-image\n'
    printf 'INPAINT_COMMAND="%s ${ROBOSNAP_ROOT}/scripts/run_gemini_image_edit.py --image {image} --mask {mask} --prompt {prompt} --output {output} --status {status}"\n' "${py_asset}"
  } > "${env_file}"
}

download_models() {
  if [[ "${DOWNLOAD_LYRA}" == "1" && "${ACCEPT_LYRA_LICENSE}" != "1" ]]; then
    echo "--download-lyra requires --accept-lyra-license." >&2
    exit 2
  fi
  if [[ "${DOWNLOAD_CORE}" != "1" && "${DOWNLOAD_LYRA}" != "1" ]]; then
    return 0
  fi
  local python_cmd
  python_cmd="$(python_for_env "${ASSET_ENV}")"
  if [[ "${python_cmd}" == "python" ]]; then
    python_cmd="$(python_for_env "${LYRA_ENV}")"
  fi
  local args=()
  [[ "${DOWNLOAD_CORE}" == "1" ]] && args+=(--core)
  if [[ "${DOWNLOAD_LYRA}" == "1" ]]; then
    args+=(--lyra --accept-lyra-license)
  fi
  [[ "${COPY_CHECKPOINTS}" == "1" ]] && args+=(--copy)
  run "${python_cmd}" "${ROOT}/scripts/download_auto_checkpoints.py" "${args[@]}"
}

if ! command -v "${CONDA_BIN}" >/dev/null 2>&1; then
  echo "conda not found. Set CONDA_EXE=/path/to/conda." >&2
  exit 127
fi

cd "${ROOT}"
log "repo root: ${ROOT}"
log "conda: ${CONDA_BIN}"
confirm

if [[ "${SETUP_SOURCES}" == "1" ]]; then
  source_args=()
  [[ "${INSTALL_SAM3}" != "1" ]] && source_args+=(--skip-sam3)
  [[ "${INSTALL_ASSET}" != "1" ]] && source_args+=(--skip-sam3d --skip-vggt)
  [[ "${INSTALL_LYRA}" != "1" ]] && source_args+=(--skip-lyra)
  run bash "${ROOT}/scripts/setup_auto_sources.sh" "${source_args[@]}"
fi

[[ "${INSTALL_SAM3}" == "1" ]] && install_sam3
[[ "${INSTALL_ASSET}" == "1" ]] && install_asset
[[ "${INSTALL_LYRA}" == "1" ]] && install_lyra
[[ "${INSTALL_SIM}" == "1" ]] && install_sim

verify_envs
write_auto_env
download_models

log "done"
log "run: export GEMINI_API_KEY=<your-api-key>; bash scripts/run_auto_pipeline.sh"
