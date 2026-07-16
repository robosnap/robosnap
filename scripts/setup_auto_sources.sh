#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROBOSNAP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SAM3_DIR="${SAM3_DIR:-${ROOT}/third_party/sam3}"
SAM3D_DIR="${SAM3D_DIR:-${ROOT}/third_party/sam-3d-objects}"
VGGT_DIR="${VGGT_DIR:-${ROOT}/third_party/vggt}"
LYRA_DIR="${LYRA_DIR:-${ROOT}/third_party/lyra}"
HUNYUAN_DIR="${HUNYUAN_DIR:-${ROOT}/third_party/Hunyuan3D-Part}"
SAM3_URL="${SAM3_URL:-https://github.com/robosnap/sam3.git}"
SAM3D_URL="${SAM3D_URL:-https://github.com/robosnap/sam-3d-objects.git}"
VGGT_URL="${VGGT_URL:-https://github.com/facebookresearch/vggt.git}"
LYRA_URL="${LYRA_URL:-https://github.com/robosnap/lyra.git}"
HUNYUAN_URL="${HUNYUAN_URL:-https://github.com/robosnap/Hunyuan3D-Part.git}"
SAM3_COMMIT="${SAM3_COMMIT:-16fff334254b7de76c2ae2fe8968fd85afc7d815}"
SAM3D_COMMIT="${SAM3D_COMMIT:-79dbb1f59adb7d4c4e16b1fe55ee38f52a1d12f0}"
VGGT_COMMIT="${VGGT_COMMIT:-44b3afbd1869d8bde4894dd8ea1e293112dd5eba}"
LYRA_COMMIT="${LYRA_COMMIT:-812d586ac7978b41c6dee560f99b07b1007e26fa}"
HUNYUAN_COMMIT="${HUNYUAN_COMMIT:-b58568a328202bde2921e7d7e01368c7f558ecb3}"
SETUP_SAM3=1
SETUP_SAM3D=1
SETUP_VGGT=1
SETUP_LYRA=1
SETUP_HUNYUAN=1

usage() {
  cat <<EOF
Usage: bash scripts/setup_auto_sources.sh [options]

Initialize the commit-pinned source trees used by RoboSnap.

Options:
  --skip-sam3     Do not initialize SAM3.
  --skip-sam3d    Do not initialize SAM-3D-Objects.
  --skip-vggt     Do not initialize VGGT.
  --skip-lyra     Do not initialize Lyra.
  --skip-hunyuan  Do not initialize Hunyuan3D-Part.
  -h, --help      Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-sam3) SETUP_SAM3=0 ;;
    --skip-sam3d) SETUP_SAM3D=0 ;;
    --skip-vggt) SETUP_VGGT=0 ;;
    --skip-lyra) SETUP_LYRA=0 ;;
    --skip-hunyuan) SETUP_HUNYUAN=0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

log() {
  printf "[sources] %s\n" "$*"
}

run() {
  printf "[sources]"
  printf " %q" "$@"
  printf "\n"
  "$@"
}

is_source_checkout() {
  local target="$1"
  [[ -e "${target}/.git" ]] || return 1
  [[ -z "$(git -C "${target}" rev-parse --show-prefix 2>/dev/null)" ]]
}

validate_source() {
  local name="$1"
  local commit="$2"
  local target="$3"
  local marker="$4"

  if [[ ! -e "${target}/${marker}" ]]; then
    echo "${name} checkout is missing ${marker}: ${target}" >&2
    exit 1
  fi
  if is_source_checkout "${target}"; then
    local actual
    actual="$(git -C "${target}" rev-parse HEAD)"
    if [[ "${actual}" != "${commit}" ]]; then
      echo "${name} is at ${actual}; expected ${commit}: ${target}" >&2
      exit 1
    fi
  fi
}

fetch_source() {
  local name="$1"
  local url="$2"
  local commit="$3"
  local target="$4"
  local recursive="$5"

  if [[ -e "${target}" ]]; then
    if [[ -d "${target}" && -z "$(find "${target}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
      rmdir "${target}"
    else
      echo "Refusing to replace incomplete source directory: ${target}" >&2
      exit 1
    fi
  fi
  run mkdir -p "$(dirname "${target}")"
  run git init "${target}"
  run git -C "${target}" remote add origin "${url}"
  run git -C "${target}" fetch --depth 1 origin "${commit}"
  run git -C "${target}" checkout --detach FETCH_HEAD
  if [[ "${recursive}" == "1" ]]; then
    run git -C "${target}" submodule update --init --recursive --depth 1
  fi
}

setup_source() {
  local name="$1"
  local relative="$2"
  local url="$3"
  local commit="$4"
  local marker="$5"
  local recursive="$6"
  local target
  case "${relative}" in
    third_party/sam3) target="${SAM3_DIR}" ;;
    third_party/sam-3d-objects) target="${SAM3D_DIR}" ;;
    third_party/vggt) target="${VGGT_DIR}" ;;
    third_party/lyra) target="${LYRA_DIR}" ;;
    third_party/Hunyuan3D-Part) target="${HUNYUAN_DIR}" ;;
    *) echo "Unknown source path: ${relative}" >&2; exit 2 ;;
  esac

  if [[ -e "${target}/${marker}" ]]; then
    validate_source "${name}" "${commit}" "${target}" "${marker}"
    log "${name} ready at ${commit}"
    return
  fi

  local module_path=""
  if [[ -f "${ROOT}/.gitmodules" ]]; then
    module_path="$(git -C "${ROOT}" config -f .gitmodules --get "submodule.${relative}.path" 2>/dev/null || true)"
  fi
  if [[ "${target}" == "${ROOT}/${relative}" ]] && git -C "${ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1 && [[ "${module_path}" == "${relative}" ]]; then
    run git -C "${ROOT}" submodule sync -- "${relative}"
    run git -C "${ROOT}" submodule update --init --recursive --depth 1 -- "${relative}"
  else
    fetch_source "${name}" "${url}" "${commit}" "${target}" "${recursive}"
  fi
  validate_source "${name}" "${commit}" "${target}" "${marker}"
  log "${name} ready at ${commit}"
}

if ! command -v git >/dev/null 2>&1; then
  echo "git is required." >&2
  exit 127
fi

[[ "${SETUP_SAM3}" == "1" ]] && setup_source "SAM3" "third_party/sam3" "${SAM3_URL}" "${SAM3_COMMIT}" "inference_image.py" 0
[[ "${SETUP_SAM3D}" == "1" ]] && setup_source "SAM-3D-Objects" "third_party/sam-3d-objects" "${SAM3D_URL}" "${SAM3D_COMMIT}" "sam3d_objects/image2glb.py" 0
[[ "${SETUP_VGGT}" == "1" ]] && setup_source "VGGT" "third_party/vggt" "${VGGT_URL}" "${VGGT_COMMIT}" "vggt/models/vggt.py" 0
[[ "${SETUP_LYRA}" == "1" ]] && setup_source "Lyra" "third_party/lyra" "${LYRA_URL}" "${LYRA_COMMIT}" "Lyra-2/lyra_2/_src/inference/lyra2_zoomgs_inference.py" 1
[[ "${SETUP_HUNYUAN}" == "1" ]] && setup_source "Hunyuan3D-Part" "third_party/Hunyuan3D-Part" "${HUNYUAN_URL}" "${HUNYUAN_COMMIT}" "P3-SAM/demo/app.py" 0

log "done"
