#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROBOSNAP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VGGT_DIR="${VGGT_DIR:-${ROOT}/third_party/vggt}"
LYRA_DIR="${LYRA_DIR:-${ROOT}/third_party/lyra}"
VGGT_URL="${VGGT_URL:-https://github.com/facebookresearch/vggt.git}"
LYRA_URL="${LYRA_URL:-https://github.com/nv-tlabs/lyra.git}"
VGGT_COMMIT="${VGGT_COMMIT:-a288dd0f14786c93483e45524328726ab7b1b4ce}"
LYRA_COMMIT="${LYRA_COMMIT:-87f79a52b81b366d1d4aa3a526aa12e54207c998}"
DRY_RUN="${DRY_RUN:-0}"
SETUP_VGGT=1
SETUP_LYRA=1

usage() {
  cat <<EOF
Usage: bash scripts/setup_auto_sources.sh [options]

Fetch the pinned VGGT and Lyra source trees used by the automatic pipeline.

Options:
  --dry-run       Print commands without executing them.
  --skip-vggt     Do not fetch VGGT.
  --skip-lyra     Do not fetch or patch Lyra.
  -h, --help      Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --skip-vggt) SETUP_VGGT=0 ;;
    --skip-lyra) SETUP_LYRA=0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

log() {
  printf '[sources] %s\n' "$*"
}

run() {
  printf '[sources]'
  printf ' %q' "$@"
  printf '\n'
  if [[ "${DRY_RUN}" != "1" ]]; then
    "$@"
  fi
}

fetch_source() {
  local name="$1"
  local url="$2"
  local commit="$3"
  local target="$4"
  local marker="$5"
  local recursive="$6"

  if [[ -e "${target}/${marker}" ]]; then
    if [[ -d "${target}/.git" ]]; then
      local actual
      actual="$(git -C "${target}" rev-parse HEAD)"
      if [[ "${actual}" != "${commit}" ]]; then
        echo "${name} exists at ${actual}; expected ${commit}: ${target}" >&2
        exit 1
      fi
      log "${name} is pinned at ${commit}"
    else
      log "using existing ${name} source snapshot: ${target}"
    fi
    return 0
  fi
  if [[ -e "${target}" ]]; then
    echo "Refusing to replace incomplete source directory: ${target}" >&2
    exit 1
  fi

  run mkdir -p "$(dirname "${target}")"
  run git init "${target}"
  run git -C "${target}" remote add origin "${url}"
  run git -C "${target}" fetch --depth 1 origin "${commit}"
  run git -C "${target}" checkout --detach FETCH_HEAD
  if [[ "${recursive}" == "1" ]]; then
    run git -C "${target}" submodule update --init --recursive --depth 1
  fi
  if [[ "${DRY_RUN}" != "1" && ! -e "${target}/${marker}" ]]; then
    echo "${name} checkout is missing ${marker}: ${target}" >&2
    exit 1
  fi
}

apply_lyra_patch() {
  local lyra2="${LYRA_DIR}/Lyra-2"
  local patch_file="${ROOT}/third_party/patches/lyra2-4090-offload.patch"
  if [[ ! -f "${patch_file}" ]]; then
    echo "Missing Lyra compatibility patch: ${patch_file}" >&2
    exit 1
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    log "patch --batch --forward -p1 < ${patch_file} (cwd=${lyra2})"
    return 0
  fi
  if (cd "${lyra2}" && patch --dry-run --batch --forward -p1 < "${patch_file}" >/dev/null 2>&1); then
    log "applying Lyra 4090/offload compatibility patch"
    (cd "${lyra2}" && patch --batch --forward -p1 < "${patch_file}")
  elif (cd "${lyra2}" && patch --dry-run --batch --reverse -p1 < "${patch_file}" >/dev/null 2>&1); then
    log "Lyra compatibility patch is already applied"
  else
    echo "Lyra compatibility patch does not match ${lyra2}" >&2
    exit 1
  fi
}

if ! command -v git >/dev/null 2>&1; then
  echo "git is required." >&2
  exit 127
fi
if [[ "${SETUP_LYRA}" == "1" ]] && ! command -v patch >/dev/null 2>&1; then
  echo "patch is required for the Lyra compatibility patch." >&2
  exit 127
fi

if [[ "${SETUP_VGGT}" == "1" ]]; then
  fetch_source "VGGT" "${VGGT_URL}" "${VGGT_COMMIT}" "${VGGT_DIR}" "vggt/models/vggt.py" 0
fi
if [[ "${SETUP_LYRA}" == "1" ]]; then
  fetch_source "Lyra" "${LYRA_URL}" "${LYRA_COMMIT}" "${LYRA_DIR}" "Lyra-2/lyra_2/_src/inference/lyra2_zoomgs_inference.py" 1
  apply_lyra_patch
fi

log "done"
