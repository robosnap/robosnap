#!/usr/bin/env bash
#
# Materialize existing local checkpoints into the RoboSnap checkpoint layout.
#
# Example:
#
# LOCAL_SAM3_CKPT=/path/to/sam3.pt \
# LOCAL_SAM3D_CHECKPOINT_DIR=/path/to/sam3d/checkpoints \
# LOCAL_ARTICULATE_CKPT=/path/to/p3sam.safetensors \
# LOCAL_SONATA_CKPT=/path/to/sonata.pth \
# MATERIALIZE_MODE=symlink \
# bash scripts/gui/bash/copy_checkpoints_from_local.sh
#

set -euo pipefail

ROOT="${ROBOSNAP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ROOT}/checkpoints}"
LOCAL_HF_CACHE="${LOCAL_HF_CACHE:-}"
LOCAL_SAM3_CKPT="${LOCAL_SAM3_CKPT:-${LOCAL_HF_CACHE:+${LOCAL_HF_CACHE}/sam3/sam3.pt}}"
LOCAL_SAM3D_CHECKPOINT_DIR="${LOCAL_SAM3D_CHECKPOINT_DIR:-${LOCAL_HF_CACHE:+${LOCAL_HF_CACHE}/sam-3d-objects/checkpoints}}"
LOCAL_ARTICULATE_CKPT="${LOCAL_ARTICULATE_CKPT:-${LOCAL_P3SAM_CKPT:-}}"
LOCAL_SONATA_CKPT="${LOCAL_SONATA_CKPT:-}"
LOCAL_MOGE_CACHE="${LOCAL_MOGE_CACHE:-${LOCAL_HF_CACHE:+${LOCAL_HF_CACHE}/models--Ruicheng--moge-vitl}}"
MATERIALIZE_MODE="${MATERIALIZE_MODE:-copy}"

usage() {
  cat <<EOF
Usage: bash scripts/gui/bash/copy_checkpoints_from_local.sh

Materialize checkpoint files from a local cache into checkpoints/.
Set the required LOCAL_* paths explicitly, or set LOCAL_HF_CACHE when your cache
uses the default RoboSnap layout:
  LOCAL_SAM3_CKPT=${LOCAL_SAM3_CKPT}
  LOCAL_SAM3D_CHECKPOINT_DIR=${LOCAL_SAM3D_CHECKPOINT_DIR}
  LOCAL_ARTICULATE_CKPT=${LOCAL_ARTICULATE_CKPT}
  LOCAL_SONATA_CKPT=${LOCAL_SONATA_CKPT}
  LOCAL_MOGE_CACHE=${LOCAL_MOGE_CACHE}

Override CHECKPOINT_DIR to write to a different checkpoint layout.
MATERIALIZE_MODE=copy|symlink|hardlink controls whether files are copied,
symlinked, or hardlinked into the release layout.
EOF
}

case "${1:-}" in
  -h|--help)
    usage
    exit 0
    ;;
esac

log() {
  printf '[checkpoint] %s\n' "$*"
}

fail() {
  printf '[checkpoint] ERROR: %s\n' "$*" >&2
  exit 1
}

require_file() {
  local path="$1"
  [ -f "$path" ] || fail "missing file: $path"
}

require_dir() {
  local path="$1"
  [ -d "$path" ] || fail "missing directory: $path"
}

run_cmd() {
  log "$*"
  "$@"
}

copy_file() {
  local src="$1"
  local dst="$2"
  if [ -z "$src" ]; then
    fail "missing source for $dst; set the corresponding LOCAL_* variable"
  fi
  require_file "$src"
  run_cmd mkdir -p "$(dirname "$dst")"
  case "$MATERIALIZE_MODE" in
    copy)
      run_cmd cp -a --reflink=auto "$src" "$dst"
      ;;
    hardlink)
      ln -f "$src" "$dst" 2>/dev/null || cp -a "$src" "$dst"
      ;;
    symlink)
      run_cmd ln -sfn "$src" "$dst"
      ;;
    *)
      fail "MATERIALIZE_MODE must be copy, hardlink, or symlink; got ${MATERIALIZE_MODE}"
      ;;
  esac
}

copy_dir_contents() {
  local src="$1"
  local dst="$2"
  if [ -z "$src" ]; then
    fail "missing source directory for $dst; set the corresponding LOCAL_* variable"
  fi
  require_dir "$src"
  case "$MATERIALIZE_MODE" in
    copy)
      run_cmd mkdir -p "$dst"
      run_cmd cp -a --reflink=auto "$src/." "$dst/"
      ;;
    hardlink)
      run_cmd mkdir -p "$dst"
      cp -al "$src/." "$dst/" 2>/dev/null || cp -a "$src/." "$dst/"
      ;;
    symlink)
      run_cmd mkdir -p "$(dirname "$dst")"
      if [ -e "$dst" ] && [ ! -L "$dst" ]; then
        fail "cannot replace existing non-symlink directory in symlink mode: $dst"
      fi
      run_cmd ln -sfn "$src" "$dst"
      ;;
    *)
      fail "MATERIALIZE_MODE must be copy, hardlink, or symlink; got ${MATERIALIZE_MODE}"
      ;;
  esac
}

copy_optional_hf_model() {
  local src="$1"
  local dst="$2"
  local name="$3"
  if [ -z "$src" ]; then
    log "skip optional ${name}: LOCAL_MOGE_CACHE is not set"
    return 0
  fi
  if [ ! -d "$src" ]; then
    log "skip optional ${name}: not found at $src"
    return 0
  fi
  if find "$src" -name '*.incomplete' -print -quit | grep -q .; then
    if [ "${COPY_INCOMPLETE_HF_CACHE:-0}" != "1" ]; then
      log "skip optional ${name}: cache has .incomplete files; set COPY_INCOMPLETE_HF_CACHE=1 to copy anyway"
      return 0
    fi
  fi
  copy_dir_contents "$src" "$dst"
}

log "root: ${ROOT}"
log "checkpoint dir: ${CHECKPOINT_DIR}"
log "mode: ${MATERIALIZE_MODE}"

copy_file "$LOCAL_SAM3_CKPT" "${CHECKPOINT_DIR}/sam3/sam3.pt"
copy_dir_contents "$LOCAL_SAM3D_CHECKPOINT_DIR" "${CHECKPOINT_DIR}/sam-3d-objects"
copy_file "$LOCAL_ARTICULATE_CKPT" "${CHECKPOINT_DIR}/articulate/articulate.safetensors"
copy_file "$LOCAL_SONATA_CKPT" "${CHECKPOINT_DIR}/sonata/sonata.pth"

copy_optional_hf_model "$LOCAL_MOGE_CACHE" "${CHECKPOINT_DIR}/hf_cache/models--Ruicheng--moge-vitl" "MoGe depth model"

log "done"
log "update configs/gui.env if needed, then start with: bash scripts/run_gui.sh"
