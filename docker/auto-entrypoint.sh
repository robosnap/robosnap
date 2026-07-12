#!/usr/bin/env bash
set -euo pipefail

cd "${ROBOSNAP_ROOT:-/workspace/robosnap}"

if [[ $# -eq 0 ]]; then
  set -- --help
fi

case "$1" in
  bash|sh|python|python3)
    exec "$@"
    ;;
  *)
    exec bash scripts/run_auto_pipeline.sh "$@"
    ;;
esac
