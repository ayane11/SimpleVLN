#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PIPELINE_PYTHON:-/root/miniforge3/envs/simplevln-server/bin/python}" "$HERE/run_full.py" "$@"
