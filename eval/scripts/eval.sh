#!/bin/bash
# Evaluate R2R (default), RxR, or both val-unseen benchmarks.
# Defaults: R2R GPU 5, RxR GPU 6; shared server on port 22001.
# Run inside dyj_vln; Python environments selected explicitly.
#
# Usage:
#   BENCHMARK=r2r EXP_NAME=my-run bash scripts/eval.sh
# Set BENCHMARK=rxr or BENCHMARK=both to select other benchmarks.

set -euo pipefail

# Local execution: direct connections and offline experiment tracking.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="localhost,127.0.0.1,::1"
export no_proxy="$NO_PROXY"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export WANDB_MODE=offline
SIMPLEVLN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export VLN_DATA_ROOT="${VLN_DATA_ROOT:-${SIMPLEVLN_ROOT}/eval/data/vln_data}"
export HF_HOME="${SIMPLEVLN_ROOT}/.cache/huggingface"
export XDG_CACHE_HOME="${SIMPLEVLN_ROOT}/.cache"
export TMPDIR="${SIMPLEVLN_ROOT}/.cache/tmp"
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$TMPDIR"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON="${PYTHON:-/root/miniforge3/envs/simplevln-eval/bin/python}"
BENCHMARK="${BENCHMARK:-r2r}"
# Physical GPU selection; Habitat uses logical device 0.
case "$BENCHMARK" in
    r2r|both) export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}" ;;
    rxr) export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}" ;;
esac
EXP_NAME="${EXP_NAME:-qwen35_4b_rl}"
RESULT_ROOT="${RESULT_ROOT:-./results}"

VLLM_HOST="${VLLM_HOST:-http://localhost}"
VLLM_PORT_START="${VLLM_PORT_START:-22001}"
VLLM_NUM_SERVERS="${VLLM_NUM_SERVERS:-1}"
VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-simplevln}"

SPLIT_NUM="${SPLIT_NUM:-2}"
PASS_K="${PASS_K:-1}"
FORWARD_DISTANCE="${FORWARD_DISTANCE:-25}"
TURN_ANGLE="${TURN_ANGLE:-15}"
MAX_ACTION_HISTORY="${MAX_ACTION_HISTORY:-200}"

run_one() {
    local cfg="$1"
    local out="$2"
    local server_args=(--vllm_host "${VLLM_HOST}" --vllm_port_start "${VLLM_PORT_START}" --vllm_num_servers "${VLLM_NUM_SERVERS}")
    if [[ -n "${VLLM_BASE_URL:-}" ]]; then
        server_args=(--vllm_base_url "${VLLM_BASE_URL}")
    fi
    if [[ "${REASONING:-0}" == 1 ]]; then
        server_args+=(--reasoning)
        out="${out}_reasoning"
    fi
    "${PYTHON}" vln/eval.py \
        --exp-config "${cfg}" \
        "${server_args[@]}" \
        --vllm_model_name "${VLLM_MODEL_NAME}" \
        --split-num "${SPLIT_NUM}" \
        --pass-k "${PASS_K}" \
        --result-path "${out}" \
        --forward-distance "${FORWARD_DISTANCE}" \
        --turn-angle "${TURN_ANGLE}" \
        --max-action-history "${MAX_ACTION_HISTORY}"
}

# Select one benchmark, or explicitly request both.
BENCHMARK="${BENCHMARK:-r2r}"
export SIM_GPU_DEVICE_ID="${SIM_GPU_DEVICE_ID:-0}"
case "$BENCHMARK" in
    r2r|rxr) run_one "config/vln_${BENCHMARK}.yaml" "${RESULT_ROOT}/${EXP_NAME}/${BENCHMARK}_pass${PASS_K}" ;;
    both)
        run_one config/vln_r2r.yaml "${RESULT_ROOT}/${EXP_NAME}/r2r_pass${PASS_K}"
        run_one config/vln_rxr.yaml "${RESULT_ROOT}/${EXP_NAME}/rxr_pass${PASS_K}"
        ;;
    *) echo "BENCHMARK must be r2r, rxr, or both" >&2; exit 2 ;;
esac
