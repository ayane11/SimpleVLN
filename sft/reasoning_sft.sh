#!/usr/bin/env bash
# Prepare, cache, or train AwareVLN-style reasoning SFT inside dyj_vln (simplevln-sft env).
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-/workspace/VLN/code/SimpleVLN}"
RUNS_ROOT="${RUNS_ROOT:-/workspace/VLN/runs}"
DATA_ROOT="${DATA_ROOT:-/data/vln}"

cd "$CODE_ROOT"

mode="${MODE:-reasoning_soft_window}"
options=()
case "$mode" in
    reasoning_soft_window)
        EMPTY_MEMORY_DROP_PROB="${EMPTY_MEMORY_DROP_PROB:-0.1}"
        options+=(--soft-window --empty-memory-drop-prob "$EMPTY_MEMORY_DROP_PROB")
        ;;
    reasoning)
        ;;
    action_control)
        options+=(--action-control)
        ;;
    *)
        echo "MODE must be reasoning_soft_window, reasoning, or action_control" >&2
        exit 2
        ;;
esac

work="${WORK_DIR:-${RUNS_ROOT}/outputs/sft/reasoning_sft_4b_v4}"
export MODEL="${MODEL:-${RUNS_ROOT}/checkpoints/SimpleVLN_Qwen35_4B_SFT}"
export MODEL_TYPE="${MODEL_TYPE:-qwen3_5}"
export TEMPLATE="${TEMPLATE:-qwen3_5}"
export JSON_DIR="${JSON_DIR:-${work}/annotations}"
export CACHE_DIR="${CACHE_DIR:-${RUNS_ROOT}/cache/simplevln-reasoning-sft4b-v4-cache}"
export OUTPUT_DIR="${OUTPUT_DIR:-${work}/checkpoints}"
export COT_INDEX="${COT_INDEX:-${RUNS_ROOT}/outputs/reasoning/full_qwen35_27b/exports/cot_index.json}"

case "${1:-}" in
    prepare)
        for ds in r2r rxr; do
            name=R2R
            if [[ "$ds" == rxr ]]; then name=RxR; fi
            python "${CODE_ROOT}/DataProcess/tool/convert_vln.py" \
                --annotation_path "${DATA_ROOT}/annotations/simplevln/sft/${ds}/annotations.json" \
                --image_root "${DATA_ROOT}/images/${ds}" \
                --output_path "${JSON_DIR}/SFT/${name}.json" \
                --cot-index "${COT_INDEX}" \
                --dataset "$ds" "${options[@]}"
        done
        ;;
    cache)
        bash "${CODE_ROOT}/sft/cache/cache_dataset.sh" SFT/R2R SFT/RxR
        ;;
    train)
        export NNODES="${NNODES:-1}"
        export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
        export NODE_RANK="${NODE_RANK:-0}"
        export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
        if [[ "$mode" == "reasoning_soft_window" ]]; then
            bash "${CODE_ROOT}/sft/stage1/train_qwen_35_4b_reasoning_4gpu.sh"
        else
            bash "${CODE_ROOT}/sft/stage1/train_qwen_35_4b_stage1.sh"
        fi
        ;;
    *)
        echo "Usage: bash ${CODE_ROOT}/sft/reasoning_sft.sh {prepare|cache|train}" >&2
        exit 2
        ;;
esac
