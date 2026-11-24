#!/usr/bin/env bash
# Run from a SimpleVLN checkout in the SFT environment.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
mode="${MODE:-reasoning}"
options=()
case "$mode" in
    reasoning)
        DROP_PRE_NODE_BUFFER="${DROP_PRE_NODE_BUFFER:-1}"
        DIRECT_ACTION_KEEP_PROB="${DIRECT_ACTION_KEEP_PROB:-0.35}"
        STOP_REPEAT="${STOP_REPEAT:-2}"
        if [[ "$DROP_PRE_NODE_BUFFER" == 1 ]]; then options+=(--drop-pre-node-buffer); fi
        options+=(--direct-action-keep-prob "$DIRECT_ACTION_KEEP_PROB" --stop-repeat "$STOP_REPEAT")
        ;;
    reasoning_soft_window)
        SOFT_WINDOW_PRE_FRAMES="${SOFT_WINDOW_PRE_FRAMES:-1}"
        DIRECT_ACTION_KEEP_PROB="${DIRECT_ACTION_KEEP_PROB:-1.0}"
        STOP_REPEAT="${STOP_REPEAT:-2}"
        EMPTY_MEMORY_DROP_PROB="${EMPTY_MEMORY_DROP_PROB:-0.1}"
        options+=(
            --soft-window
            --soft-window-pre-frames "$SOFT_WINDOW_PRE_FRAMES"
            --direct-action-keep-prob "$DIRECT_ACTION_KEEP_PROB"
            --stop-repeat "$STOP_REPEAT"
            --empty-memory-drop-prob "$EMPTY_MEMORY_DROP_PROB"
        )
        ;;
    action_control) options+=(--action-control) ;;
    *) echo "MODE must be reasoning, reasoning_soft_window, or action_control" >&2; exit 2 ;;
esac
work="${WORK_DIR:-$root/outputs/sft/$mode}"
export MODEL="${MODEL:-$root/checkpoints/SimpleVLN_Qwen35_4B_RL}"
export MODEL_TYPE=qwen3_5 TEMPLATE=qwen3_5
export JSON_DIR="${JSON_DIR:-$work/annotations}"
export CACHE_DIR="${CACHE_DIR:-$work/cache}"
export OUTPUT_DIR="${OUTPUT_DIR:-$work/checkpoints}"
case "${1:-}" in
    prepare)
        for ds in r2r rxr; do
            name=R2R
            if [[ "$ds" == rxr ]]; then name=RxR; fi
            python DataProcess/tool/convert_vln.py \
                --annotation_path "$root/data/annotations/simplevln/sft/$ds/annotations.json" \
                --image_root "$root/data/images/$ds" --output_path "$JSON_DIR/SFT/$name.json" \
                --cot-index "${COT_INDEX:-$root/outputs/reasoning/full_qwen35_27b/exports/cot_index.json}" \
                --dataset "$ds" "${options[@]}"
        done ;;
    cache) bash sft/cache/cache_dataset.sh SFT/R2R SFT/RxR ;;
    train)
        export NNODES="${NNODES:-1}" NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
        export NODE_RANK="${NODE_RANK:-0}" MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
        if [[ "$mode" == "reasoning_soft_window" ]]; then
            bash sft/stage1/train_qwen_35_4b_reasoning_4gpu.sh
        else
            bash sft/stage1/train_qwen_35_4b_stage1.sh
        fi ;;
    *) echo "Usage: bash sft/reasoning_sft.sh {prepare|cache|train}" >&2; exit 2 ;;
esac
