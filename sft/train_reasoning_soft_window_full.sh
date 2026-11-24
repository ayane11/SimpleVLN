#!/usr/bin/env bash
# End-to-end launcher for full normal-ratio (no downsampling) R2R+RxR AwareVLN SFT
# with 5-frame trapezoidal soft-window reasoning sampling and steps_ago memory prompts.
#
# Usage (inside dyj_vln / simplevln-sft conda env):
#   bash sft/train_reasoning_soft_window_full.sh prepare
#   bash sft/train_reasoning_soft_window_full.sh cache
#   bash sft/train_reasoning_soft_window_full.sh train
#   bash sft/train_reasoning_soft_window_full.sh all
set -eo pipefail

if [[ -f /root/miniforge3/etc/profile.d/conda.sh ]]; then
    source /root/miniforge3/etc/profile.d/conda.sh
    conda activate "${SFT_CONDA_ENV:-simplevln-sft}"
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export MODE=reasoning_soft_window
export SOFT_WINDOW_PRE_FRAMES="${SOFT_WINDOW_PRE_FRAMES:-1}"
export DIRECT_ACTION_KEEP_PROB="${DIRECT_ACTION_KEEP_PROB:-1.0}"
export STOP_REPEAT="${STOP_REPEAT:-2}"
export EMPTY_MEMORY_DROP_PROB="${EMPTY_MEMORY_DROP_PROB:-0.1}"

export WORK_DIR="${WORK_DIR:-$ROOT/outputs/sft/reasoning_rl_4b_v3_soft_window}"
export JSON_DIR="${JSON_DIR:-$WORK_DIR/annotations}"
export CACHE_DIR="${CACHE_DIR:-/nas/simplevln-reasoning-rl4b-v3-soft-window-cache}"
export OUTPUT_DIR="${OUTPUT_DIR:-$WORK_DIR/checkpoints}"
export MODEL="${MODEL:-$ROOT/checkpoints/SimpleVLN_Qwen35_4B_RL}"
export COT_INDEX="${COT_INDEX:-$ROOT/outputs/reasoning/full_qwen35_27b/exports/cot_index.json}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export MASTER_PORT="${MASTER_PORT:-29785}"

case "${1:-all}" in
    prepare)
        bash sft/reasoning_sft.sh prepare
        ;;
    cache)
        mkdir -p "$CACHE_DIR/hf"
        export HF_DATASETS_CACHE="$CACHE_DIR/hf" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
        bash sft/reasoning_sft.sh cache
        ;;
    train)
        bash sft/reasoning_sft.sh train
        ;;
    all)
        bash sft/reasoning_sft.sh prepare
        mkdir -p "$CACHE_DIR/hf"
        export HF_DATASETS_CACHE="$CACHE_DIR/hf" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
        bash sft/reasoning_sft.sh cache
        bash sft/reasoning_sft.sh train
        ;;
    *)
        echo "Usage: bash sft/train_reasoning_soft_window_full.sh {prepare|cache|train|all}" >&2
        exit 2
        ;;
esac
