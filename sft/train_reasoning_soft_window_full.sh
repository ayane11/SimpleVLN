#!/usr/bin/env bash
# End-to-end launcher for v4 AwareVLN 5-frame soft-window SFT on SimpleVLN_Qwen35_4B_SFT.
#
# Usage (inside dyj_vln):
#   bash /workspace/VLN/code/SimpleVLN/sft/train_reasoning_soft_window_full.sh prepare
#   bash /workspace/VLN/code/SimpleVLN/sft/train_reasoning_soft_window_full.sh cache
#   bash /workspace/VLN/code/SimpleVLN/sft/train_reasoning_soft_window_full.sh train
#   bash /workspace/VLN/code/SimpleVLN/sft/train_reasoning_soft_window_full.sh all
set -eo pipefail

if [[ -f /root/miniforge3/etc/profile.d/conda.sh ]]; then
    source /root/miniforge3/etc/profile.d/conda.sh
    conda activate "${SFT_CONDA_ENV:-simplevln-sft}"
fi

export CODE_ROOT="${CODE_ROOT:-/workspace/VLN/code/SimpleVLN}"
export RUNS_ROOT="${RUNS_ROOT:-/workspace/VLN/runs}"
export DATA_ROOT="${DATA_ROOT:-/data/vln}"
cd "$CODE_ROOT"

export CACHE_NUM_PROC="${CACHE_NUM_PROC:-32}"
export MODE=reasoning_soft_window
export EMPTY_MEMORY_DROP_PROB="${EMPTY_MEMORY_DROP_PROB:-0.1}"

export WORK_DIR="${WORK_DIR:-${RUNS_ROOT}/outputs/sft/reasoning_sft_4b_v4}"
export JSON_DIR="${JSON_DIR:-${WORK_DIR}/annotations}"
export CACHE_DIR="${CACHE_DIR:-${RUNS_ROOT}/cache/simplevln-reasoning-sft4b-v4-cache}"
export OUTPUT_DIR="${OUTPUT_DIR:-${WORK_DIR}/checkpoints}"
export MODEL="${MODEL:-${RUNS_ROOT}/checkpoints/SimpleVLN_Qwen35_4B_SFT}"
export COT_INDEX="${COT_INDEX:-${RUNS_ROOT}/outputs/reasoning/full_qwen35_27b/exports/cot_index.json}"

export LR="${LR:-5e-6}"
export MIN_LR="${MIN_LR:-5e-7}"
export MAX_LENGTH="${MAX_LENGTH:-16384}"
export SAVE_STEPS="${SAVE_STEPS:-100}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-4}"
# Keep TP=2 on two GPUs; Megatron derives DP=1 from the world size.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export MASTER_PORT="${MASTER_PORT:-29785}"

case "${1:-all}" in
    prepare)
        bash "${CODE_ROOT}/sft/reasoning_sft.sh" prepare
        ;;
    cache)
        mkdir -p "${CACHE_DIR}/hf"
        export HF_DATASETS_CACHE="${CACHE_DIR}/hf" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
        bash "${CODE_ROOT}/sft/reasoning_sft.sh" cache
        ;;
    train)
        bash "${CODE_ROOT}/sft/reasoning_sft.sh" train
        ;;
    all)
        bash "${CODE_ROOT}/sft/reasoning_sft.sh" prepare
        mkdir -p "${CACHE_DIR}/hf"
        export HF_DATASETS_CACHE="${CACHE_DIR}/hf" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
        bash "${CODE_ROOT}/sft/reasoning_sft.sh" cache
        bash "${CODE_ROOT}/sft/reasoning_sft.sh" train
        ;;
    *)
        echo "Usage: bash ${CODE_ROOT}/sft/train_reasoning_soft_window_full.sh {prepare|cache|train|all}" >&2
        exit 2
        ;;
esac
