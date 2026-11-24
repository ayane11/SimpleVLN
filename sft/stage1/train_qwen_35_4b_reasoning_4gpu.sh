#!/usr/bin/env bash
# 4-GPU (RTX 4090 48GB) full-data Megatron SFT for AwareVLN-style Qwen3.5-4B reasoning.
# Uses TP=2 x DP=2, max_length=16384, global_batch_size=128, and TE fused cross-entropy.
set -eo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export MODEL="${MODEL:-$ROOT/checkpoints/SimpleVLN_Qwen35_4B_RL}"
export MODEL_TYPE="${MODEL_TYPE:-qwen3_5}"
export TEMPLATE="${TEMPLATE:-qwen3_5}"
export CACHE_DIR="${CACHE_DIR:-/nas/simplevln-reasoning-rl4b-v3-soft-window-cache}"
export OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/sft/reasoning_rl_4b_v3_soft_window/checkpoints}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-SimpleVLN}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-reasoning-soft-window-full-lr5e-6}"
export WANDB_DIR="${WANDB_DIR:-$(dirname "$OUTPUT_DIR")}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$CACHE_DIR/hf}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

export NNODES="${NNODES:-1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29785}"

# Wire RTX 4090 sm89 bf16 FlashAttn2 head_dim=256 compat & offline wandb deps if present.
compat_dir="${BF16_COMPAT_DIR:-$ROOT/outputs/sft/reasoning_rl_4b/bf16_4090_compat}"
runtime_deps_dir="${RUNTIME_DEPS_DIR:-$ROOT/outputs/sft/reasoning_rl_4b/runtime_deps}"
extra_py=""
if [[ -d "$compat_dir" ]]; then extra_py="$compat_dir"; fi
if [[ -d "$runtime_deps_dir" ]]; then extra_py="${extra_py:+$extra_py:}$runtime_deps_dir"; fi
if [[ -n "$extra_py" ]]; then
    export PYTHONPATH="$extra_py${PYTHONPATH:+:$PYTHONPATH}"
fi

mkdir -p "$OUTPUT_DIR" "$HF_DATASETS_CACHE"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "CACHE_DIR=${CACHE_DIR}"

precision_args=()
checkpoint_dtype=$(python -c 'import sys; from swift.model.model_meta import get_model_info_meta; m,_=get_model_info_meta(sys.argv[1], model_type="qwen3_5", download_model=False); print(m.torch_dtype)' "$MODEL" | tail -n 1)
if [[ "$checkpoint_dtype" == "torch.float32" ]]; then
    precision_args=(--torch_dtype bfloat16 --fp16 false --bf16 true)
fi
echo "Checkpoint dtype=$checkpoint_dtype  precision overrides=${precision_args[*]}"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NNODES="${NNODES}" \
NODE_RANK="${NODE_RANK}" \
MASTER_ADDR="${MASTER_ADDR}" \
MASTER_PORT="${MASTER_PORT}" \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
MAX_PIXELS="${MAX_PIXELS:-1003520}" \
megatron sft \
    "${precision_args[@]}" \
    --model "${MODEL}" \
    --model_type "${MODEL_TYPE}" --template "${TEMPLATE}" \
    --report_to wandb --wandb_project "${WANDB_PROJECT}" --wandb_exp_name "${WANDB_EXP_NAME}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT:-3}" --save_safetensors true \
    --cached_dataset "${CACHE_DIR}/SFT/R2R/train" "${CACHE_DIR}/SFT/RxR/train" \
    --tensor_model_parallel_size "${TP_SIZE:-2}" \
    --sequence_parallel true \
    --packing true \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --micro_batch_size "${MICRO_BATCH_SIZE:-1}" \
    --global_batch_size "${GLOBAL_BATCH_SIZE:-128}" \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --cross_entropy_loss_fusion true --cross_entropy_fusion_impl te \
    --overlap_grad_reduce true --overlap_param_gather true \
    --lr "${LR:-5e-6}" \
    --lr_warmup_fraction "${LR_WARMUP_FRACTION:-0.05}" \
    --min_lr "${MIN_LR:-5e-7}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS:-1}" \
    --output_dir "${OUTPUT_DIR}" \
    --logging_steps "${LOGGING_STEPS:-1}" \
    --save_steps "${SAVE_STEPS:-50}" \
    --max_length "${MAX_LENGTH:-16384}" \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-4}" \
    --no_save_optim true \
    --no_save_rng true \
    --dataset_num_proc 32
