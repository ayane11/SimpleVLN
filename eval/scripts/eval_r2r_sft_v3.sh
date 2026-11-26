#!/usr/bin/env bash
# Evaluate v3 AwareVLN soft-window SFT checkpoints (or the pre-RL base SimpleVLN_Qwen35_4B_SFT).
# Trained checkpoints are read directly from /workspace/VLN/runs/outputs/sft/reasoning_sft_4b_v3/checkpoints.
#
# Usage (inside dyj_vln):
#   SERVER_GPU=2 bash /workspace/VLN/code/SimpleVLN/eval/scripts/eval_r2r_sft_v3.sh serve
#   bash /workspace/VLN/code/SimpleVLN/eval/scripts/eval_r2r_sft_v3.sh probe
#   EVAL_GPU=0 bash /workspace/VLN/code/SimpleVLN/eval/scripts/eval_r2r_sft_v3.sh eval
#
# Optional overrides:
#   CKPT_PATH=/workspace/VLN/runs/outputs/sft/reasoning_sft_4b_v3/checkpoints/v0-.../checkpoint-400
#   EVAL_BASE_SFT=1  (serve/eval the pre-RL base SimpleVLN_Qwen35_4B_SFT without --reasoning)
set -euo pipefail

SCRIPTS_DIR="/workspace/VLN/code/SimpleVLN/eval/scripts"
RUNS_ROOT="/workspace/VLN/runs"
BASE_SFT="/workspace/VLN/runs/checkpoints/SimpleVLN_Qwen35_4B_SFT"
V3_WORK="/workspace/VLN/runs/outputs/sft/reasoning_sft_4b_v3"
V3_CKPT_ROOT="${V3_WORK}/checkpoints"
PORT="${BASE_PORT:-22015}"

if [[ "${EVAL_BASE_SFT:-0}" == "1" ]]; then
    ckpt="$BASE_SFT"
    name="qwen35_4b_sft_base"
    reasoning_flag=0
else
    if [[ -n "${CKPT_PATH:-}" ]]; then
        ckpt="$CKPT_PATH"
    else
        latest_link=$(ls -dt "${V3_CKPT_ROOT}"/*/last-checkpoint 2>/dev/null | head -n 1 || true)
        if [[ -n "$latest_link" ]]; then
            ckpt=$(readlink -f "$latest_link")
        else
            ckpt=$(ls -dt "${V3_CKPT_ROOT}"/*/checkpoint-* 2>/dev/null | head -n 1 || true)
        fi
    fi
    if [[ -z "$ckpt" || ! -d "$ckpt" ]]; then
        echo "No v3 checkpoint found under ${V3_CKPT_ROOT} (set CKPT_PATH=/abs/path/to/checkpoint)" >&2
        exit 1
    fi
    name="reasoning_sft_4b_v3_$(basename "$(realpath "$ckpt")")"
    reasoning_flag="${REASONING:-1}"
fi

case "${1:-}" in
    serve)
        if [[ "${EVAL_BASE_SFT:-0}" == "1" ]]; then
            model_dir="$ckpt"
        else
            # Keep vLLM Transformers-4 config shim in runtime cache, never in runs/checkpoints.
            model_dir="${RUNS_ROOT}/simplevln-runtime/cache/vllm-models/${name}"
            mkdir -p "$model_dir"
            for file in "$ckpt"/*.safetensors "$ckpt"/*.json "$ckpt"/chat_template.jinja; do
                [[ -e "$file" ]] || continue
                [[ ${file##*/} == tokenizer_config.json ]] && continue
                ln -sfn "$(realpath "$file")" "$model_dir/"
            done
            for file in tokenizer_config.json video_preprocessor_config.json; do
                if [[ -f "${BASE_SFT}/${file}" ]]; then
                    ln -sfn "${BASE_SFT}/${file}" "$model_dir/"
                elif [[ -f "${RUNS_ROOT}/checkpoints/SimpleVLN_Qwen35_4B_RL/${file}" ]]; then
                    ln -sfn "${RUNS_ROOT}/checkpoints/SimpleVLN_Qwen35_4B_RL/${file}" "$model_dir/"
                fi
            done
        fi
        export MODEL_PATH="$model_dir" MODEL_NAME="$name" BASE_PORT="$PORT"
        export CUDA_VISIBLE_DEVICES="${SERVER_GPU:-2}"
        exec bash "${SCRIPTS_DIR}/serve_vllm.sh" 1 1
        ;;
    probe)
        exec /root/miniforge3/envs/simplevln-eval/bin/python \
            "${RUNS_ROOT}/outputs/sft/reasoning_rl_4b_v2/probe_ckpt.py" \
            --annotations "${V3_WORK}/annotations/SFT/R2R.json" \
            --base-url "http://127.0.0.1:${PORT}/v1" \
            --model "$name"
        ;;
    eval)
        export BENCHMARK="${BENCHMARK:-r2r}"
        export REASONING="$reasoning_flag"
        export PASS_K="${PASS_K:-1}"
        export EXP_NAME="$name"
        export CUDA_VISIBLE_DEVICES="${EVAL_GPU:-0}"
        export SIM_GPU_DEVICE_ID=0
        export VLLM_BASE_URL="http://127.0.0.1:${PORT}/v1"
        export VLLM_MODEL_NAME="$name"
        exec bash "${SCRIPTS_DIR}/eval.sh"
        ;;
    *)
        echo "Usage: bash $0 {serve|probe|eval}" >&2
        exit 2
        ;;
esac
