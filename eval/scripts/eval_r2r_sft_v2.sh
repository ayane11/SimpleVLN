#!/usr/bin/env bash
set -euo pipefail
scripts="/workspace/VLN/code/SimpleVLN/eval/scripts"
runs="/workspace/VLN/runs"
work="/workspace/VLN/runs/outputs/sft/reasoning_rl_4b_v2"
ckpt="${CKPT_PATH:-/workspace/VLN/runs/outputs/sft/reasoning_rl_4b_v2/sft_full_lr5e6/checkpoints/v0-20261124-004839/checkpoint-467}"
[[ -d "$ckpt" ]] || { echo "Missing checkpoint: $ckpt" >&2; exit 1; }
name="sft_full_lr5e6_$(basename "$(realpath "$ckpt")")"
port="${BASE_PORT:-22014}"
case "${1:-}" in
    serve)
        model_dir="$runs/simplevln-runtime/cache/vllm-models/$name"
        base="$runs/checkpoints/SimpleVLN_Qwen35_4B_RL"
        mkdir -p "$model_dir"
        for file in "$ckpt"/*.safetensors "$ckpt"/*.json "$ckpt"/chat_template.jinja; do
            [[ ${file##*/} == tokenizer_config.json ]] && continue
            ln -sfn "$(realpath "$file")" "$model_dir/"
        done
        for file in tokenizer_config.json video_preprocessor_config.json; do
            [[ -f "$base/$file" ]] && ln -sfn "$base/$file" "$model_dir/"
        done
        export MODEL_PATH="$model_dir" MODEL_NAME="$name" BASE_PORT="$port" CUDA_VISIBLE_DEVICES="${SERVER_GPU:-1}"
        exec bash "$scripts/serve_vllm.sh" 1 1 ;;
    probe)
        exec /root/miniforge3/envs/simplevln-eval/bin/python "$work/probe_ckpt.py" --base-url "http://127.0.0.1:$port/v1" --model "$name" ;;
    eval)
        export BENCHMARK=r2r REASONING=1 PASS_K=1 EXP_NAME="$name"
        export CUDA_VISIBLE_DEVICES="${EVAL_GPU:-5}" SIM_GPU_DEVICE_ID=0
        export VLLM_BASE_URL="http://127.0.0.1:$port/v1" VLLM_MODEL_NAME="$name"
        exec bash "$scripts/eval.sh" ;;
    *) echo "Usage: bash $0 {serve|probe|eval}" >&2; exit 2 ;;
esac
