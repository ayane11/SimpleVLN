# Source after setting RUNS_ROOT and OUTPUT_DIR.
export WANDB_MODE=offline
export WANDB_DIR="${WANDB_DIR:-$OUTPUT_DIR}"
mkdir -p "$WANDB_DIR"
wandb_deps="$RUNS_ROOT/outputs/sft/reasoning_rl_4b/runtime_deps"
if [[ -d "$wandb_deps/wandb" ]]; then
    export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$wandb_deps"
fi
