#!/bin/bash
RUNS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)/runs"
set -euo pipefail

# Export already-converted training JSONs as ms-swift Arrow caches.
#
# Inputs : ${JSON_DIR}/<dataset>.json   (produced by DataProcess/tool/convert_vln.py
#          or convert_dagger.py; see README section 3)
# Outputs: ${CACHE_DIR}/<dataset>/train/
#
# Usage:
#   bash cache_dataset.sh                    # cache all five datasets
#   bash cache_dataset.sh SFT/R2R SFT/RxR    # cache selected datasets

MODEL="${MODEL:-${RUNS_ROOT}/checkpoints/Qwen3-VL-4B-Instruct}"
CACHE_DIR="${CACHE_DIR:-${RUNS_ROOT}/simplevln-runtime/cache/sft}"
JSON_DIR="${JSON_DIR:-${CACHE_DIR}/annotations}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

ALL_DATASETS=(SFT/R2R SFT/RxR SFT/ScaleVLN DAgger/R2R DAgger/RxR)
if [ $# -eq 0 ]; then
    DATASETS=("${ALL_DATASETS[@]}")
else
    DATASETS=("$@")
fi

model_args=()
if [[ -n "${MODEL_TYPE:-}" ]]; then model_args+=(--model_type "$MODEL_TYPE"); fi

if [[ -n "${TEMPLATE:-}" ]]; then model_args+=(--template "$TEMPLATE"); fi

for DATASET in "${DATASETS[@]}"; do
    case "${DATASET}" in
        SFT/R2R|SFT/RxR|SFT/ScaleVLN|DAgger/R2R|DAgger/RxR)
            ;;
        *)
            echo "Unknown dataset: ${DATASET}" >&2
            echo "Available: ${ALL_DATASETS[*]}" >&2
            exit 1
            ;;
    esac

    JSON="${JSON_DIR}/${DATASET}.json"
    if [ ! -f "${JSON}" ]; then
        echo "Missing converted JSON: ${JSON}" >&2
        echo "Run the corresponding converter first (see README section 3)." >&2
        exit 1
    fi

    echo "=========================================="
    echo "Caching dataset: ${DATASET}"
    echo "Input : ${JSON}"
    echo "Output: ${CACHE_DIR}/${DATASET}"
    echo "=========================================="

    MAX_PIXELS=1003520 \
    python "${SCRIPT_DIR}/fast_cache_dataset.py" \
        --model "${MODEL}" "${model_args[@]}" \
        --dataset "${JSON}" \
        --dataset_num_proc "${CACHE_NUM_PROC:-64}" \
        --to_cached_dataset true \
        --output_dir "${CACHE_DIR}/${DATASET}"

    echo "Done: ${DATASET}"
    echo ""
done

echo "All done! Cached datasets are in: ${CACHE_DIR}"
