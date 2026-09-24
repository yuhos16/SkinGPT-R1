#!/usr/bin/env bash
# SkinGPT-R1 SFT launcher.
set -euo pipefail
PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PACKAGE_ROOT"

export PYTHONPATH="$PACKAGE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export SKINGPT_MODEL_DIR="${SKINGPT_MODEL_DIR:-$PACKAGE_ROOT/external/base_model}"
export SKINGPT_DATASET_DIR="${SKINGPT_DATASET_DIR:-$PACKAGE_ROOT/data}"
export SKINGPT_IMAGE_ROOT="${SKINGPT_IMAGE_ROOT:-$PACKAGE_ROOT/external/images}"
export SKINGPT_FEATURE_ROOT="${SKINGPT_FEATURE_ROOT:-$PACKAGE_ROOT/external/teacher_features}"
export SKINGPT_OUTPUT_DIR="${SKINGPT_OUTPUT_DIR:-$PACKAGE_ROOT/outputs/sft}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export DDP_STATIC_GRAPH="${DDP_STATIC_GRAPH:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Default weights for the four-term training objective.
export SFT_WEIGHT="${SFT_WEIGHT:-1.0}"
export DISTILL_WEIGHT="${DISTILL_WEIGHT:-0.1}"
export MOE_AUX_WEIGHT="${MOE_AUX_WEIGHT:-0.001}"
export SKIN_LOSS_WEIGHT="${SKIN_LOSS_WEIGHT:-0.1}"

PYTHON_BIN="${SKINGPT_PYTHON:-python}"
WORKERS="${NPROC_PER_NODE:-8}"
COMMAND=("$PYTHON_BIN" -m torch.distributed.run --standalone
         --nproc_per_node="$WORKERS" "$PACKAGE_ROOT/src/train.py"
         --cfg "$PACKAGE_ROOT/configs/sft.yaml")

if [[ "${1:-}" == "--dry-run" ]]; then
    shift
    printf 'Command:'
    printf ' %q' "${COMMAND[@]}" "$@"
    printf '\nExternal data: %s\nImages: %s\nTeacher features: %s\nModel: %s\nOutput: %s\n' \
        "$SKINGPT_DATASET_DIR" "$SKINGPT_IMAGE_ROOT" "$SKINGPT_FEATURE_ROOT" \
        "$SKINGPT_MODEL_DIR" "$SKINGPT_OUTPUT_DIR"
    exit 0
fi

"$PYTHON_BIN" "$PACKAGE_ROOT/scripts/check_inputs.py"
exec "${COMMAND[@]}" "$@"
