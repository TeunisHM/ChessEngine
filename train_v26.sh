#!/usr/bin/env bash
# v26: continuation at skill 1. v25 (first run vs the skill-1 teacher) landed
# in the gray zone — 54.0% vs SF standalone (+3.5pp over v24, z~0.9) with the
# in-run trend ending on its two best readings. One 300-batch run may be too
# short against a meaningfully harder teacher; this continues from v25@299
# with the recipe unchanged.
set -euo pipefail

cd "$(dirname "$0")"

INIT="models/ppo_search_v25_checkpoint_299.pt"
MODEL_NAME="ppo_search_v26"
LOG="logs/v26_pipeline.log"

if [ ! -f "$INIT" ]; then
    echo "ERROR: init checkpoint not found: $INIT" >&2
    exit 1
fi

mkdir -p logs

echo "=== Launching v26 (skill-1 continuation) from ${INIT} ==="
echo "Logging to ${LOG}"

nohup env MIOPEN_FIND_MODE=FAST venv-rocm/bin/python train.py \
    --init-from "${INIT}" \
    --model-name "${MODEL_NAME}" \
    --num-batches 300 \
    --eval-interval 50 \
    --eval-games 32 \
    --engine-ratio 0.35 \
    --engine-skill-level 1 \
    --trainee-search \
    --lookahead-alpha 1.0 \
    --value-weight 1.0 \
    --dtz-shaping-weight 0.15 \
    --wdl-weight 1.0 \
    --seed 4401 \
    > "${LOG}" 2>&1 &

echo "Started PID $! — tail with: tail -f ${LOG}"
