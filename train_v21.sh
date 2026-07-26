#!/usr/bin/env bash
# v21: clean-value RL. Warm-start from v19@449 (champion) with the validated
# search-in-training recipe PLUS a dense clean-value anchor (--clean-value-weight):
# the value head is supervised toward objective game outcome (mover POV,
# Syzygy-exact in the endgame) every batch, so it stays a calibrated evaluator
# and search remains useful. Value-only anchor — the policy is never imitated.
set -euo pipefail

cd "$(dirname "$0")"

INIT="models/ppo_search_v19_checkpoint_449.pt"
MODEL_NAME="ppo_search_v21"
LOG="logs/v21_pipeline.log"

if [ ! -f "$INIT" ]; then
    echo "ERROR: init checkpoint not found: $INIT" >&2
    exit 1
fi

mkdir -p logs

echo "=== Launching v21 clean-value RL from ${INIT} ==="
echo "Logging to ${LOG}"

nohup venv-rocm/bin/python train.py \
    --init-from "${INIT}" \
    --model-name "${MODEL_NAME}" \
    --num-batches 450 \
    --eval-interval 150 \
    --num-filters 192 \
    --num-residual-blocks 16 \
    --trainee-search \
    --lookahead-alpha 1.0 \
    --value-weight 1.0 \
    --dtz-shaping-weight 0.15 \
    --clean-value-weight 0.5 \
    --seed 1401 \
    > "${LOG}" 2>&1 &

echo "Started PID $! — tail with: tail -f ${LOG}"
