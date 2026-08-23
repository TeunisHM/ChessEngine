#!/usr/bin/env bash
# v24: curriculum stage 3. Continue from v23@299 (champion, 39% vs Stockfish)
# with the engine-curriculum weight raised 0.25 -> 0.35, testing whether the
# 31% -> 33.5% -> 39% trend keeps climbing under harder opposition.
#
# Rides along all session improvements: batched quiescence (~7x faster search
# batches), dead-tail rollout cutoff, Stockfish paired-opening progress evals,
# generation-weighted opponent sampling, and the PPO degenerate-row filter.
set -euo pipefail

cd "$(dirname "$0")"

INIT="models/ppo_search_v23_checkpoint_299.pt"
MODEL_NAME="ppo_search_v24"
LOG="logs/v24_pipeline.log"

if [ ! -f "$INIT" ]; then
    echo "ERROR: init checkpoint not found: $INIT" >&2
    exit 1
fi

mkdir -p logs

echo "=== Launching v24 curriculum stage 3 from ${INIT} ==="
echo "Logging to ${LOG}"

nohup env MIOPEN_FIND_MODE=FAST venv-rocm/bin/python train.py \
    --init-from "${INIT}" \
    --model-name "${MODEL_NAME}" \
    --num-batches 300 \
    --eval-interval 50 \
    --eval-games 32 \
    --engine-ratio 0.35 \
    --trainee-search \
    --lookahead-alpha 1.0 \
    --value-weight 1.0 \
    --dtz-shaping-weight 0.15 \
    --wdl-weight 1.0 \
    --seed 2401 \
    > "${LOG}" 2>&1 &

echo "Started PID $! — tail with: tail -f ${LOG}"
