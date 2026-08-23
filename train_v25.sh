#!/usr/bin/env bash
# v25: curriculum stage 4. Continue from v24@299 (champion, 50.5% vs skill-0
# Stockfish) with the curriculum teacher raised to Skill Level 1 — the single
# variable changed vs the v24 recipe.
#
# Teacher choice measured, not guessed: v24@299 scores 32.5% vs skill-1/10ms
# (healthy 30-40% curriculum gap) but 46.0% vs skill-0/50ms (more time is a
# dead knob at this level). The in-training progress eval stays pinned at the
# protocol skill 0/10ms for comparability.
set -euo pipefail

cd "$(dirname "$0")"

INIT="models/ppo_search_v24_checkpoint_299.pt"
MODEL_NAME="ppo_search_v25"
LOG="logs/v25_pipeline.log"

if [ ! -f "$INIT" ]; then
    echo "ERROR: init checkpoint not found: $INIT" >&2
    exit 1
fi

mkdir -p logs

echo "=== Launching v25 curriculum stage 4 from ${INIT} ==="
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
    --seed 3401 \
    > "${LOG}" 2>&1 &

echo "Started PID $! — tail with: tail -f ${LOG}"
