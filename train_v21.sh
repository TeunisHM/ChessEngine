#!/usr/bin/env bash
# v21: WDL-head RL. Warm-start from v19@449 (champion) with the validated
# search-in-training recipe PLUS training the SEPARATE WDL head (--wdl-weight):
# cross-entropy toward objective game outcome, with the shared trunk DETACHED so
# it only updates wdl_head and never perturbs the policy/critic (unlike the
# earlier value-scalar anchor, which destabilised PPO). A calibrated evaluator
# for search, trained on the self-play distribution as a safe passenger.
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
    --wdl-weight 1.0 \
    --seed 1401 \
    > "${LOG}" 2>&1 &

echo "Started PID $! — tail with: tail -f ${LOG}"
