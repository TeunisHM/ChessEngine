#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ -f venv/bin/activate ]; then
    source venv/bin/activate
fi

PGN1="lichess_elite_2025-07.pgn"
PGN2="lichess_elite_2025-08.pgn"
PASS1_OUT="pretrained_pass1.pt"
FINAL_OUT="simple2.pt"

echo "=== Pretraining pass 1: 3 epochs on ${PGN1} ==="
python pretrain_from_pgn.py \
    --pgn "${PGN1}" \
    --epochs 3 \
    --output-model "${PASS1_OUT}"

echo "=== Pretraining pass 2: 3 epochs on ${PGN2} (init from ${PASS1_OUT}) ==="
python pretrain_from_pgn.py \
    --pgn "${PGN2}" \
    --epochs 3 \
    --init-model "${PASS1_OUT}" \
    --output-model "${FINAL_OUT}"

echo "=== Starting RL training from ${FINAL_OUT} ==="
python train.py
