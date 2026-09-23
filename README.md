# ChessEngine

Chess policy/value networks in PyTorch, with PPO and supervised training,
quiescence search, and a Gumbel search backend. Board rules and move generation
use `python-chess`.

The next experiment is **v50: Stockfish-supervised value learning from v26**,
with a budget of about 24 hours on the current machine. Stockfish, PGNs, and
Syzygy are permitted data sources. The experiment is planned, not implemented.
See [NEXT_TRAINING_PLAN.md](NEXT_TRAINING_PLAN.md).

## Project map

| Files | Purpose |
|---|---|
| `models.py`, `helper.py` | Network definitions, checkpoint loading, board/action encoding |
| `train.py` | PPO and search-target training, rollouts, progress evaluation |
| `pretrain_from_pgn.py`, `pretrain_from_puzzles.py`, `pretrain_from_tablebase.py` | Supervised training entry points |
| `train_wdl_head.py`, `finetune_clean_value.py` | Existing value-training utilities; limitations in [IMPROVEMENTS.md](IMPROVEMENTS.md) |
| `search_backends.py`, `lookahead.py`, `gumbel_search.py` | Search selection and implementations |
| `evaluate_vs_engine.py`, `evaluate_vs_model.py`, `conversion_suite.py` | Playing-strength and endgame evaluation |
| `search_strength_gate.py`, `diagnose_wdl.py`, `analyze_policy.py` | Diagnostics |
| `handover.md` | Verified baseline and interpretation of retained measurements |

Checkpoints, datasets, engine binaries, and run logs are local artifacts excluded
from Git. Old shell launchers have been removed; use the Python entry points.

## Environment and checks

Install `requirements.txt` in a virtual environment with a PyTorch build suitable
for the machine. The existing local ROCm environment is `venv-rocm`:

```bash
venv-rocm/bin/python -m unittest test_helper.py
venv-rocm/bin/python train.py --help
venv-rocm/bin/python pretrain_from_pgn.py --help
```

The helper tests cover encoding and legal masks; they do not validate the entire
training or search pipeline. Confirm GPU availability and benchmark throughput
before scheduling the experiment. Use FP32 for the planned run.

## Baseline evaluation

The initializer and control checkpoint is
`models/ppo_search_v26_checkpoint_299.pt`. With that local checkpoint and the
bundled Stockfish executable available:

```bash
venv-rocm/bin/python evaluate_vs_engine.py \
    --model models/ppo_search_v26_checkpoint_299.pt \
    --engine-path ./stockfish/stockfish \
    --games 128 --paired-openings \
    --engine-skill-level 0 --engine-move-time 0.01 \
    --search-backend quiescence --temperature 0 \
    --lookahead-k 4 --lookahead-alpha 1.0 \
    --max-qdepth 2 --check-budget 1
```

This is a screening run. Promotion requires the independent comparisons specified
in the experiment plan. `evaluate_vs_model.py --paired-openings` currently ignores
`--games` and plays its built-in book once per color.

At temperature zero, `--gumbel-deterministic-candidates` does not change Gumbel
search behavior. At positive temperature it selects candidates without Gumbel
noise and samples the played move from the completed-Q policy target.
