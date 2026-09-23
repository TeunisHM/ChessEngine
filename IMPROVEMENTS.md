# Verified implementation work

Updated 2026-09-23. These items describe current code behavior; their Elo impact
has not been measured. The selected experiment is in
[NEXT_TRAINING_PLAN.md](NEXT_TRAINING_PLAN.md).

## Required for v50

- **Teacher data and loss:** add Stockfish-labeled parent/child groups, soft WDL
  targets, sibling ranking, and splits by source game. Existing PGN/outcome
  trainers do not implement this experiment.
- **Evaluator selection:** expose the trained WDL evaluator to search as
  `P(win) - P(loss)` while preserving terminal handling. Current search calls
  use the scalar value. Use the same feature path as `models.py:forward` when
  caching features; `finetune_clean_value.py:_trunk` skips interleaved attention
  and calls `transformer` even when it can be `None`.
- **Independent openings:** extend `evaluate_vs_model.py` to accept a held-out
  opening suite. Its current paired mode ignores `--games` and uses only the
  built-in book. Repeating deterministic games does not add new observations.
- **Endgame reporting:** separate checkmates from `zeroing-win` successes in
  `conversion_suite.py`, with a fixed defender across candidate comparisons.

## Completed cleanup

File-reflection augmentation has been removed from `train.py`,
`pretrain_from_tablebase.py`, and `helper.py`. It did not preserve castling:
`e1g1` on `r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1` maps to illegal `d1b1`.
Keep it disabled in the new trainer. The playing-strength effect of removing it
has not been measured.

## Separate follow-up work

- `helper.py` encodes the fullmove number but not the halfmove clock. Positions
  with different fifty-move-rule status can have identical input tensors.
- PPO rollouts bootstrap genuine timeouts from zero. Review truncation targets
  separately from terminal outcomes.
- In `train.py`, self-play outcome counters classify some tablebase-adjudicated
  nonterminal boards as unfinished even though training labels exist.
- Gumbel's m=8/n=64 schedule reduces to one root candidate after 56 simulations,
  then spends eight more simulations on it. Review the allocation independently
  of evaluator training.
- The scalar head has an unbounded output while Gumbel uses terminal values
  ±1. Check value scale and terminal ordering when changing evaluators.

Keep search algorithm changes and shared-trunk adaptation out of the first value
comparison so its measured effect remains attributable to the trained evaluator.
