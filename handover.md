# Handover — ChessEngine PPO training

Last updated: 2026-07-25. Full per-run history lives in the auto-loaded memory
index: `~/.claude/projects/-var-home-eunis-Python-ChessEngine/memory/MEMORY.md`.

## Current state

**Baseline: `models/ppo_search_v19_checkpoint_449.pt`.** First and only
generation of the scaled architecture — **16 residual blocks, 192 filters, 1
transformer layer, 11.5M params (4.1x the old 2.77M net)** — fresh PGN
pretrain, then 450 batches of the search-in-training recipe. It beat the old
small-net ceiling decisively (+106 Elo raw / +227 Elo with search vs v16;
best engine score by far at 27.2%). Scaling the architecture, not tweaking the
loss, is what broke the ceiling.

**Continuations do not stack.** Two straight continuations of v19's recipe
have now failed to improve on it:
- **v17** (small-net era) — no gain over its init.
- **v20** (300 batches from v19@449, identical recipe, finished 2026-07-25) —
  H2H **tie** with search (40W/28D/32B, +28 Elo, CI [−30,+87]), and **worse
  vs the engine** (19.5% vs v19's 33.0%, a ~2.3σ regression). Verdict: v20 is
  not an improvement; **v19@449 stays the baseline**. Backup of v20's
  mid-run restart point kept as `ppo_search_v20_checkpoint_148.pt`.

**Next lever must be a change, not a continuation** — another architecture
size step (the move that worked for v19) or a recipe change, seeded fresh or
from v19@449.

## Evaluation protocol

- **Ranking H2H is run *with* search**, at the trained policy-scale form:
  `evaluate_vs_model.py --lookahead-k 4 --lookahead-alpha 1.0 --value-weight 1.0`
  (no `--raw`). Since v15 the models are *trained* under trainee-search, so
  that is the deployment condition. (The old raw-vs-raw rule targeted a
  now-retired α=0.33 value-dominant wrapper that flattened π and taxed every
  generation −95…−185 Elo; the α=1.0 form does not.)
- **Standalone engine eval** is the orthogonal second axis:
  `evaluate_vs_engine.py --engine-skill-level 0 --engine-move-time 0.01` with
  the same search flags. Keep skill 0 / 10ms fixed for comparability.
- **The two axes do not always agree** — v20 led slightly on H2H but regressed
  on engine. Don't promote on one alone.
- Report W/D/L, score, SE, and an Elo CI (be rigorous about the noise floor).
- The in-training vs-Stockfish counter runs well ahead of standalone matches
  (v19: 39.8% counter vs 27.2% standalone) — informative across runs, not a
  standalone ranking number. Cause partly unexplained (leading suspect:
  EnginePool 4-process concurrency), untested by user's choice.

## Runtime / infra

- Host: AMD Radeon 8060S (`gfx1151`), 125GB unified RAM. No NVIDIA.
- Use `venv-rocm/bin/python` for all training/eval (PyTorch 2.11 + ROCm 7.13);
  `venv` for general work.
- `MIOPEN_FIND_MODE=FAST` is required (packaged gfx1151 FindDb is unreadable);
  `train.py` and both `evaluate_vs_*.py` set it at import. AMP is disabled —
  FP16 produced NaN gradients; FP32 only.
- v19-scale timing: self-play batches ~15-20s, checkpoint/engine batches
  4-7 min, PPO update ~15-19s; full 450-batch run ~16h. Compute-bound, not
  memory-bound.
- No optimizer/scheduler state is saved in checkpoints (weights only), so any
  `--init-from` continuation resets AdamW momentum and restarts the cosine LR.

## Recipe (v19 / v20)

Search-in-training with the policy-scale formula, on the scaled net:
`--trainee-search --lookahead-alpha 1.0 --value-weight 1.0 --dtz-shaping-weight
0.15 --num-filters 192 --num-residual-blocks 16`, `tablebase_terminate_prob`
0.25. Trainee search applies in checkpoint + engine batches, never self-play.
`gamma=0.98`, `gae_lamb=0.95`, `entropy_weight=0.005`, `batch_size=32`,
`opening_prob=0.6`, `temperature=1.0`, `ppo_clip=0.2`, `target_kl=0.015`,
`opponent_ratio=0.45`, `engine_ratio=0.1`. AdamW lr=5e-5, cosine to 10%.

## Code state — load-bearing, don't change without reason

- **`lookahead.py`**: widened candidate set = top-k(π) ∪ captures ∪ checks;
  policy-scale score `log π − β·V`; quiescence handles in-check evasion.
  `select_moves_from_policy` stores `log_pi` at the actual sampling temperature.
- **`train.py`**: default entrypoint is pure PPO. CLI exposes `--trainee-search`,
  `--lookahead-alpha`/`--value-weight` (default 1.0/1.0), `--dtz-shaping-weight`
  (default 0, dense conversion-progress reward inside confirmed ≤5-man TB wins),
  and `--tb-value-aux-weight` (default 0, **dormant** — v18 tried it at 0.5 and
  regressed clearly; retry only at a much lower weight, e.g. 0.05–0.1, if ever).
  Per-batch PPO denominator gate uses search probs only when trainee search is
  on. `net_from_state_dict()` auto-detects dense vs conv heads, so all old
  checkpoints still load.
- Step penalty is load-bearing — don't remove it (see memory).

## Open questions

- **Conversion is still unsolved and TB adjudication hides it**: reaching a
  TB-won position can credit a win, and WDL has no distance-to-mate, so neither
  RL nor search gets a progress signal inside won endgames. DTZ shaping (weight
  0.15, unvalidated) and the dormant TB-aux loss are the candidates — but build
  a conversion testsuite first to isolate either.
- **Training-vs-standalone engine-score gap** (above) — don't trust the
  in-training counter for ranking.
- **Eval reproducibility**: `evaluate_vs_model.py` starts every game from the
  standard position with no seed or paired opening suite. Add a seed, fixed
  paired openings, and PGN output before the next close H2H.

## Models folder

`models/` holds all checkpoints; the opponent loader samples `.pt` files from
it dynamically. Current baseline: `ppo_search_v19_checkpoint_449.pt`.
