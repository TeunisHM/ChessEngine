# Handover — ChessEngine self-play RL

Last updated: 2026-07-28. Full per-run history: the auto-loaded memory index
`~/.claude/projects/-var-home-eunis-Python-ChessEngine/memory/MEMORY.md`.

## Current state

**Strongest model: `models/ppo_search_v23_checkpoint_299.pt`.** Best objective
score the project has posted — **39% vs Stockfish** (skill 0 / 10ms, search),
up from v19/v21's 31%. Also beats v21@449 by +24 Elo head-to-head.

Lineage this session (all 16×192 / 11.5M-param net):
- **v19@449** — scaled-architecture baseline (16 residual blocks, 192 filters,
  1 transformer layer). Scaling, not loss tweaks, broke the old small-net ceiling.
- **v21@449** — WDL-head run from v19@449. +50 Elo raw H2H over v19 (the bug-fixed
  search became a real +114-Elo improvement operator again), but **flat vs
  Stockfish (31%)** — the H2H gain did not transfer to absolute strength.
- **v23@299** — curriculum run from v21@449 (pruned at-level pool + engine-ratio
  0.25). **+8pp vs Stockfish → 39%**, confirmed identical at @149 and @299, and
  +24 Elo H2H over v21. Curriculum trend: **31% → 33.5% → 39%** (v21→v22→v23).

**Two proven absolute-strength levers: (1) harder curriculum (validated by v23),
(2) architecture scale-up (v19).** Self-play *continuation alone* plateaus on
absolute strength. Highest-EV next run: **combine them** — a bigger net trained
under the curriculum from the start.

## Key findings (this session)

- **H2H-vs-predecessor gains DON'T transfer to absolute strength.** v21 beat v19
  +50 H2H but scored identically vs Stockfish. **Track vs-Stockfish as the honest
  metric**, not just H2H vs the previous baseline. Only *scale* and *curriculum*
  have moved the objective number.
- **Search bug-fixes were worth +114 Elo** (v19-search vs v19-raw, 132 paired):
  mate scored a fixed `1.0` that `value_weight>1` could outrank (→ dominant
  `MATE_SCORE`, value_weight-independent); search board copies used `stack=False`
  → wrong repetition/50-move planes (→ `stack=True`). Also draws reward set to 0
  (was −0.1, which negates to +0.1 in the critic).
- **Separate WDL evaluator head** (Path A) built + trained during v21 with the
  **trunk detached** (policy-safe — unlike the earlier value-scalar anchor, which
  backprop'd into the trunk, blew up KL, and was scrapped). Calibrates to ~78%
  WDL but does **not** clearly beat the value head in search (+161 vs +138 over
  v19-raw, within noise). Validated as safe/trainable; not yet earning deployment.
- **MCTS is off the table** on this iGPU — latency-bound sequential inference,
  can't batch, infeasible for training. **Stockfish distillation rejected** by
  user ("cheating"). Path: self-play + the net's own (now-stronger) search as the
  improvement operator + curriculum + scale. Stockfish-as-*opponent* in RL is fine
  (a sparring partner, win/lose signal — not move-imitation).

## Evaluation protocol

- **vs-Stockfish is the honest absolute-strength metric** — `evaluate_vs_engine.py
  --engine-skill-level 0 --engine-move-time 0.01`, search `k=4/α=1.0/vw=1.0`,
  100+ games; keep skill 0 / 10ms fixed for comparability. H2H vs the previous
  baseline is *necessary but insufficient* (can rise while absolute strength is flat).
- **H2H**: `evaluate_vs_model.py`, use `--paired-openings` (every book line ×
  both colors = 132 games, low-noise) and `--use-wdl` to search via the WDL head.
- Report W/D/L, score, SE, Elo CI. Be rigorous about the noise floor.
- The in-training vs-random eval is saturated (100/0/0) — a collapse tripwire
  only. Replacing it is on the TODO.

## Runtime / infra

- Host: AMD Ryzen AI Max+ 395, Radeon 8060S iGPU (`gfx1151`), 128GB unified RAM.
  No NVIDIA. iGPU ~37 TFLOPS practical; CPU (16 Zen5) ~5 TFLOPS — the GPU is the
  accelerator, but it's throughput-bound (bad at latency-bound sequential work).
- Use `venv-rocm/bin/python` for training/eval (PyTorch 2.11 + ROCm 7.13);
  `venv` for general work.
- `MIOPEN_FIND_MODE=FAST` is required (packaged gfx1151 FindDb unreadable);
  set at import in `train.py` and both `evaluate_vs_*.py`. AMP off (FP16 → NaN
  grads); FP32 only.
- Timing: self-play batches ~15-20s, checkpoint/engine batches 2-7 min, PPO
  update ~15-19s. 450-batch run ~16h. Compute-bound.
- Checkpoints store weights only — any `--init-from` continuation resets AdamW
  momentum and restarts the cosine LR.
- **Smoke-test train.py changes** with `--num-batches 2 --eval-interval 99
  --eval-games 4` to reach a batch summary fast. Diagnose GPU crashes with
  `HIP_LAUNCH_BLOCKING=1` (an opaque HSA exception naming `index_elementwise_kernel`
  == an out-of-bounds tensor index).

## Recipe (v23, current best)

`--trainee-search --lookahead-alpha 1.0 --value-weight 1.0 --dtz-shaping-weight
0.15 --wdl-weight 1.0 --engine-ratio 0.25 --num-filters 192 --num-residual-blocks
16`, `tablebase_terminate_prob 0.25`. Trainee search applies in checkpoint +
engine batches, never self-play. `gamma=0.98`, `gae_lamb=0.95`, `entropy=0.005`,
`batch_size=32`, `opening_prob=0.6`, `temperature=1.0`, `ppo_clip=0.2`,
`target_kl=0.015`, `opponent_ratio=0.45`, `draw_penalty=0`. AdamW lr=5e-5, cosine
to 10%.

## Code state — load-bearing, don't change without reason

- **`lookahead.py`**: candidate set = top-k(π) ∪ captures ∪ checks; score
  `log π − β·V`; dominant `MATE_SCORE`; `stack=True` copies; `--use-wdl` leaf eval
  = P(win)−P(loss) threaded through *all* quiescence recursions.
- **`train.py`**: `--wdl-weight` trains the detached WDL head (CE) with a
  *separate* grad-clip so it can't scale the PPO update; `--engine-ratio` tunes
  the engine-curriculum weight; TB-adjudicated outcomes are retained for the WDL
  label; `--tb-value-aux-weight` dormant (v18 failed at 0.5). **Any NEW per-state
  tensor in the PPO loop must be mirror-doubled (~line 847)** or minibatch indexing
  OOB-crashes the GPU. Step penalty is load-bearing.
- **`models.py`**: separate `wdl_head` (conv adapter + 3 logits);
  `forward(x, with_wdl=True, wdl_detach=True)`. Backward-compatible — old
  checkpoints load with a fresh head via `net_from_state_dict`.
- **Tools**: `diagnose_wdl.py` (`--head value|wdl` vs Syzygy), `train_wdl_head.py`
  (supervised WDL head on a frozen trunk), `finetune_clean_value.py`.

## Opponent pool

Loader samples `.pt` from `models/` (non-recursive). **Pruned 2026-07-27**: weak
small nets (v4–v18, 2.77M) + `pretrained_big*` moved to `models/archive_weak/`;
pool is now only at-level 16×192 nets (v19–v23) so opponent batches give real
signal. Strongest: `ppo_search_v23_checkpoint_299.pt`.

## TODO / open questions

- **Combine scale-up + curriculum** — the next high-EV run (both levers proven).
- **Replace the saturated in-training auto-eval** with a raw H2H vs a fixed
  reference (add `--eval-ref`, default = `--init-from`; reuse
  `evaluate_vs_model.play_game`); >50% = progress / <50% = regression. Optional
  `diagnose_wdl` line.
- **Conversion still unsolved** — DTZ shaping (0.15) unvalidated; build a
  conversion testsuite before tuning it.
- **WDL head doesn't beat the value head in search yet** — revisit only if a
  deeper search makes calibration matter more.
