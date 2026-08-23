# Handover — ChessEngine self-play RL

Last updated: 2026-08-23. Full per-run history: the auto-loaded memory index
`~/.claude/projects/-var-home-eunis-Python-ChessEngine/memory/MEMORY.md`.

## Current state

**Strongest model: `models/ppo_search_v24_checkpoint_299.pt`.** Best objective
score the project has posted — **50.5% vs Stockfish** (skill 0 / 10ms, search,
131W/41D/128L over 300 games), up from v23's 39%. Also +37 Elo H2H over v23@299
(CI [−14,+88], necessary-condition check). **v25 is running** (launched
2026-08-23): same recipe with the curriculum teacher raised to Skill Level 1.

Lineage (all 16×192 / 11.5M-param net):
- **v19@449** — scaled-architecture baseline. Scaling broke the small-net ceiling.
- **v21@449** — WDL-head run. +50 Elo H2H over v19 but flat vs Stockfish (31%).
- **v23@299** — curriculum (pruned at-level pool + engine-ratio 0.25) → 39%.
- **v24@299** — curriculum stage 3 (engine-ratio 0.35) on the rewritten batched
  search → **50.5%**, the biggest single-step gain. Curriculum trend:
  **31% → 33.5% → 39% → 50.5%** (v21→v22→v23→v24).

**Proven absolute-strength levers: (1) harder curriculum (validated three runs
straight), (2) architecture scale-up.** v24 reached *parity* with the skill-0/10ms
teacher, so further curriculum steps must strengthen the teacher itself.
Teacher probes on v24@299 (100 games each): **32.5% vs skill-1/10ms** (healthy
30–40% gap → chosen for v25) vs **46.0% vs skill-0/50ms** — more time is a dead
knob at skill 0 (the error injection binds, not search depth).

## Key findings

- **H2H-vs-predecessor gains DON'T transfer to absolute strength** (v21 vs v19).
  **Track vs-Stockfish as the honest metric.** Only scale and curriculum have
  moved it.
- **Quiescence is now level-synchronous batched negamax** (one net forward per
  tree level, chunked at 2048) — **×10 wall-clock** on this latency-bound iGPU,
  verified equivalent to the old recursive alpha-beta on v23 weights
  (max |Δv| 1.5e-7 over 240 boards, 100% argmax agreement). Search batches
  dropped from 150–200s to 50–110s; a 300-batch run ≈ 7h.
- **In-training progress eval** = 32 paired-opening games vs protocol Stockfish
  each 50 batches. **Its absolute score is inflated** (4 concurrent games via
  EnginePool; same mechanism as the old in-training counter gap) — treat as a
  within-run trend gauge; promotion decisions use the standalone protocol only.
  The eval pool skill is pinned at 0 (`eval_engine_skill`) independent of the
  curriculum teacher's `--engine-skill-level`.
- **Dead-tail cutoff** (`--min-live-boards 3`, `--search-max-plies 300`) aborts
  opponent-batch rollouts when <3 games remain; truncated games GAE-bootstrap
  from 0 and are WDL-masked. Saves ~25% DataGen; exonerated by v24's gain. If
  ever suspected: ablate with `--min-live-boards 0 --search-max-plies 600`
  (params only), or fix the estimator (bootstrap truncations from the critic).
- **PPO valid-rows filter** drops rows with behavior log-prob < −15 (degenerate
  IS math from never-supervised logits, often mirrored actions). Note: KL-gauge
  spikes on search batches persist (filter keys on b; spikes come from tiny
  π_old) — cosmetic, the actor ratio uses b.
- **Search bug-fix history**: dominant `MATE_SCORE`; `stack=True` board copies
  (repetition/50-move planes); draws reward 0. Worth +114 Elo at v19.
- **Separate WDL head** calibrates to ~78% but doesn't beat the value head in
  search (within noise). Safe passenger (trunk-detached). `train_wdl_head.py
  --tablebase syzygy` can now mix Syzygy-exact labels.
- **MCTS off the table** (latency-bound iGPU); **Stockfish distillation rejected**
  by user. Stockfish-as-opponent is fine (sparring, not imitation).

## Evaluation protocol

- **vs-Stockfish is the honest absolute-strength metric** — `evaluate_vs_engine.py
  --engine-skill-level 0 --engine-move-time 0.01`, search `k=4/α=1.0/vw=1.0`,
  100+ games (300 preferred; SE ~2.7pp); keep skill 0 / 10ms fixed forever for
  comparability, regardless of the curriculum teacher's strength. H2H vs the
  previous baseline is *necessary but insufficient*.
- **H2H**: `evaluate_vs_model.py --paired-openings` (132 games, low-noise);
  `--use-wdl` to search via the WDL head.
- Report W/D/L, score, SE, Elo CI. Be rigorous about the noise floor.
- `conversion_suite.py` measures TB-win conversion rate + plies-over-DTZ
  (defender = same net). v23@299 smoke: 12/12 multi-piece but 2/6 single-piece
  (KQvK-type) at +40 plies — bare-conversion gap; confirm with ≥30 games/class.

## Runtime / infra

- Host: AMD Ryzen AI Max+ 395, Radeon 8060S iGPU (`gfx1151`), 128GB unified RAM.
  No NVIDIA. Throughput-bound GPU — batch everything; single-sample inference
  wastes it (hence the batched quiescence).
- Use `venv-rocm/bin/python` for training/eval (PyTorch 2.11 + ROCm 7.13);
  `venv` for general work.
- `MIOPEN_FIND_MODE=FAST` required; AMP off (FP16 → NaN); FP32 only.
- Timing (post-batched-search): self-play batches ~3s, checkpoint/engine batches
  50–110s, PPO update 2–10s. 300-batch run ≈ 7h. Standalone 300-game eval ≈ 20min.
- Checkpoints store weights only — `--init-from` continuations reset AdamW
  momentum and restart the cosine LR.
- **Smoke-test train.py changes** with `--num-batches 2 --eval-interval 99
  --eval-games 4`. Diagnose GPU crashes with `HIP_LAUNCH_BLOCKING=1` (opaque HSA
  exception naming `index_elementwise_kernel` == out-of-bounds tensor index).

## Recipe (v24 = current best; v25 = running)

`--trainee-search --lookahead-alpha 1.0 --value-weight 1.0 --dtz-shaping-weight
0.15 --wdl-weight 1.0 --engine-ratio 0.35`, `tablebase_terminate_prob 0.25`
(default). v25 adds `--engine-skill-level 1` (single variable). Trainee search
applies in checkpoint + engine batches, never self-play. `gamma=0.98`,
`gae_lamb=0.95`, `entropy=0.005`, `batch_size=32`, `opening_prob=0.6`,
`temperature=1.0`, `ppo_clip=0.2`, `target_kl=0.015`, `opponent_ratio=0.45`,
`draw_penalty=0`. AdamW lr=5e-5, cosine to 10%. Launch scripts: `train_v24.sh`,
`train_v25.sh`.

## Code state — load-bearing, don't change without reason

- **`lookahead.py`**: candidate set = top-k(π) ∪ captures ∪ checks; dominant
  `MATE_SCORE`; `stack=True` copies; quiescence = level-synchronous batched
  negamax (`quiesce_batched`), semantics equal to recursive alpha-beta at the
  root; `_MAX_LEVEL_EVALS=65536` raises on pathological trees (never observed).
- **`train.py`**: `--wdl-weight` trains the detached WDL head with a separate
  grad-clip; `--engine-ratio` + `--engine-skill-level` tune the curriculum;
  `OPPONENT_WEIGHTS` biases pool sampling by generation prefix (**add each new
  vN prefix when promoting**); eval pool pinned at protocol skill 0. **Any NEW
  per-state tensor in the PPO loop must be mirror-doubled (mirror block ~line
  1040)** or minibatch indexing OOB-crashes the GPU. Step penalty is load-bearing.
- **`models.py`**: separate `wdl_head`; `forward(x, with_wdl=True,
  wdl_detach=True)`; old checkpoints load via `net_from_state_dict`.
- **Tools**: `diagnose_wdl.py`, `train_wdl_head.py` (now with `--tablebase`),
  `finetune_clean_value.py`, `conversion_suite.py`.

## Opponent pool

Loader samples `.pt` from `models/` (non-recursive), weighted by
`OPPONENT_WEIGHTS` generation prefixes (v23/v24 at 3.0, older at 1.0–2.0; a
running generation's own checkpoints enter at base weight 1.0). Pool is at-level
16×192 nets only (v19+; weak nets archived in `models/archive_weak/`).
Strongest: `ppo_search_v24_checkpoint_299.pt`.

## TODO / open questions

- **v25 verdict** — standalone protocol match at the end; promote only beyond SE.
- **Combine scale-up + curriculum** — still the big-ticket run (both levers proven).
- **Conversion baseline** — run `conversion_suite.py` at ≥30 games/class on the
  baseline (and dtz-shaping on/off) now that the tool exists; the single-piece
  conversion gap is the known technique hole.
- **WDL head gate** — TB-finetune (`train_wdl_head.py --tablebase syzygy`), then
  paired `--use-wdl` H2H; deploy or retire.
- **Truncation bootstrap** — if dead-tail is ever implicated, bootstrap truncated
  trajectories from the critic's last value instead of 0.
