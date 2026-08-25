# Handover — ChessEngine self-play RL

Last updated: 2026-08-25. Full per-run history: the auto-loaded memory index
`~/.claude/projects/-var-home-eunis-Python-ChessEngine/memory/MEMORY.md`.

## Current state

**Strongest model: `models/ppo_search_v26_checkpoint_299.pt`** — but only
nominally: v25@299 and v26@299 are *measurably interchangeable* at **54.0% vs
Stockfish** (skill 0 / 10ms, search, 300 games). Take either as the baseline.

Lineage (all 16×192 / 11.5M-param net):
- **v19@449** — scaled-architecture baseline. Scaling broke the small-net ceiling.
- **v21@449** — WDL-head run. +50 Elo H2H over v19 but flat vs Stockfish (31%).
- **v23@299** — curriculum (pruned at-level pool + engine-ratio 0.25) → 39%.
- **v24@299** — curriculum stage 3 (engine-ratio 0.35) on the rewritten batched
  search → **50.5%** (131/41/128), the biggest single-step gain.
- **v25@299** — first run at teacher skill 1 → 54.0% (144/36/120), z≈0.9 vs v24.
- **v26@299** — second run at teacher skill 1 → 54.0% (140/44/116), **z = 0.00
  vs v25**; H2H +29 Elo, CI [−24,+84]. Flat.

Curriculum trend: **31% → 33.5% → 39% → 50.5% → 54.0% → 54.0%** (v21→v26).

**The curriculum lever is spent.** 600 batches against the skill-1 teacher bought
+3.5pp over v24 in total (z ≈ 0.93, never significant) and the second 300 added
exactly nothing. v24's +11.5pp jump came from the engine-ratio 0.35 + rewritten
batched search recipe, *not* from teacher strength — raising skill 0→1 did not
reproduce it. **The one remaining proven lever is architecture scale-up** (v19
is the precedent: scaling decisively broke the previous plateau).

Teacher probes on v24@299 (100 games each, historical): **32.5% vs skill-1/10ms**
vs **46.0% vs skill-0/50ms** — more move time is a dead knob at skill 0 (the
error injection binds, not search depth). The 30–40% gap that motivated skill 1
turned out not to predict a training gain.

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

## Recipe (v25/v26 = current best, tied)

`--trainee-search --lookahead-alpha 1.0 --value-weight 1.0 --dtz-shaping-weight
0.15 --wdl-weight 1.0 --engine-ratio 0.35 --engine-skill-level 1`,
`tablebase_terminate_prob 0.25` (default). Trainee search applies in checkpoint
+ engine batches, never self-play. `gamma=0.98`, `gae_lamb=0.95`,
`entropy=0.005`, `batch_size=32`, `opening_prob=0.6`, `temperature=1.0`,
`ppo_clip=0.2`, `target_kl=0.015`, `opponent_ratio=0.45`, `draw_penalty=0`.
AdamW lr=5e-5, cosine to 10%. Launch scripts: `train_v24.sh`, `train_v25.sh`,
`train_v26.sh`. **Re-running this recipe unchanged is known not to gain** — the
next run needs a structural change.

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
Strongest: `ppo_search_v26_checkpoint_299.pt` (tied with v25@299). `OPPONENT_WEIGHTS`
currently tops out at v23/v24/v25 = 3.0; **add `"ppo_search_v26": 3.0` before the
next run.**

## TODO / open questions

**NEXT UP (user's choice, 2026-08-25): the WDL-head gate.** TB-finetune the
detached WDL head on `models/ppo_search_v26_checkpoint_299.pt` with
`train_wdl_head.py --tablebase syzygy`, check calibration with
`diagnose_wdl.py`, then gate it with a paired H2H —
`evaluate_vs_model.py --model-a <finetuned> --model-b
models/ppo_search_v26_checkpoint_299.pt --paired-openings --temperature 0
--lookahead-k 4 --lookahead-alpha 1.0 --value-weight 1.0 --use-wdl`
(A searches via the WDL head, B via the value scalar; same weights otherwise, so
this isolates the head). Deploy or retire the head on the result. Cheap
(~minutes), orthogonal to training, doesn't consume a training slot. Prior: the
head calibrates to ~78% but has never beaten the value head in search (within
noise) — this is the run that settles it.

Then, in rough order:
- **Scale-up run** — the only proven lever left after the v25/v26 curriculum
  plateau; v19 is the precedent. Bigger trunk from the v26 seed, teacher held at
  skill 1.
- **Conversion baseline** — `conversion_suite.py` at ≥30 games/class on the
  baseline, dtz-shaping on/off; the single-piece conversion gap (KQvK-type) is
  the known technique hole. NB: the tool's `except Exception: return True, plies`
  counts a failed TB probe as a success — fix before trusting the numbers.
- **Truncation bootstrap** — if dead-tail is ever implicated, bootstrap truncated
  trajectories from the critic's last value instead of 0.

## Uncommitted at last update (2026-08-25)

`train.py` (`OPPONENT_WEIGHTS += "ppo_search_v25": 3.0`) and `train_v26.sh`.
Never pushed for v26. Also long-uncommitted: `IMPROVEMENTS.md`,
`conversion_suite.py`, `train_v24.sh`, and edits to `lookahead.py`,
`pretrain_from_*.py`, `train_wdl_head.py`.
