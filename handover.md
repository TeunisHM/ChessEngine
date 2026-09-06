# Handover — ChessEngine self-play RL

Last updated 2026-09-01. **Per-run history lives in the auto-loaded memory index**
`~/.claude/projects/-var-home-eunis-Python-ChessEngine/memory/MEMORY.md` — this file
is only the current state, the standing rules, and what's load-bearing in the code.

## Current state

**Baseline: `models/ppo_search_v26_checkpoint_299.pt`** — 53.9% vs Stockfish
(skill 0 / 10ms, search; 792g pooled 2026-09-03). Unchanged since 2026-08-25.
Nothing in v27–v33 earned promotion.

**v32 KILLED, v33 FAILED (2026-09-01/02).** v32's 63.2% at skill 0 did not
survive confirmation: 600g pooled at skill 1 = 35.6% vs v26's 34.7%, z=+0.27.
v33 (100 batches from v32@149, skill sampled from {0,1,2}) then **regressed**:
skill 0 50.2% (z=−3.44 vs v32), skill 1 33.3% (tied), conversion 53/90, and
bare-king **0/27** — it lost KQvK as well as KRvK. The skill-mix hypothesis is
rejected: v32's skill-0 gain was error-profile-specific and diluting the
teacher removed it without adding transferable strength.

**v34 is RUNNING (2026-09-03) — clean cold start.** No pretraining, no
`--init-from`: fresh random 20×192 + 5-attention net, pure self-play
(`--opponent-ratio 0 --engine-ratio 0`), `--entropy-weight 0.02`, 500 batches,
`train_v34.sh`. Phase 2 (wide search, k=12) was deferred — see the cost note
below. **Progress eval is 0.000 at every point through batch 400** (0 wins in
288 games); entropy sharpens normally (−2.79 → −1.61) so optimization works,
but nothing transfers. Treat as a likely null pending a v34@399-vs-v34@49 H2H.

Curriculum trend (vs-SF skill 0): 31% → 33.5% → 39% → **50.5%** → 54.0% → 54.0%
(v21→v26), flat since. v24's jump came from engine-ratio 0.35 + the rewritten
batched search, not from teacher strength.

## Standing rules

- **vs-Stockfish is the honest strength metric.** `evaluate_vs_engine.py
  --engine-skill-level 0 --engine-move-time 0.01`, search `k=4/α=1.0/vw=1.0`.
  Keep skill 0 / 10ms fixed forever for comparability. **Use ~1000 games, not
  300** — see the calibration finding below.
- **Use skill 1 as a standing second eval point** (~20 min, good discriminator —
  it caught v31's regression and killed v32). **Avoid skill 3**: 11–13% is near
  the floor and score compression eats the SE.
- **Never rank two models on the in-training progress eval.** It is inflated
  (4 concurrent games via EnginePool) and on v28 it pointed the *wrong way* by
  14pp. Within-run trend only.
- **Track conversion as a second metric — vs-SF is blind to endgame technique.**
  v26→v27 fixed the K+R mate outright (KRvK 3/38 → 24/38, p<1e-06) with zero
  vs-SF movement. `conversion_suite.py`, 30 games/class, ~7 min. **Always pass
  `--defender-model`** for cross-model comparisons (shared-net defence confounds).
  v26@299 baseline: 37/87/97% for 3/4/5-man, 66/90 total.
- **H2H vs predecessor is necessary but insufficient** — v21 beat v19 by +50 Elo
  and was flat vs Stockfish. Self-play gains are predecessor-specific.
- Report W/D/L, score, SE, Elo CI; be rigorous about the noise floor.

## Key findings

- **THE METRIC IS UNDERPOWERED — this reframes the whole v28–v33 program
  (measured 2026-09-03).** Four replicates of v26@299, identical config,
  span **53.8% – 60.4%**. Run-to-run SD is 3.07pp unpaired / 3.42pp paired
  against a binomial SE of 2.8pp — i.e. **ordinary binomial noise, just larger
  than the project has been treating as meaningful**. There is no
  over-dispersion; games are effectively independent. Consequence: a single
  300-game result cannot distinguish a 5pp effect, and v28a's 57.7%, v32's
  63.2% and the k=12 width result all sit inside v26's own noise band.
  **~1000 games/arm gives SE_diff ≈ 2pp** — the fix is power, not a new metric.
  `--paired-openings` was added to `evaluate_vs_engine.py` and **does not help**
  (defaults unchanged; harmless to ignore).
- **Search width/depth ladder: NULL (2026-09-03).** On v26@299, k=12 first
  measured +10.3pp over k=4 (z=2.82) and **failed to replicate** (64.2% → 56.0%
  on identical config). The ladder is internally inconsistent — k=20 landed at
  49.3%, *below* the k=4 control and 13.5pp below k=32 — and at skill 1 the gain
  is +3.0pp (z=0.82). Quiescence depth (qd=4) is +5.3pp (z=1.43) at 2.3× the
  cost. Nothing here is established. `--max-qdepth` / `--check-budget` are now
  exposed on the engine eval; `check_budget` had never been reachable at all.
- **v26 vs v33 (latest attention net): TIED, leaning negative.** 3 paired
  replicates each, n=792/arm — the best-powered comparison in the project:
  v26 53.85% vs v33 51.83%, **−2.0pp, z=−0.88**. The attention scale-up is not
  an improvement on the v26 baseline.
- **Searched self-play on a cold-start net is unaffordable** (~25 min/batch,
  ~125h for 300 batches) vs ~1 min raw. Search re-ranks by V(child), which is
  noise at init, and untrained play trades nothing so the captures∪checks
  candidate set stays huge. Note `--rollout-max-plies` (default 600) assumes
  "raw policy, cheap per ply" and `--min-live-boards` is opponent-batches-only,
  so **both cost controls are off for searched self-play.**
- **Attention beat pure conv in the v28 A/B** (57.7% vs 48.2%, z=2.33) — but
  that was one 300-game run, the power level the calibration above says cannot
  resolve 9pp reliably, and the same architecture (v33) later tied v26 at
  n=792. Suggestive, not settled. Supervised pretrain accuracy was a dead heat
  and did **not** predict RL strength — don't pick an architecture on it.
- **Fresh PGN seeds lose the bare-king mate.** Both v28 arms: KRvK+KQvK 0/27 and
  3/27 vs v26's 11/27. Total conversion is unaffected — damage is localised.
  Self-play from openings almost never *reaches* bare KRvK, so a short fresh run
  never learns it. This blocks promotion of any fresh-seed lineage.
- **The "training is a random walk" diagnosis was WRONG and is retracted.**
  0/398 gradient cosines negative, all 220–500× above chance. Optimization is
  healthy; different opponents push *orthogonally*, not in opposition. The
  plateau is a **transfer** problem. Do not spend more runs on gradient tuning.
- **Self-play makes lower-signal updates** — cosine +0.073 vs +0.097 for
  checkpoint/engine, z=3.35 with the search confound removed; it also takes the
  largest steps. Real, but the mix lever is far too small to matter (v30 moved
  the cosine exactly as forecast and strength did not budge).
- **Levers measured and dead:** curriculum/teacher skill (600 batches at skill 1
  bought +3.5pp total, z≈0.9), teacher skill *mixing* (v33 regressed), LR
  annealing (v29 null), opponent mix (v30 null), batch size 8 (v31 regressed,
  z=−3.06), endgame seeding (v27 A/B, p=0.63),
  the separate WDL head as a search evaluator (ties the value scalar even at
  90.3% calibration), MCTS (latency-bound iGPU), Stockfish distillation (user
  rejected). More move time at skill 0 is also dead — the error injection binds.
- **Material-shaping drift on unequal material is CORRECT PBRS.** Verified
  numerically 2026-08-26, policy invariance holds. Closed — don't re-litigate.
- **Search bug-fix history** (worth +114 Elo at v19): dominant `MATE_SCORE`,
  `stack=True` board copies, draws reward 0. Quiescence is now level-synchronous
  batched negamax — ×10 wall-clock, verified equivalent to the old recursive
  alpha-beta (max |Δv| 1.5e-7, 100% argmax agreement).

## Next up

1. **Raise the standard eval to ~1000 games/arm** (~56 min vs 17 for 300).
   At n=300 the metric cannot see the effects this project chases; at n=1000
   SE_diff ≈ 2pp. Every false positive in the v28–v33 program would have been
   called correctly the first time.
2. **Re-measure search width at that power.** k=12 looked like +10.3pp and
   evaporated on replication; it is neither established nor refuted.
3. **Batch size is NOT cleanly measured.** Within-batch gradient agreement
   doesn't rise monotonically (8→0.04, 48→0.41, 64→0.33) because source
   composition dominates (self-play 0.235 vs engine 0.454 at batch 64). Any
   future test must hold the mix fixed.
4. **Truncation bootstrap** — only if the dead-tail cutoff is ever implicated:
   bootstrap truncated trajectories from the critic instead of 0.

**Scale-up is NOT the open lever it was described as.** v28–v33 already *are*
the scale-up (v26 = 12.17M/16 blocks → v33 = 16.67M/20 blocks + 5 attn) and
went 0-for-6. v28b, a clean depth-scale of v26's architecture, scored *below*
baseline. Confounded with fresh seeds and short runs, so not fully settled —
but it is spent, not untried.

## Recipe (v25/v26 = current best)

`--trainee-search --lookahead-alpha 1.0 --value-weight 1.0 --dtz-shaping-weight
0.15 --wdl-weight 1.0 --engine-ratio 0.35 --engine-skill-level 1`,
`tablebase_terminate_prob 0.25`. Trainee search applies in checkpoint + engine
batches, never self-play. `gamma=0.98`, `gae_lamb=0.95`, `entropy=0.005`,
`batch_size=32`, `opening_prob=0.6`, `temperature=1.0`, `ppo_clip=0.2`,
`target_kl=0.015`, `opponent_ratio=0.45`, `draw_penalty=0`. AdamW lr=5e-5,
cosine to 10%. **Re-running this unchanged is known not to gain.**

## Runtime / infra

- AMD Ryzen AI Max+ 395, Radeon 8060S (`gfx1151`), 128GB unified. No NVIDIA.
  Throughput-bound — batch everything.
- `venv-rocm/bin/python` for training/eval; `venv` for general work.
  `MIOPEN_FIND_MODE=FAST` required; **AMP off (FP16 → NaN), FP32 only.**
- Timing: self-play batch ~3s, checkpoint/engine 50–110s at batch 32 (200–340s
  at batch 64), PPO update 2–10s. 300-batch run ≈ 7h; 300-game eval ≈ 20 min.
- Checkpoints are weights only — `--init-from` resets AdamW momentum and
  restarts the cosine LR.
- Smoke-test train.py with `--num-batches 2 --eval-interval 99 --eval-games 4`.
  Debug GPU crashes with `HIP_LAUNCH_BLOCKING=1` (opaque HSA exception naming
  `index_elementwise_kernel` == out-of-bounds index).
- Recover a killed run by relaunching with the last checkpoint as `--init-from`.

## Code state — load-bearing, don't change without reason

- **`train.py`**: **any NEW per-state tensor in the PPO loop must be
  mirror-doubled (~line 1040)** or minibatch indexing OOB-crashes the GPU.
  **Step penalty is load-bearing.** `OPPONENT_WEIGHTS` biases pool sampling by
  generation prefix — **add each new vN prefix when promoting**. Eval pool skill
  is pinned at 0 independent of the curriculum teacher's `--engine-skill-level`
  (which now accepts a list, sampled per batch). Diagnostic flags, all
  off/neutral by default: `--batch-size`, `--lr-min-ratio` (**1.0 disables
  annealing**), `--opponent-ratio`, `--trainee-search-selfplay`,
  `--log-step-cosine`, `--log-batch-diversity`, `--endgame-start-prob`,
  `--lookahead-k`, `--entropy-weight`, and cold-start arch flags
  `--num-attention-layers` / `--rel-bias` / `--attn-after-stem` (ignored with
  `--init-from`). **`--trainee-search` never applies to self-play** — pair it
  with `--trainee-search-selfplay` or a pure-self-play run does no search at
  all and `--lookahead-k` is inert.
- **`models.py`**: attention is configurable — `num_attention_layers` (0 = pure
  conv), `attn_after_stem`, `rel_bias`. Layout saved in the `attn_positions`
  buffer; the trailing layer keeps the name `transformer` so **pre-v28
  checkpoints load byte-identically**. **GOTCHA**:
  `nn.TransformerEncoderLayer`'s fused kernel returns **NaN** for a 3D float
  `src_mask` on this ROCm build, so the rel-bias path runs the pre-LN block via
  the layer's own submodules (CPU-verified equivalent). ROCm attention kernels
  are numerically loose (~2e-3) — pre-existing.
- **`lookahead.py`**: candidate set = top-k(π) ∪ captures ∪ checks; dominant
  `MATE_SCORE`; `stack=True` copies; `quiesce_batched` equals recursive
  alpha-beta at the root; `_MAX_LEVEL_EVALS=65536` guard (never tripped).
- **Dead-tail cutoff** (`--min-live-boards 3`, `--search-max-plies 300`) aborts
  opponent rollouts when <3 games remain; truncated games GAE-bootstrap from 0
  and are WDL-masked. Saves ~25% DataGen; exonerated by v24's gain. Ablate with
  `--min-live-boards 0 --search-max-plies 600` if ever suspected.
- **PPO valid-rows filter** drops rows with behavior log-prob < −15. KL-gauge
  spikes on search batches are cosmetic (the actor ratio uses `b`).
- **Eval harness**: `--use-wdl` in `evaluate_vs_model.py` was applied to *both*
  sides before 2026-08-25 — **any WDL A/B before that date measured nothing.**
- Tools: `diagnose_wdl.py`, `train_wdl_head.py`, `finetune_clean_value.py`,
  `conversion_suite.py` (probe-failure + plies-averaging bugs fixed 2026-08-25;
  earlier numbers unreliable).

## Opponent pool

Loader samples `ppo_search_*_checkpoint_*.pt` from `models/` (non-recursive),
sorted then weighted by `OPPONENT_WEIGHTS` (v23–v26 at 3.0, older 1.0–2.0; a
running generation enters at 1.0). Pool is 16×192 (v19+) plus each run's own
checkpoints. Weak nets archived in `models/archive_weak/`. Non-`_checkpoint_`
files (`ppo_search_v26_299_wdltb.pt`, supervised seeds) are excluded.

## Uncommitted

Nothing committed since v26 — `git status` is the source of truth. Working tree
carries every change from v27 onward: `train.py`, `models.py`, `lookahead.py`,
the three eval/suite scripts, `pretrain_from_pgn.py`, `test_helper.py`, all
`train_v*.sh` / `run_*.sh`, and `IMPROVEMENTS.md`.
