# Findings & Low-Compute Experiments

> **STATUS UPDATE (2026-08-23): items 1–7 below are implemented and measured.
> Results in "Session results" at the bottom. v24 curriculum run launched
> (`train_v24.sh`, PID logged in `logs/v24_pipeline.log`).**

Reviewed: `train.py`, `lookahead.py`, `models.py`, `helper.py`, `pretrain_from_{pgn,puzzles,tablebase}.py`,
`evaluate_vs_{random,model,engine}.py`, `diagnose_wdl.py`, `train_wdl_head.py`, `finetune_clean_value.py`,
`train_v21.sh`, plus `logs/v23_pipeline.log` timing. Budget constraint: ≤ ~50% extra compute per experiment;
several items below *reduce* compute instead.

Current baseline: v23@299, 39% vs Stockfish (skill 0 / 10ms), recipe in `handover.md`.
Proven levers: harder curriculum, architecture scale-up. Everything below respects the two
infra hard rules (FP32 only — AMP NaNs on gfx1151; `MIOPEN_FIND_MODE=FAST`).

---

## 1. Free wins: make search batches 3–5× cheaper (no algorithmic risk)

### 1a. Batch the leaf-value inference in quiescence  ← biggest single win

**Evidence.** Search-based batches dominate wall-clock: self-play DataGen ~2–3 s vs
checkpoint-opponent ~150–200 s and engine ~100–155 s (`logs/v23_pipeline.log`, batches 284–300),
at ~70% of batches being search batches (engine_ratio 0.25 + opponent_ratio 0.45). Train is only
2–15 s. The gap is `select_moves_with_lookahead` → `_quiesce_ab` evaluating leaves **one board at a
time**: `_leaf_value` (`lookahead.py:27`) builds a single-sample tensor and runs a solo forward pass,
and `quiesce_batched` (`lookahead.py:197`) is a serial Python DFS. On a latency-bound iGPU each
1-sample forward wastes the GPU's batch parallelism entirely.

**Change.** Rewrite quiescence breadth-first instead of depth-first:
- Level 0: batch-evaluate stand-pat V for *all* candidate children of all boards in one forward pass.
- Level 1: batch-evaluate all capture replies that survive the stand-pat bound check, etc.
With `max_qdepth=2` there are only 2–3 levels, so this is ~2–3 large batched forwards per ply
instead of hundreds/thousands of serial ones. Alpha-beta pruning loses some selectivity, but pruned
leaves were exactly the ones that were expensive; batched evaluation is nearly free per leaf.
Same idea applies to `_leaf_value` calls inside check-evasion recursion.

**Cost/benefit.** Pure engineering, zero training-math change (values identical up to pruning order).
Expected 3–5× faster DataGen on engine/checkpoint batches → roughly **2× overall training
throughput**, or equivalently +50–100% batches at unchanged wall-clock. Verify first with a paired
H2H (`evaluate_vs_model.py --paired-openings`) of old-search vs new-search on the same weights —
expect ~50/50.

### 1b. Stop grinding dead tails in rollout batches

**Evidence.** Batches routinely spend 100+ plies with 1–3 live games (log batch 285: 32→1 live by
ply 190, then grinds to ply 330+; batch 288 similar). Those tail plies still pay per-ply engine
calls, TB probes, and Python overhead serially — ~25–30% of DataGen time for almost no states.

**Change (either/both):**
- Add a `min_live_boards` cutoff (e.g. abort when < 3 live) — timeout trajectories already
  bootstrap from 0 and are masked from the WDL label (`train.py:530-532`), so the machinery exists.
- Cap `max_plies` at ~300 for search batches specifically (default is 600, `train.py:268`);
  self-play games rarely need it, search batches never finish 600 live anyway.

**Risk.** Slightly more timeout-masked states; negligible at these cutoffs.

### 1c. Micro-cleanups (small, safe)

- `board.copy(stack=True)` per candidate child (`lookahead.py:297`) copies the full move stack —
  needed for correct repetition planes (keep it!), but consider copying once per board and pushing
  candidates onto clones rather than re-copying per candidate.
- `legal_moves_mask` + `board_to_tensor` are recomputed for the same boards across policy-select
  and search paths; a tiny memo cache keyed on `board._transposition_key()`-style state would shave
  CPU in the rollout hot loop.
- `_load_random_opponent` re-loads a ~50 MB checkpoint from disk every opponent batch
  (`train.py:185`). Cache the loaded nets (pool is only ~11 files, 128 GB RAM) — saves a few seconds
  per opponent batch and lets you sample without I/O cost.

---

## 2. Training-loop improvements within budget

### 2a. Replace the saturated in-training eval (your TODO — endorsed, concrete design)

vs-random is 100/0/0 forever (log: `[Eval] Wins: 4/4` at eval_games=4). It burns ~2–10 min per
checkpoint for zero signal. Swap `evaluate_vs_random` in `train_actor_critic`
(`train.py:729-736, 1065-1075`) for a paired-openings H2H vs a frozen reference:
- Reuse `evaluate_vs_model.play_game` (already importable, supports `--paired-openings` logic).
- Reference = `--init-from` checkpoint by default; 16 book lines × 2 colors = 32 games ≈ 3–6 min
  with the 1a speedup. Log score + simple SE; >55% = progress, <45% = regression tripwire.
This makes the eval CSV meaningful for early stopping/promotion decisions at ~equal cost.

### 2b. Weighted opponent sampling

`_load_random_opponent` samples the pool uniformly, so half the opponent strength mass is
v19/v20@149-era nets. After the v23 experience ("opponent batches must give real signal"), bias
sampling toward the top of the pool (e.g. weights {v21@449: 3, v23@299: 3, v23@149: 2, rest: 1}).
One-line change in `train.py:185-206`; fold into the next RL run rather than running alone.

### 2c. Highest-EV RL run: v24 = curriculum stage 3

The curriculum trend is the project's cleanest signal: 31% → 33.5% → 39% (v21→v22→v23) as
engine pressure rose. Next step that fits the budget: **continue from v23@299 with engine_ratio
0.35–0.40, 250–300 batches** (~same length as the v23 run; with 1a+1b it's cheaper). Keep
everything else per the v23 recipe. Decision rule from your own protocol: promote only if the
paired vs-Stockfish eval beats 39% beyond SE; H2H-vs-v23 is necessary but insufficient.
Optionally co-test 2b in the same run (single variable caveat: if it regresses you won't know
which knob — if you want clean attribution, spend the budget on two short runs instead).

### 2d. Make the WDL head earn deployment or retire it

Status: calibrated to ~78%, ties the value head in search (+161 vs +138 over raw, within noise).
Two cheap follow-ups before deciding:
1. **Syzygy-supervised WDL finetune** (extend `train_wdl_head.py` to take a tablebase path and
   label with exact WDL instead of PGN outcomes; reuse `pretrain_from_tablebase._label_position`).
   Minutes of supervised compute on the frozen trunk, targets exactly the endgame regime where
   search calibration matters most. Check with `diagnose_wdl.py --head wdl` (78% → ?).
2. Paired 132-game H2H: v23 searched with `--use-wdl` vs v23 searched with the value scalar,
   before and after (1). If post-TB-finetune `--use-wdl` wins beyond noise, deploy it in search;
   otherwise the head rides along unharmed (it's detached, policy-safe).

### 2e. Build the conversion testsuite before touching dtz_shaping (your TODO)

DTZ shaping (0.15) is in the champion recipe but unvalidated. Cheap CPU-only harness:
- Generate KQvK/KRvK/KRPvK start positions via `helper.random_endgame_board`, force the net to play
  the winning side (argmax + search) against itself/defender, measure conversion rate + mean DTZ
  error vs `tablebase.probe_dtz`.
- Metric: % won conversions and plies-over-optimal. Run on v23 with shaping on/off (inference
  only, no training) to see if 0.15 is helping, hurting, or invisible. Only then consider tuning.

---

## 3. Pretraining data-quality fixes (near-zero compute, do before the scale-up run)

These matter because the planned "scale-up + curriculum" run will want a strong pretrained seed.

### 3a. Puzzle value targets are dishonestly +1 everywhere

`pretrain_from_puzzles.py:118` sets `value_target = 1.0` for **every** solver position. That's
roughly right for mating tactics but wrong for quiet/defensive/endgame-themed puzzles, and it
trains the value head to shout "winning" whenever a tactic exists — exactly the miscalibration
`diagnose_wdl.py` was built to catch. Options (cheapest first):
- Filter themes to tactic families where +1 is defensible (`mate`, `winningMaterial`, `crushing`)
  and drop `defensiveMove`/`quietMove`/`endgame`;
- or set value target from a quick material/WDL heuristic instead of a constant;
- or train puzzles policy-head-only (`value_loss_weight=0`) so the biased label can't reach the
  value head.

### 3b. `--min-result` default is a silent no-op

`pretrain_from_pgn.py` filters positions with `value_target >= min_result` (:152), but targets are
discounted into (-1, 1] and the default is `-1.5` (:296) — nothing is ever filtered. Either the
intent (drop hopeless lost positions) needs `min_result ≥ 0`-style logic, or remove the flag.
As-is it misleads anyone tuning data mixes.

### 3c. `value_discount=0.99` crushes midgame value signal — worth one A/B

A 150-ply game discounts the outcome to 0.99^150 ≈ 0.22, so midgame value targets hover near 0
regardless of who's winning; AlphaZero-style pretraining uses the *undiscounted* outcome and relies
on averaging over many games. Experiment: pretrain two seeds (discount 0.99 vs 1.0, everything else
equal — ~1 h supervised each), then run **short** 100–150-batch RL continuations from each and
compare vs-random-free metrics (H2H between the two results + vs-Stockfish at reduced games).
Total ≈ one normal RL run spread over four stages; fits the budget.

### 3d. Puzzle/tablebase pretrainers ignore architecture flags — blocks the scale-up plan

`pretrain_from_puzzles.py:191` and `pretrain_from_tablebase.py:312` hardcode
`ActorCriticResNet()` (128×8 default). When you do the proven-lever scale-up run (16×192 or
larger, cold start), you'll want these seeds at the target width. Port the
`--num-filters/--num-residual-blocks/--se` handling from `pretrain_from_pgn.py` (5-line change).
Do it now so it's not a blocker later.

---

## 4. Representation notes (deferred — only at the next cold start)

Flagged for completeness; each invalidates all existing checkpoints, so bundle them with the
scale-up run rather than paying a fresh from-scratch run on their own:
- **No halfmove-clock plane.** Plane 17 is scaled fullmove number (`helper.py:360`); the 50-move
  clock — central to conversion play and threefold/repetition strategy — is unencoded. One extra
  input plane is the standard fix.
- Repetition is a single binary `is_repetition(2)` plane (`helper.py:374`); a count-based pair
  (twofold/threefold) is strictly more informative.
- Castling-rights planes are full-board constants; fine, just noting the encoding is compact
  enough that one more plane is cheap.

None of these require more compute *per se*, but retraining from scratch does — hence deferred.

---

## 5. Things that look off but are fine (checked, no action)

- Mirror doubling of `returns/tb/outcome` targets (`train.py:845-862`) is correct — mirror-invariant
  quantities duplicated, actions remapped via `MIRROR_ACTION_PERM`.
- IS ratio uses log b (search behavior policy) as denominator with π_old kept separately for the
  KL diagnostic (`train.py:871-879`) — well-formed importance sampling.
- Draw penalty 0 with asymmetric terminal rewards (+1.25/−1.0) is deliberate; step penalty remains
  load-bearing.
- GroupNorm (not BN) is the right call for PPO train/eval consistency.
- High clip fraction (0.35–0.40) on search batches with early stops at epoch 2–3 looks alarming but
  is expected: b ≠ π by construction there, so ratios legitimately sit near the clip band. If you
  ever revisit, the knob is `ppo_epochs`/lr on search batches — low priority.

---

## Recommended order

| # | Item | Type | Compute | Expected payoff |
|---|------|------|---------|-----------------|
| 1 | 1a batched leaf inference | eng | negative (faster) | 2×+ throughput; unlocks everything |
| 2 | 1b dead-tail cutoff | eng | negative | +20–30% on search batches |
| 3 | 2a real in-training eval | eng | ~neutral | usable progress signal per checkpoint |
| 4 | 2e conversion testsuite | tooling | CPU-only | validates dtz_shaping=0.15 in champion recipe |
| 5 | 2d WDL head: TB finetune + H2H gate | superv. | minutes–1 h | deploys or retires the WDL head cleanly |
| 6 | 3a–3d pretraining fixes | data | ~neutral | better seeds for the future scale-up run |
| 7 | 2c v24 curriculum @ engine_ratio 0.35 (+2b folded in) | RL | ~1 normal run | direct shot at >39%; highest-EV training run |
| 8 | 3c value_discount A/B | superv.+RL | ~1 normal run split | better midgame critic |

Items 1–4 make every subsequent run cheaper; 5–6 are cheap gates; 7 is the headline experiment.
Deferred (needs > budget, per handover): scale-up + curriculum combination, input-plane revision (§4).

---

## Session results (2026-08-23)

### Bugs fixed
1. **Puzzle value labels** (`pretrain_from_puzzles.py`): +1.0 for every solver position →
   theme-gated (+1 only for `mate*`/`crushing`/`winningMaterial`; conservative 0.0 otherwise).
   New `--value-mode {themes,const}` (default `themes`). Verified: 2000-sample mix = 69% / 31%.
2. **`--min-result` no-op** (`pretrain_from_pgn.py`): filtered the *discounted* target (never fires)
   → now filters the undiscounted mover outcome; help text documents `0` drops lost games.
3. **Arch flags**: puzzle/tablebase pretrainers accept `--num-filters/--num-residual-blocks/--se`
   (were hardcoded to 128×8 — would have blocked scale-up pretraining).
4. **NEW BUG found & fixed — degenerate-logit PPO poisoning** (pre-existing in HEAD): trained nets emit
   astronomically negative raw logits on legal actions (e.g. −237 nats), mostly on mirrored states;
   those PPO rows have meaningless behavior probabilities → IS ratios underflow/explode, k3-KL spikes
   into the thousands (historical logs show self-play batches with KL 416/318; smoke tests hit 25661),
   causing **spurious early stops and gradient garbage**. Fix: rows with old_lp ≤ −15 are excluded
   from the update (`train.py`, `valid_rows`). Smoke test: KL 2177 → **0.032**, actor loss sane.

### Speed work
5. **Batched quiescence** (`lookahead.py` rewritten as level-synchronous batched negamax):
   - Equivalence vs old alpha-beta: max |Δ| 5e-7 over 75 positions × {qd 2/3, budget 0/1/2, wdl on/off}
     (root-exact by construction: cutoffs fire iff best ≥ beta since incoming alpha < beta).
   - CPU small-net benchmark: **45.6× faster**. GPU 16×192 real net: 20.1 → 3.05 ms per root (**7×**;
     now python-chess movegen-bound). Production engine batches: 100–155 s → **43–76 s**.
6. **Dead-tail cutoff**: search batches cap at `--search-max-plies 300` (self-play keeps 600) and abort
   when < `--min-live-boards 3` games remain. Tail grinding past ply 300 is gone.
7. **Weighted opponent sampling**: pool weights by generation (`OPPONENT_WEIGHTS`: v23×3 … v19×1).

### Eval replacement (§2a)
vs-random is gone from training. In-training eval = paired book openings vs protocol Stockfish
(skill 0 / 10ms, k=4 α=1 vw=1) or a frozen reference net (`--eval-opponent {engine,ref,none}`,
`--eval-ref` defaults to init checkpoint). CSV now logs score. Baseline sanity: v23@299 scored
62.5% vs SF on 32 paired games (note: different setup than the 39% protocol number — absolute
levels differ, relative signal is what matters run-to-run).

### Conversion testsuite (§2e) — new tool `conversion_suite.py`
v23@299 plays Syzygy-confirmed clean wins against itself:
| extra pieces | converted | plies over optimal |
|---|---|---|
| 1 (KQvK/KRvK-class) | **50%** | +47 |
| 2 | 75% | +16 |
| 3 | 100% | +11 |
| **total** | **75%** | |
The champion throws away ~25% of forced wins, worst on basic endings — likely tied to the missing
halfmove-clock plane (can't see the 50-move rule approaching). dtz_shaping=0.15 is NOT solving it.

### WDL head TB finetune (§2d)
`train_wdl_head.py --tablebase syzygy` mixes 40k exact-label endgames with PGN outcomes (~3 min GPU).
Calibration on random ≤5-man positions (`diagnose_wdl.py --head wdl`):

| model | WDL acc | MAE | draw bias |
|---|---|---|---|
| v23 value head | 66.6% | 0.397 | +0.32 (optimistic) |
| v23 WDL head (PGN-only) | 72.1% | 0.382 | +0.37 |
| **v23 WDL head (TB-finetuned)** | **89.8%** | **0.124** | +0.08 |

Deployment gate (132 paired openings, identical trunk, k=4 α=1): **TB-WDL-search beats value-search
59W–46B (+27D) ≈ 54.9% ≈ +34 Elo (1.13σ)** — first positive gate for the WDL head; worth confirming
with more games before switching the eval protocol's default.

### Launched: v24 curriculum stage 3
`./train_v24.sh` — from v23@299, `engine_ratio 0.35`, 300 batches, all improvements active.
Early pace ~35 s/batch (≈3 h projected vs ~10 h historically); KL healthy (0.02–0.06);
watch: `tail -f logs/v24_pipeline.log`. Promote only if paired vs-SF beats 39% beyond noise.

### Not run (out of session scope)
- §3c value_discount A/B (needs 2 pretrains + 2 short RL runs).
- Longer WDL-gate confirmation match; conversion suite at higher n / after v24.
- One-off flake seen: `malloc(): unaligned tcache chunk detected` during one launch (MIOpen import
  race); relaunch was clean — if it recurs, clear the MIOpen cache dir.
