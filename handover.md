# Handover — ChessEngine PPO training

Last updated: 2026-07-17. See `~/.claude/projects/-var-home-eunis-Python-ChessEngine/memory/` for the auto-loaded memory index (`MEMORY.md`) and per-topic files.

## Current state

**Running (launched 2026-07-17): v16, 450 batches from v15-search@149, eval
every 150.** Same recipe as v15-search (`--trainee-search --lookahead-alpha
1.0 --value-weight 1.0`) plus the new `--dtz-shaping-weight 0.15`: dense
reward inside confirmed <=5-man tablebase wins for shrinking DTZ (distance to
the forced zeroing move), aimed at the draws-with-winning-material conversion
gap. See "DTZ conversion shaping" below for the mechanism. Log:
`logs/v16_run.log`; checkpoints at `models/ppo_search_v16_checkpoint_{149,299,449}.pt`.
Verdict on completion: raw-vs-raw H2H vs v15-search@149.

**v15 matched-pair search-in-training test SUCCEEDED — biggest single-run gain
to date.** Two 150-batch arms from the same init (v14@299, seed 1401):
`ppo_search_v15_search` (`--trainee-search --lookahead-alpha 1.0
--value-weight 1.0`, trainee samples from policy-scale search in checkpoint
AND engine batches, pure π in self-play) vs `ppo_search_v15_control`
(identical, no trainee search). Both arms' checkpoint opponents used the
fixed formula (α=1, β=1) so the single manipulated variable was trainee
search-behavior. Gate held: search-batch clip fraction ~0.39–0.48 but
`π↔search KL` only 0.024–0.040 nats (disagree ~20–25%) — a mild perturbation
of π, nothing like v13's near-uniform behavior policy (72–75% outside clip
from a *saturated* distribution, not just a narrow band). H2H results
(101 games, temp 1.0, raw-vs-raw unless noted):

- **v15-search-raw vs v15-control-raw: 67W/24D/10L → +222 Elo, CI [+140,+304]**
- **v15-search-raw vs v14-raw: 56W/21D/24L → +114 Elo, CI [+43,+185] — real
  generational gain, on par with v11's historic +114**
- v15-control-raw vs v14-raw: 29W/25D/47L → **−63 Elo, CI [−131,+6]** (z=−1.82,
  leans real but not 95%-significant) — plain PPO continuation likely
  *regressed* against the harder shared curriculum (both arms' opponents got
  stronger from the formula fix; the search arm's own moves coped, the
  pure-π control arm apparently didn't)
- search-raw vs search-search(β=2) on the v15-search checkpoint: wrapper wins
  46/21/34, +41 Elo, CI [−27,+110] — same inconclusive-but-positive pattern
  as the v14 self-match; still unresolved without paired openings

**v15-search@149 is the new strongest known model.** Files:
`models/ppo_search_v15_search_checkpoint_149.pt` (promote as next baseline),
`models/ppo_search_v15_control_checkpoint_149.pt` (kept for the record, not
a baseline candidate). Full match log: `logs/v15_h2h_suite_20260717.log`;
training logs `logs/v15_search_run.log` / `logs/v15_control_run.log`.

**Infra fix along the way**: `evaluate_vs_model.py` / `evaluate_vs_engine.py`
never set `MIOPEN_FIND_MODE=FAST` (only `train.py` did), so on gfx1151
(unreadable packaged FindDb) raw-policy forward passes hard-failed with
`miopenStatusUnknownError`. Both scripts now set it via
`os.environ.setdefault` at import time, mirroring train.py.

v14 (300-batch warm-start from v13 control@99) completed
2026-07-16 and **gained nothing: v14@299-raw vs control@99-raw is a statistical
tie** (36W/32D/33L). But the post-mortem produced the day's real findings:

1. **The k=4/α=0.3 search wrapper has been a large handicap for generations**
   (search vs own raw policy, 101 games each): v10@399 −95, v11@399 −185,
   v13-control@99 −162, v14@299 −125 Elo. Every historical H2H measured
   model+wrapper; the taxes roughly canceled between adjacent generations, so
   the ladder survives: **v13-control-raw beats v11-raw 65W/30D/6L (+232 Elo,
   CI [+175,+302])** — the +284 was real strength, not artifact.
2. **Root cause is one constant, not the search concept.** The old score
   `−V + 0.3·log π` sampled at temp 1 ≡ playing π at temperature 3.3 over a
   candidate set salted with every capture/check. Value ablation
   (`--value-weight 0`) made things *worse* (−197), proving the value/quiescence
   pathway carries ~+70 Elo of real signal. The **policy-scale formula
   `--lookahead-alpha 1.0 --value-weight β` (score = log π − β·V)** recovers
   fully: β=1 +17, β=2 **+45 Elo** vs raw (CI [−13,+103]) — search is
   neutral-to-positive again, pending a properly seeded confirmation match.
3. **Engine ground truth** (control@99 vs SF skill 0, 10ms, 101 games):
   raw 11W/6D/84L = 13.9% — externally validates the in-training vs-SF
   counter (~14%); old-formula search collapses it to 4.5%.

**Strongest known model: v15-search@149-raw (+114 Elo over v14@299-raw,
2026-07-17 — see Current state).** All evaluation should default to
raw-vs-raw (or the policy-scale search once confirmed); `evaluate_vs_model.py`
has `--raw-a/--raw-b/--value-weight`, `evaluate_vs_engine.py` has `--raw`.

- **v11** = two changes over v10: (1) **conv policy head** (`ConvPolicyHead` in `models.py`: 3×3 conv → GN → ReLU → 1×1 conv to 73 planes, square-major reshape; 157k params vs the legacy dense head's 603k), and (2) **fresh lineage** — seeded from `pretrained_conv.pt` (two 3-epoch PGN passes over the lichess elite files; 42.2% top-1 move accuracy vs 37.5% for the dense head on the identical recipe), *not* warm-started from v10. Trained 2026-07-12/13, 400 batches of the v10 recipe, clean exit.
- **H2H run 2026-07-13** (`logs/h2h_v11-399_vs_v10-399_*.log`), 101 games, temp=1.0, k=4, α=0.3:

  **v11@399 vs v10@399: 43W / 47D / 11L → score 65.8%, +114 Elo, 95% CI [+66, +166], LOS ≈ 1.0**

  One 400-batch run from a supervised seed beat the whole five-generation v4→v10 lineage. The result reads on {conv head + fresh seed} jointly, but the confound cut against v11, so the head/seed upgrade is the natural cause.
- Prior baseline for reference: v10@399 was +137 Elo over v6@399 (45W/49D/7L, 2026-07-12, `logs/h2h_v10-399_vs_v6-399_*.log`).
- Checkpoint compatibility: `net_from_state_dict()` auto-detects dense vs conv heads, so all old checkpoints still load at full fidelity (opponent pool works with mixed architectures).

**Caution for future runs:** v10's and v11's within-run signals (vs-random, vs-SF counter) are saturated/flat and badly understate cross-run progress. Don't kill a run on flat proxies alone; always settle rank with H2H vs the previous baseline.

## Next steps

v15 search-in-training test succeeded (+114 Elo over v14, +222 over the
matched pure-PPO control — see Current state). v15-search@149 is the new
baseline; v16 (running) bundles the generational continuation with DTZ
conversion shaping in one run, rather than testing them separately, per
user's call. Candidates, in rough priority order:

1. **v16 verdict — pending**. H2H raw-vs-raw vs v15-search@149 once it
   finishes. Note this is *not* a clean 1-variable test (continuation +
   DTZ shaping together), so a gain doesn't isolate which part helped; if
   it's a clear win that's fine (we get a better model), but don't credit
   DTZ specifically without a follow-up ablation (rerun without
   `--dtz-shaping-weight` if the credit assignment ever matters).
2. **Conversion testsuite** (still not built): a set of TB-won positions
   with known DTZ, scored by how often/fast the model actually converts —
   needed to directly confirm DTZ shaping fixed the draws-with-winning-
   material gap, independent of aggregate Elo.
3. **Investigate the v15-control-arm regression** (−63 Elo vs v14, CI
   [−131,+6]): plain PPO continuation against the harder shared curriculum
   (opponents now fixed-formula, stronger) may have actively hurt rather than
   merely plateaued. Not blocking, but worth understanding before assuming
   future pure-PPO continuations are simply "flat."
4. **Eval hygiene (LOW PRIORITY — user deprioritized 2026-07-17)**: seed +
   paired opening suite + PGN output in `evaluate_vs_model.py`. Still the
   right tool for the still-unresolved β=2-wrapper-vs-raw question (wrapper
   showed no benefit on v15-search: self-match +41 Elo CI [−27,+110]; engine
   16.8% raw vs 13.4% wrapped — both ties, but current lean is deploy raw).

AMD runtime note: the host is a Radeon 8060S (`gfx1151`). `venv-rocm` contains
PyTorch 2.11.0 + ROCm 7.13. The packaged MIOpen wheel has no readable gfx1151
FindDb, so runs use `MIOPEN_FIND_MODE=FAST`. AMP is disabled because an actual
v11 forward/backward smoke produced NaN FP16 gradients; FP32 gradients are
finite.

## Recipe history (most recent first)

| Run | Init | Recipe change | Result |
|---|---|---|---|
| **v15-search (succeeded, +114 Elo, new baseline)** | v14@299 | Fixed-formula search (α=1, β=1): checkpoint opponents in both arms use it; search arm also samples trainee behavior from it in checkpoint+engine batches (self-play stays pure π); 150 batches, seed 1401 | H2H **56W/21D/24L vs v14-raw** (+114, CI [+43,+185]); **67W/24D/10L vs matched control-raw** (+222, CI [+140,+304]) |
| v15-control (regressed?, −63 Elo) | v14@299 | Same run, no trainee search (pure-π continuation against the same harder curriculum) | H2H **29W/25D/47L vs v14-raw** (−63, CI [−131,+6], z=−1.82 — leans real, not 95%-significant); decisively behind v15-search |
| v14 (flat) | v13 control@99 | Pure warm-start continuation, 300 batches, seed 1401, dynamic curriculum incl. v13/v14 checkpoints | **Raw-vs-raw tie with init** (36W/32D/33L); within-run proxies healthy but no gain — continuations don't stack |
| **v13 PPO control (succeeded, +284 Elo search-wrapped / +232 raw)** | v11@399 | Raw-policy trainee rollouts; no distillation; 100-batch matched diagnostic | H2H **70W/29D/2L vs v11@399** (search-wrapped); **65W/30D/6L raw-vs-raw**; **strongest baseline** |
| v13 search arm (failed, -77 Elo) | v11@399 | Search behavior on checkpoint batches; `b_search` PPO denominator; positive-advantage distillation 0.025 | H2H **17W/45D/39L vs v11@399**; initial search-batch clip fraction 72-75%; do not continue |
| v12 (neutral) | `pretrained_conv_se.pt` (fresh PGN seed) | **SE blocks** in residual tower (+34k params); otherwise identical to v11 pipeline | H2H **23W/60D/18L vs v11@399** (+17 Elo, CI [−26,+61], LOS 78%) — statistical tie; SE not worth keeping on current evidence |
| **v11 (succeeded, +114 Elo)** | `pretrained_conv.pt` (fresh PGN seed) | **Conv policy head**; fresh lineage; recipe hparams unchanged from v10 | H2H **43W/47D/11L vs v10@399** (LOS ≈ 1.0); **then-current generational baseline** |
| v10 (succeeded, +137 Elo) | v9@249 | `opponent_ratio` 0.6 → **0.45**; `distill_weight` 0.05 → **0.025**; `batch_size` 24 → 32; `tablebase_terminate_prob` 0.33 → 0.25 | H2H **45W/49D/7L vs v6@399** (LOS ≈ 1.0) |
| v9 (killed @ b249) | v8@249 | `opponent_ratio` 0.4 → **0.6**; added engine W/D/L tracking (per-batch + cumulative) | Draws climbed 1→7 by b250 (late-onset v7 signature); killed, v10 launched from v9@249 |
| v8 (killed @ b250, possibly competitive) | v6@399 | Outcome-filtered distill (positive-advantage only); `distill_weight=0.05`; `temperature=1.0` | Own eval 98/2/0; v9's baseline (= v8@249 weights) scored 99/1/0 — statistically tied with v6@399 on vs-random. **H2H never run, true rank vs v6 unknown.** See `project_v8_plateaued.md` |
| v7 (failed, killed @ b50) | v6@399 | `distill_weight=0.05` (no outcome filter); `temperature=1.0` | Draws spiked 2→11 at first eval; same v5-style failure mode at lower magnitude; see `project_v7_failed.md` |
| **v6 (succeeded, +70 Elo)** | v4@599 | Widened lookahead `top-k(π) ∪ captures ∪ checks`; search only vs checkpoints; `distill_weight=0.0`; `temperature=1.25` | H2H **32W/57D/12L vs v4@599** (LOS ~99.9%); vs-random draws 5 → 1; **then-current generational baseline** |
| v5 (failed) | v4@599 | Heavy explicit distillation (`distill_weight=0.4`, search rollouts everywhere) | Aborted @ b149 after H2H 0W/33D/37L (~−100 Elo regression); distill drowned PPO outcome signal |
| v4 (historical baseline) | v3@199 | First PPO+search+light-distill recipe | +70 Elo over v3 cross-run, flat within-run |

## Code state — what's locked in (don't change without reason)

### `lookahead.py`
- `select_moves_with_lookahead`: **widened candidate set** = top-k(π) ∪ legal captures ∪ legal non-capture checks. Returns 10-tuple including `log_b_chosen`, `log_b_topk` for IS / distill.
- `select_moves_from_policy`: stores `log_pi` under the **actual sampling temperature** (`log_softmax(masked/T)`), so PPO old_log_prob matches the sampling distribution. T=1.0 in current config so this is mathematically neutral, but the fix matters if T≠1 is reintroduced.
- Quiescence has in-check evasion handling (stand-pat skipped in check; recursion floor at depth ≤ 0).

### `train.py`
- Default entrypoint is pure PPO (`trainee_search=False`, `distill_weight=0.0`).
  As of 2026-07-17 the CLI exposes `--trainee-search`, `--lookahead-alpha`
  (default 1.0) and `--value-weight` (default 1.0): the policy-scale search
  formula, shared by checkpoint opponents and (when enabled) the trainee
  behavior policy. Trainee search applies in checkpoint AND engine batches,
  never self-play. Distillation is not CLI-exposed.
- **Per-batch PPO denominator gate**: recorded search behavior probabilities
  are used only when `trainee_search_now=True`; pure-PPO batches use `pi_old`.
- **Outcome-filtered distillation** remains dormant with weight 0.
- **Schulman-k3 KL diagnostic**: `(exp(log_r) - 1 - log_r).mean()`. Non-negative, makes `target_kl=0.015` early-stop functional.
- **Padding** of variable-length `topk_idx` / `log_b_topk` to rollout-global max before stacking — prevents crash when `distill_weight > 0`.
- **CSV log hparam header**: first line of `logs/{model}_*.csv` is `# k1=v1 k2=v2 ...` of all hparams (parsers should skip lines starting with `#`).
- **Per-batch trainee outcomes**: `generate_batch` returns `stats["trainee_outcomes"] = {"W":, "D":, "L":, "T":}` for non-self-play batches. Printed inline; engine batches get cumulative running totals.
- **DTZ conversion shaping** (`--dtz-shaping-weight`, default 0.0, off): dense
  potential-based reward inside a *confirmed unconditional* ≤5-man tablebase
  win for the mover (`_dtz_progress_potential` in `train.py`), rewarding
  moves that shrink `probe_dtz`'s distance to the forced zeroing move. Bounded
  in [−1, 0] (dtz magnitude / 100), `None` (no shaping) outside the tablebase
  domain, on a draw/loss, or on a cursed win (dtz magnitude > 100 — a
  50-move-rule edge case). Same cycle-based Φ(s) pattern as the existing
  material shaping; a cycle only contributes when *both* endpoints are
  defined, so entering/leaving the domain is silent rather than injecting a
  spurious jump — this means a blunder that throws away a TB win is NOT
  penalized by this term specifically (relies on the existing terminal/WDL
  reward for that). Sign convention verified against a real KQvK tablebase
  position before the v16 launch (`phi(white)` rose monotonically −0.13 → 0
  as White played optimally toward mate; `phi(black)` was `None` throughout,
  correctly gated since Black wasn't winning). v16 uses weight 0.15, a first
  guess not yet tuned — no conversion testsuite exists yet to calibrate it.

### Prepared v14 configuration
- Required `--init-from` and `--model-name`; defaults are 400 batches, eval
  every 50, 100 eval games, and seed 1401.
- Init from v13 pure-PPO control@99; `use_se=False`.
- Radeon 8060S via `venv-rocm`; MIOpen FAST fallback; FP32 only.
- `gamma=0.98`, `gae_lamb=0.95`, `entropy_weight=0.005`, `batch_size=32`, `opening_prob=0.6`
- `temperature=1.0`, `ppo_clip_ratio=0.2`, `target_kl=0.015`, `ppo_minibatch_size=256`
- `opponent_ratio=0.45`, `engine_ratio=0.1`, `engine_skill_level=0`, `engine_move_time=0.01`
- `step_penalty=0.001`, `draw_penalty=0.1`, `material_shaping_per_pawn=0.025`
- `lookahead_k=4`; `lookahead_alpha`/`value_weight` now CLI (default 1.0/1.0 —
  the policy-scale formula; v14 itself ran the old α=0.3);
  `trainee_search` CLI flag (default off), `distill_weight=0.0`.
- `tablebase_terminate_prob=0.25`
- Optimizer: AdamW, lr=5e-5, cosine schedule to 10%

## Models folder

`models/` holds the 22 pre-v14 checkpoints (251 MB) plus six v14 checkpoints
(@49–@299) added by the 2026-07-15/16 run. The opponent loader dynamically
samples `.pt` files from this directory.

**Strongest known models**: v13 pure-PPO control@99 and v14@299, statistically
tied, *playing raw* (+232 Elo raw-vs-raw over v11@399-raw). Historic Elo
figures were measured search-wrapped; the wrapper taxed all generations
−95…−185 Elo (see Current state), so cross-era comparisons should use raw
protocol. Mixed head architectures load through `net_from_state_dict`.

## Open questions (still relevant even after v10's gain)

- **Search structure is the deep limit** — but fix the arithmetic before the architecture: the 2026-07-16/17 post-mortem showed the old `−V + 0.3·log π` score sampled at temp 1 flattened π to π^0.3 (−125…−197 Elo), while the value pathway itself contributes ~+70. The policy-scale form `log π − β·V` (α=1, β≈2) is neutral-to-positive. MCTS-lite / deeper search remains the long-term candidate, but any future search must keep the policy at its native scale.
- **Conversion remains unsolved and TB adjudication hides it**: 25% of rollout games credit a win upon *reaching* a TB-won position (train.py `tb_terminate`), and WDL has no distance-to-mate, so neither RL nor search gets a progress signal inside won endgames. DTZ shaping or joint TB-supervised aux loss are the candidates; build a conversion testsuite first.
- **The distillation magnitude trap**: even outcome-filtered, distill gradient is unbounded while PPO actor is clip-truncated in checkpoint batches (clip frac 0.78+). The asymmetry is structural. If we keep distillation, also clipping the distill gradient (or per-state weight clipping by advantage magnitude) might help.
- **Search behavior is not on-policy enough for clipped PPO**: in v13, 72-75%
  of initial search-batch ratios were outside `[0.8, 1.2]`, about 40% of actor
  samples were saturated, and the arm regressed. Revisit search-guided learning
  only as a separate supervised or explicitly off-policy objective.
- **vs-random is saturated; the vs-SF counter is now informative *across* runs but flat *within* them**: it moved ~1.5% (v10 era) → ~14% (v13/v14) and was externally validated by the 2026-07-16 standalone engine eval (13.9% raw). Track it cumulatively (~300-game aggregates, keep skill 0 / 10ms fixed); raw-vs-raw H2H remains the ranking measurement.
- **Evaluation reproducibility**: `evaluate_vs_model.py` starts every game from
  the standard position and has no explicit seed or paired opening suite. Add a
  seed option, fixed paired openings, and PGN output before the next close H2H.

## Files modified in working tree (uncommitted)

- `train.py`: `--dtz-shaping-weight` CLI flag and `_dtz_progress_potential`
  helper; DTZ shaping cycle plumbed through `generate_batch` alongside the
  existing material shaping (per-move + end-of-batch closing).
- This handover: v16 (search continuation + DTZ shaping) documented as
  running; next steps reordered around its verdict.
