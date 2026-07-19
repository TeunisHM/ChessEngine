# Handover — ChessEngine PPO training

Last updated: 2026-07-17. See `~/.claude/projects/-var-home-eunis-Python-ChessEngine/memory/` for the auto-loaded memory index (`MEMORY.md`) and per-topic files.

## Current state

**Nothing running. v18 FAILED CLEARLY — reverted to v16@449 as baseline
(2026-07-19).** v18 (300 batches from v16@449, `--tb-value-aux-weight 0.5
--tablebase-terminate-prob 0.0` on top of v16's recipe) was the first real
use of the tablebase value-head auxiliary loss (see mechanism below). Result,
unlike v16/v17's close calls, is a clear and statistically significant
regression on every measurement (101 games each, temp 1.0):
- v18-raw vs v16-raw: 12W/46D/43L → **−110 Elo, CI [−181,−39]**
- v18-search(β=2) vs v16-search(β=2): 15W/29D/57L → **−154 Elo, CI [−228,−79]**
- v18-raw vs engine: 4W/10D/87L → 8.9% (v16-raw was 12.4%)
- v18-search(β=2) vs engine: 5W/6D/90L → 7.9% (v16-search was 19.8%, the
  best-to-date figure it was supposed to build on)

Warning signs were visible *during* training and should have been weighted
more: vs-random declined monotonically all run (98→96→92→86), unlike v16
(floor 95) or v17 (flat 97) — the lowest and only strictly-declining pattern
seen across any run so far. Draw counts in the H2H (46/101, 29/101) are also
far above the typical 15–30 range, consistent with the model becoming
generally more passive, not just better at conversion.

**Working diagnosis (unconfirmed — would need the conversion testsuite to
verify properly)**: `tb_value_aux_weight=0.5` is co-equal in magnitude with
`critic_loss_weight=0.5`, so the auxiliary loss may have fought the ordinary
GAE-based critic loss hard enough to destabilize the shared trunk generally
(not just improve endgame values) — compounded by simultaneously setting
`tablebase_terminate_prob=0`, so far more long, hard-to-resolve endgames fed
into that unstable signal. Two aggressive changes landing on the value head
at once, with no isolation between them.

**Resolution**: per user's stated fallback ("if it gets worse we'll revert
the most recent changes"), v16@449 is the working baseline again. The
TB-aux-loss code stays in the repo (dormant, `--tb-value-aux-weight` defaults
to 0) rather than being deleted — it may be worth retrying at a much lower
weight (e.g. 0.05–0.1) with `tablebase_terminate_prob` left nonzero, but not
without the conversion testsuite to actually diagnose what went wrong first.
Do not reuse `--tb-value-aux-weight 0.5` / `--tablebase-terminate-prob 0.0`
as configured in v18.

**v17 (2026-07-18, 300 batches from v16@449, plain continuation, no new
mechanism) did NOT improve on v16 — first "no gain" signal on this recipe.**
H2H suite (101 games each, temp 1.0):
- v17-raw vs v16-raw: 39W/15D/47L → −28 Elo, CI [−96,+40]
- v17-search(β=2) vs v16-search(β=2): 34W/27D/40L → −21 Elo, CI [−88,+47]
- v17-raw vs engine: 8W/13D/80L → 14.4% (v16-raw was 12.4%, a wash)
- v17-search(β=2) vs engine: 7W/10D/84L → **11.9%, down from v16's 19.8%**

Three of four measurements lean negative (none individually significant),
despite v17's in-training cumulative vs-SF counter looking *better* than
v16's (28.6% vs 21.3%) — another data point that the counter doesn't
reliably track raw strength. Per the train-to-ceiling strategy, this is the
signal to try a new mechanism rather than another plain continuation — v18
(above) is that attempt. **v16@449 remains the strongest known checkpoint.**

**Tablebase value-head auxiliary loss** (`--tb-value-aux-weight`, implemented
2026-07-18, first real run is v18 above): user's insight was that a
not-yet-converged policy's own rollout returns only teach the critic what it
*currently* achieves in an endgame, not what's *achievable* — so the critic's
baseline can silently sink to match repeated failed conversions, shrinking
exactly the advantage signal that should be punishing them. Fix:
`_tb_value_target` (train.py) supervises the value head directly toward
`probe_wdl` ground truth (+1/0/−1, cursed outcomes collapsed to 0) at every
visited ≤5-man position, via a loss term completely separate from the
reward/return stream — the actor never sees an injected reward, only a
better-calibrated critic baseline, so a failed conversion now produces a
correctly large negative advantage through the ordinary PPO mechanism rather
than a hand-fed penalty. Pairs with `--tablebase-terminate-prob 0` (also
newly exposed via CLI, was hardcoded 0.25) so the actor's reward stream
carries zero artificial TB-injected signal. Smoke-tested clean before v18
(2 batches, weight=0.1, terminate-prob=0: `TBaux: 0.2940` appeared in the log
line as expected, no crashes). v18 uses weight 0.5 (roughly critic-loss-weight
scale, a first guess — no tuning done).

**v16 (2026-07-17/18) is the current strongest / promoted baseline** —
450 batches from v15-search@149 (same search-in-training recipe) plus the
new `--dtz-shaping-weight 0.15` (dense reward inside confirmed <=5-man
tablebase wins for shrinking DTZ; see "DTZ conversion shaping" below).
Verdict was directionally positive but not individually significant on any
one measurement — promoted anyway because every measurement pointed the
same way:
- v16-raw vs v15-search-raw: 40W/25D/36L, +14 Elo, CI [−54,+82]
- v16-search(β=2) vs v15-search-search(β=2): 40W/32D/29L, +38 Elo, CI [−30,+106]
- v16-raw vs engine: 12.4% (8W/9D/84L)
- **v16-search(β=2) vs engine: 19.8% (12W/16D/73L) — best engine score to
  date**, and the first wrapper result with a balanced color split (6W/6L),
  unlike earlier wrapper runs' white-heavy skew. Since v16 was trained *with*
  search (unlike v15-search's raw-trained policy being evaluated with an
  unfamiliar wrapper), the wrapper being additive again here tracks: training
  and evaluation conditions now match.
- Caveat unresolved: v16 bundled continuation + DTZ shaping in one run, so a
  gain doesn't isolate DTZ's contribution; no conversion-specific testsuite
  exists yet to check directly. Still on the to-do list (see Next steps).

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

**Strongest known model: v16@449, played with the search wrapper (α=1,
β=2) — best engine score to date (19.8%), directionally ahead of v15-search
on both raw and wrapped H2H (neither individually significant). See Current
state.** Unlike v15-search, the wrapper helps v16 rather than being a wash —
sensible since v16 was itself trained with search. `evaluate_vs_model.py` has
`--raw-a/--raw-b/--value-weight`, `evaluate_vs_engine.py` has
`--value-weight`/`--lookahead-alpha` (default now the policy-scale formula).

- **v11** = two changes over v10: (1) **conv policy head** (`ConvPolicyHead` in `models.py`: 3×3 conv → GN → ReLU → 1×1 conv to 73 planes, square-major reshape; 157k params vs the legacy dense head's 603k), and (2) **fresh lineage** — seeded from `pretrained_conv.pt` (two 3-epoch PGN passes over the lichess elite files; 42.2% top-1 move accuracy vs 37.5% for the dense head on the identical recipe), *not* warm-started from v10. Trained 2026-07-12/13, 400 batches of the v10 recipe, clean exit.
- **H2H run 2026-07-13** (`logs/h2h_v11-399_vs_v10-399_*.log`), 101 games, temp=1.0, k=4, α=0.3:

  **v11@399 vs v10@399: 43W / 47D / 11L → score 65.8%, +114 Elo, 95% CI [+66, +166], LOS ≈ 1.0**

  One 400-batch run from a supervised seed beat the whole five-generation v4→v10 lineage. The result reads on {conv head + fresh seed} jointly, but the confound cut against v11, so the head/seed upgrade is the natural cause.
- Prior baseline for reference: v10@399 was +137 Elo over v6@399 (45W/49D/7L, 2026-07-12, `logs/h2h_v10-399_vs_v6-399_*.log`).
- Checkpoint compatibility: `net_from_state_dict()` auto-detects dense vs conv heads, so all old checkpoints still load at full fidelity (opponent pool works with mixed architectures).

**Caution for future runs:** v10's and v11's within-run signals (vs-random, vs-SF counter) are saturated/flat and badly understate cross-run progress. Don't kill a run on flat proxies alone; always settle rank with H2H vs the previous baseline.

## Next steps

Train-to-ceiling strategy (user, 2026-07-18): keep launching continuations
from the winning checkpoint until a generation's H2H shows no gain, to find
this architecture/recipe's limit. v17 hit that signal, and v18's attempt at
a new mechanism failed clearly (see Current state) — **v16@449 is the
working baseline again**. Candidates, in rough priority order:

1. **Conversion testsuite** (still not built, now higher priority): a set of
   TB-won positions with known DTZ, scored by how often/fast the model
   actually converts. Needed before retrying the TB value-aux loss at a
   lower weight — v18 failed clearly enough that blind retuning isn't
   advisable; diagnose first.
2. **If revisiting the TB aux loss**: try a much lower weight (0.05–0.1, not
   0.5) and change only *one* variable at a time — either the aux loss or
   `tablebase_terminate_prob=0`, not both together. v18 bundled both, so we
   don't know which one (or the combination) caused the regression.
3. **Structural change instead of another loss-function tweak**: this
   recipe (search-in-training + fixed-formula opponents, current network
   architecture) has now failed to improve twice in a row (v17 plain
   continuation, v18 new loss term) — bigger/different network capacity,
   deeper search (real multi-ply, not 1-ply value lookahead), or revisiting
   distillation (dormant since v5/v7/v8 failures, but under a fixed-formula
   behavior policy the original failure mode — near-uniform b — no longer
   applies, unlike when it was last tried) are the remaining candidates if
   item 1/2 doesn't pan out.
4. **Investigate the v15-control-arm regression** (−63 Elo vs v14, CI
   [−131,+6]): plain PPO continuation against the harder shared curriculum
   (opponents now fixed-formula, stronger) may have actively hurt rather than
   merely plateaued. Not blocking, but worth understanding before assuming
   future pure-PPO continuations are simply "flat."
5. **Eval hygiene (LOW PRIORITY — user deprioritized 2026-07-17)**: seed +
   paired opening suite + PGN output in `evaluate_vs_model.py`. v16/v17's
   close calls would benefit from paired openings; v18's regression was
   large enough not to need it, but future close calls will.
   resolve these close calls instead of accumulating more ties.
6. **(LOW PRIORITY, not yet built) Diagnose the training-vs-standalone
   engine-score gap directly**, per user 2026-07-19: call `generate_batch`
   with a real checkpoint + engine opponent, batch_size=32, several batches
   — training's actual code path (4-process `EnginePool` under
   `ThreadPoolExecutor`, no PPO update) — and compare its outcome rate to a
   standalone `evaluate_vs_engine.py` match. If it reproduces something near
   the in-training cumulative counter, that confirms the engine-pool's
   concurrency/contention (not GPU/CPU sharing with the main training
   process, which was tested and ruled out — see the gap-tracking note
   below) as the driver. Keep tracking the gap in the meantime (see below);
   only build this if the gap becomes something we need to act on (e.g. to
   decide whether the in-training counter can ever be trusted as a ranking
   signal).

**Standing watch: training-time vs-engine counter is running well ahead of
standalone `evaluate_vs_engine.py` scores, gap only partially explained
(2026-07-19 investigation)**. v16: training counter 21.3% vs standalone best
19.8% (β=2) — close, within noise. **v17: training counter 28.6% vs
standalone best only 14.4%** — a large, mostly unexplained gap. Tested and
ruled out / only-partial:
- Value-weight mismatch (training used β=1, my initial "best" standalone
  tests used β=2): re-tested v16/v17 at the exact trained β=1 — v16 got
  17.3% (between raw 12.4% and β=2's 19.8%, doesn't close the gap for v16
  either); **v17 got 9.9%, actually the lowest score yet** — ruled out as
  the driver.
- Opening-book randomization (training's `generate_batch` starts 60% of
  games from one of 66 named theory lines via `opening_prob=0.6`;
  `evaluate_vs_engine.py` always started from the standard position only) —
  now fixed via `--opening-prob` (imports `_start_position` from `train.py`).
  Re-tested v17 at β=1 + `--opening-prob 0.6`: **14.4%, up from 9.9%** — a
  real +4.5pp contribution, but far short of closing a ~14pp gap.
- Truncated-game silent exclusion (`max_plies=600` games neither win/loss/
  draw get dropped from the reported %, only `T` internally) — checked
  actual log ply counts for engine batches: max seen was 430, well under the
  cap. Ruled out, not happening.
- Shared CPU/GPU power budget (Ryzen AI Max APU) throttling Stockfish during
  training — the re-test above happened *while v18 was training* (same
  GPU-contention condition), and still didn't reproduce anything close to
  28.6%, which argues against this being the dominant mechanism (though
  doesn't fully rule it out as a smaller contributor).
- **Leading remaining hypothesis (untested, item 6 above)**: training's
  engine opponent runs through a 4-process `EnginePool` handling up to 32
  concurrent board requests per ply via `ThreadPoolExecutor` — a much
  heavier contention pattern (4 Stockfish processes competing with each
  other) than `evaluate_vs_engine.py`'s single serial engine playing one
  game at a time. Untested pending the diagnostic script above.
- **Practical implication for now**: don't trust the in-training vs-SF
  counter as a ranking signal on its own (already flagged elsewhere, e.g.
  v17's counter read *better* than v16's while H2H said v17 was worse) —
  always confirm with a standalone match. Keep watching whether the gap
  widens/narrows across future runs; if it stays large and this becomes
  decision-relevant, build the item-6 diagnostic.

AMD runtime note: the host is a Radeon 8060S (`gfx1151`). `venv-rocm` contains
PyTorch 2.11.0 + ROCm 7.13. The packaged MIOpen wheel has no readable gfx1151
FindDb, so runs use `MIOPEN_FIND_MODE=FAST`. AMP is disabled because an actual
v11 forward/backward smoke produced NaN FP16 gradients; FP32 gradients are
finite.

## Recipe history (most recent first)

| Run | Init | Recipe change | Result |
|---|---|---|---|
| v18 (FAILED, clear regression) | v16@449 | Same recipe + tablebase value-head auxiliary loss (`tb_value_aux_weight=0.5`) and `tablebase_terminate_prob=0` (always play out endgames); 300 batches, eval every 100 | H2H **12W/46D/43L vs v16-raw** (−110 Elo, CI[−181,−39]); search(β=2) **15W/29D/57L** (−154 Elo, CI[−228,−79]); engine raw 8.9% (down from 12.4%), engine search(β=2) 7.9% (down from 19.8%) — all four measurements clearly negative, vs-random declined monotonically all run (98→96→92→86); reverted to v16@449 as baseline |
| v17 (no gain, first ceiling signal) | v16@449 | Straight continuation, same recipe (search-in-training + DTZ shaping 0.15), 300 batches, eval every 100 | H2H **39W/15D/47L vs v16-raw** (−28 Elo, CI[−96,+40]); search(β=2) **34W/27D/40L** (−21 Elo, CI[−88,+47]); engine raw 14.4% (wash); engine search(β=2) **11.9%, down from v16's 19.8%** — 3/4 measurements negative despite a better in-training vs-SF counter (28.6% vs 21.3%) |
| **v16 (promoted, marginal/directional +14 to +38 Elo)** | v15-search@149 | Same search recipe + `--dtz-shaping-weight 0.15` (new: DTZ conversion progress shaping); 450 batches, eval every 150 | H2H vs v15-search: raw +14 Elo CI[−54,+82]; search(β=2) +38 Elo CI[−30,+106] — neither significant alone, all measurements agree in direction; **engine (β=2) 19.8%, best to date**; continuation+DTZ bundled, not isolated |
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

### `evaluate_vs_engine.py`
- `--opening-prob` (2026-07-19): imports `_start_position` from `train.py` so
  standalone matches can replicate training's 60%-book-opening mix instead
  of always starting from the standard position. Added specifically to
  investigate the training-vs-standalone engine-score gap (see Next steps);
  default 0.0 preserves old behavior.

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
- **Tablebase value-head auxiliary loss** (`--tb-value-aux-weight`, default
  0.0, off): `_tb_value_target` returns ground-truth +1/0/−1 (cursed
  outcomes collapsed to 0) for *every* visited ≤5-man position, not just
  clean wins — unlike the DTZ potential, this is defined on draws and losses
  too. Recorded per trajectory step alongside the existing search/distill
  fields (trajectory tuples now 8 elements; `generate_batch` returns a 10-tuple
  including `all_tb_targets`), duplicated for the mirror augmentation (the
  target is file-flip invariant). Applied as `F.mse_loss` against the
  network's own value-head output, masked to only the entries with a real
  (non-NaN) target, added to `total_loss` with its own weight — entirely
  separate from `critic_loss` (which still targets GAE returns) and from the
  reward stream (no term here ever touches `white_rewards`/`black_rewards`).
  Meant to pair with `--tablebase-terminate-prob 0` (now CLI-exposed, was
  hardcoded 0.25): with terminate-prob at 0, the actor's reward stream
  carries zero artificial TB signal, and this loss is the sole channel
  correcting the critic. Logged as `TBaux:` in the per-batch print line.
  Smoke-tested only (2 batches) — not yet used in a real training run.

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
