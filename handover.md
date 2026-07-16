# Handover — ChessEngine PPO training

Last updated: 2026-07-17. See `~/.claude/projects/-var-home-eunis-Python-ChessEngine/memory/` for the auto-loaded memory index (`MEMORY.md`) and per-topic files.

## Current state

Nothing is running. v14 (300-batch warm-start from v13 control@99) completed
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

**Strongest known agents: v13-control@99-raw ≈ v14@299-raw (tied).** All
evaluation should default to raw-vs-raw (or the policy-scale search once
confirmed); `evaluate_vs_model.py` has `--raw-a/--raw-b/--value-weight`,
`evaluate_vs_engine.py` has `--raw`.

- **v11** = two changes over v10: (1) **conv policy head** (`ConvPolicyHead` in `models.py`: 3×3 conv → GN → ReLU → 1×1 conv to 73 planes, square-major reshape; 157k params vs the legacy dense head's 603k), and (2) **fresh lineage** — seeded from `pretrained_conv.pt` (two 3-epoch PGN passes over the lichess elite files; 42.2% top-1 move accuracy vs 37.5% for the dense head on the identical recipe), *not* warm-started from v10. Trained 2026-07-12/13, 400 batches of the v10 recipe, clean exit.
- **H2H run 2026-07-13** (`logs/h2h_v11-399_vs_v10-399_*.log`), 101 games, temp=1.0, k=4, α=0.3:

  **v11@399 vs v10@399: 43W / 47D / 11L → score 65.8%, +114 Elo, 95% CI [+66, +166], LOS ≈ 1.0**

  One 400-batch run from a supervised seed beat the whole five-generation v4→v10 lineage. The result reads on {conv head + fresh seed} jointly, but the confound cut against v11, so the head/seed upgrade is the natural cause.
- Prior baseline for reference: v10@399 was +137 Elo over v6@399 (45W/49D/7L, 2026-07-12, `logs/h2h_v10-399_vs_v6-399_*.log`).
- Checkpoint compatibility: `net_from_state_dict()` auto-detects dense vs conv heads, so all old checkpoints still load at full fidelity (opponent pool works with mixed architectures).

**Caution for future runs:** v10's and v11's within-run signals (vs-random, vs-SF counter) are saturated/flat and badly understate cross-run progress. Don't kill a run on flat proxies alone; always settle rank with H2H vs the previous baseline.

## Next steps

v14 ran (300 batches, seed 1401) and was flat — plain warm-start continuation
is confirmed dead as a gain source (v4/v10 pattern). Candidates, in rough
priority order:

1. **Eval hygiene before any close call**: seed + paired opening suite + PGN
   output in `evaluate_vs_model.py`, then a confirmation match of the
   policy-scale search (`--lookahead-alpha 1.0 --value-weight 2.0`) vs raw —
   +45 Elo on 101 games needs pairing to resolve.
2. **Conversion/endgame fix** (draws-with-winning-material gap): DTZ-aware
   potential shaping inside the ≤5-man domain (`probe_dtz`; dense progress
   signal that WDL lacks), or joint TB-supervised auxiliary loss on TB-domain
   rollout states only (NOT an isolated fine-tune — that regressed, see
   `project_tb_pretrain_regresses`). Build a TB-won conversion testsuite as
   the metric first.
3. **Re-insert search in training (staged; user-endorsed 2026-07-17)**. The
   v13 arm failed because old-formula b was near-uniform (72–75% of ratios
   outside clip); new-formula b ∝ π·e^(−βV) keeps ratios ~e^(±β·ΔV), so the
   pathology should shrink an order of magnitude. Stages, each gated:
   (a) **Freebie first**: v14's checkpoint *opponents* still used the broken
   α=0.3 formula (`_checkpoint_opponent_fn` inherits run lookahead args) —
   fixing opponent search to α=1/β=2 is zero-PPO-risk and hardens the
   curriculum; legitimate standalone v15 variable.
   (b) Plumb `value_weight` + trainee/opponent search flags through train.py
   CLI (v14 CLI hardcodes trainee_search=False).
   (c) Diagnostics-only: 5–10 batches, trainee search-behavior at β=1, read
   `kl_b_pi` + initial clip fraction. Gate: <~30% outside clip (v13 arm was
   72–75%), else lower β.
   (d) Matched pair from same init (v13 design): search-behavior arm vs
   pure-PPO control, 100 batches, H2H both raw-vs-raw and new-search-wrapped.
   Distillation stays off throughout (v5/v7/v8 unchanged).
4. **v15 generational run**: whatever recipe change wins above, fresh
   comparison against control@99-raw with raw-vs-raw H2H as the bar.

AMD runtime note: the host is a Radeon 8060S (`gfx1151`). `venv-rocm` contains
PyTorch 2.11.0 + ROCm 7.13. The packaged MIOpen wheel has no readable gfx1151
FindDb, so runs use `MIOPEN_FIND_MODE=FAST`. AMP is disabled because an actual
v11 forward/backward smoke produced NaN FP16 gradients; FP32 gradients are
finite.

## Recipe history (most recent first)

| Run | Init | Recipe change | Result |
|---|---|---|---|
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
- The normal entrypoint is pure PPO: raw-policy trainee actions,
  `trainee_search=False`, and `distill_weight=0.0`.
- The generic search/distillation implementation remains available to callers,
  but it is not exposed as part of the v14 CLI.
- **Per-batch PPO denominator gate**: recorded search behavior probabilities
  are used only when `trainee_search_now=True`; pure-PPO batches use `pi_old`.
- **Outcome-filtered distillation** remains dormant with weight 0.
- **Schulman-k3 KL diagnostic**: `(exp(log_r) - 1 - log_r).mean()`. Non-negative, makes `target_kl=0.015` early-stop functional.
- **Padding** of variable-length `topk_idx` / `log_b_topk` to rollout-global max before stacking — prevents crash when `distill_weight > 0`.
- **CSV log hparam header**: first line of `logs/{model}_*.csv` is `# k1=v1 k2=v2 ...` of all hparams (parsers should skip lines starting with `#`).
- **Per-batch trainee outcomes**: `generate_batch` returns `stats["trainee_outcomes"] = {"W":, "D":, "L":, "T":}` for non-self-play batches. Printed inline; engine batches get cumulative running totals.

### Prepared v14 configuration
- Required `--init-from` and `--model-name`; defaults are 400 batches, eval
  every 50, 100 eval games, and seed 1401.
- Init from v13 pure-PPO control@99; `use_se=False`.
- Radeon 8060S via `venv-rocm`; MIOpen FAST fallback; FP32 only.
- `gamma=0.98`, `gae_lamb=0.95`, `entropy_weight=0.005`, `batch_size=32`, `opening_prob=0.6`
- `temperature=1.0`, `ppo_clip_ratio=0.2`, `target_kl=0.015`, `ppo_minibatch_size=256`
- `opponent_ratio=0.45`, `engine_ratio=0.1`, `engine_skill_level=0`, `engine_move_time=0.01`
- `step_penalty=0.001`, `draw_penalty=0.1`, `material_shaping_per_pawn=0.025`
- `lookahead_k=4`, `lookahead_alpha=0.3`; `trainee_search=False`,
  `distill_weight=0.0`.
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

- `train.py`: simplified warm-start CLI, deterministic seed, pure-PPO v14
  entrypoint, and the P1 behavior-denominator correction.
- `evaluate_vs_model.py`: `--raw-a`/`--raw-b` (raw-policy sides) and
  `--value-weight` (scales net-derived quiescence values; 0 ablates learned
  eval, terminal ground truth keeps full weight); mode + vw printed in footer.
- `evaluate_vs_engine.py`: `--raw` flag, mode/skill/move-time printed in footer.
- `lookahead.py`: `value_weight` param in `select_moves_with_lookahead`
  (default 1.0 — training callers unchanged).
- All 2026-07-16/17 match logs in `logs/h2h_*_2026071{6,7}*.log` and
  `logs/engine_*_20260716.log`.
- `test_helper.py`: current 20-plane shape contract.
- `README.md` and this handover: v14 command and completed v13 results.
