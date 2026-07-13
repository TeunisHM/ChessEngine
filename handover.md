# Handover — ChessEngine PPO training

Last updated: 2026-07-12. See `~/.claude/projects/-var-home-eunis-Python-ChessEngine/memory/` for the auto-loaded memory index (`MEMORY.md`) and per-topic files.

## Current state

Nothing running. **v11 is the strongest known model and the new generational baseline.**

- **v11** = two changes over v10: (1) **conv policy head** (`ConvPolicyHead` in `models.py`: 3×3 conv → GN → ReLU → 1×1 conv to 73 planes, square-major reshape; 157k params vs the legacy dense head's 603k), and (2) **fresh lineage** — seeded from `pretrained_conv.pt` (two 3-epoch PGN passes over the lichess elite files; 42.2% top-1 move accuracy vs 37.5% for the dense head on the identical recipe), *not* warm-started from v10. Trained 2026-07-12/13, 400 batches of the v10 recipe, clean exit.
- **H2H run 2026-07-13** (`logs/h2h_v11-399_vs_v10-399_*.log`), 101 games, temp=1.0, k=4, α=0.3:

  **v11@399 vs v10@399: 43W / 47D / 11L → score 65.8%, +114 Elo, 95% CI [+66, +166], LOS ≈ 1.0**

  One 400-batch run from a supervised seed beat the whole five-generation v4→v10 lineage. The result reads on {conv head + fresh seed} jointly, but the confound cut against v11, so the head/seed upgrade is the natural cause.
- Prior baseline for reference: v10@399 was +137 Elo over v6@399 (45W/49D/7L, 2026-07-12, `logs/h2h_v10-399_vs_v6-399_*.log`).
- Checkpoint compatibility: `net_from_state_dict()` auto-detects dense vs conv heads, so all old checkpoints still load at full fidelity (opponent pool works with mixed architectures).

**Caution for future runs:** v10's and v11's within-run signals (vs-random, vs-SF counter) are saturated/flat and badly understate cross-run progress. Don't kill a run on flat proxies alone; always settle rank with H2H vs the previous baseline.

## Next steps

1. **v12**: init from `models/ppo_search_v11_checkpoint_399.pt` (same conv architecture now, so warm start is full-fidelity), keep the validated recipe. Tests whether within-lineage PPO stacks on the new head.
2. H2H new model vs **v11@399** — that's the bar now.

## Recipe history (most recent first)

| Run | Init | Recipe change | Result |
|---|---|---|---|
| **v11 (succeeded, +114 Elo)** | `pretrained_conv.pt` (fresh PGN seed) | **Conv policy head**; fresh lineage; recipe hparams unchanged from v10 | H2H **43W/47D/11L vs v10@399** (LOS ≈ 1.0); **current generational baseline** |
| v10 (succeeded, +137 Elo) | v9@249 | `opponent_ratio` 0.6 → **0.45**; `distill_weight` 0.05 → **0.025**; `batch_size` 24 → 32; `tablebase_terminate_prob` 0.33 → 0.25 | H2H **45W/49D/7L vs v6@399** (LOS ≈ 1.0) |
| v9 (killed @ b249) | v8@249 | `opponent_ratio` 0.4 → **0.6**; added engine W/D/L tracking (per-batch + cumulative) | Draws climbed 1→7 by b250 (late-onset v7 signature); killed, v10 launched from v9@249 |
| v8 (killed @ b250, possibly competitive) | v6@399 | Outcome-filtered distill (positive-advantage only); `distill_weight=0.05`; `temperature=1.0` | Own eval 98/2/0; v9's baseline (= v8@249 weights) scored 99/1/0 — statistically tied with v6@399 on vs-random. **H2H never run, true rank vs v6 unknown.** See `project_v8_plateaued.md` |
| v7 (failed, killed @ b50) | v6@399 | `distill_weight=0.05` (no outcome filter); `temperature=1.0` | Draws spiked 2→11 at first eval; same v5-style failure mode at lower magnitude; see `project_v7_failed.md` |
| **v6 (succeeded, +70 Elo)** | v4@599 | Widened lookahead `top-k(π) ∪ captures ∪ checks`; search only vs checkpoints; `distill_weight=0.0`; `temperature=1.25` | H2H **32W/57D/12L vs v4@599** (LOS ~99.9%); vs-random draws 5 → 1; **current generational baseline** |
| v5 (failed) | v4@599 | Heavy explicit distillation (`distill_weight=0.4`, search rollouts everywhere) | Aborted @ b149 after H2H 0W/33D/37L (~−100 Elo regression); distill drowned PPO outcome signal |
| v4 (historical baseline) | v3@199 | First PPO+search+light-distill recipe | +70 Elo over v3 cross-run, flat within-run |

## Code state — what's locked in (don't change without reason)

### `lookahead.py`
- `select_moves_with_lookahead`: **widened candidate set** = top-k(π) ∪ legal captures ∪ legal non-capture checks. Returns 10-tuple including `log_b_chosen`, `log_b_topk` for IS / distill.
- `select_moves_from_policy`: stores `log_pi` under the **actual sampling temperature** (`log_softmax(masked/T)`), so PPO old_log_prob matches the sampling distribution. T=1.0 in current config so this is mathematically neutral, but the fix matters if T≠1 is reintroduced.
- Quiescence has in-check evasion handling (stand-pat skipped in check; recursion floor at depth ≤ 0).

### `train.py`
- **Per-batch `trainee_search_now`**: True only when `source.startswith("checkpoint")`. Pure π in self-play and engine batches. Master flag `trainee_search=True`.
- **Outcome-filtered distillation** (PPO loop): `pos_mask = (mb_adv > 0).float()`; `distill_loss = (distill_per_state * pos_mask).sum() / n_pos`. Only positive-advantage states contribute → aligns distill gradient with PPO actor (both advantage-weighted). Currently `distill_weight=0.05` with `pos_mask` enabled.
- **Schulman-k3 KL diagnostic**: `(exp(log_r) - 1 - log_r).mean()`. Non-negative, makes `target_kl=0.015` early-stop functional.
- **Padding** of variable-length `topk_idx` / `log_b_topk` to rollout-global max before stacking — prevents crash when `distill_weight > 0`.
- **CSV log hparam header**: first line of `logs/{model}_*.csv` is `# k1=v1 k2=v2 ...` of all hparams (parsers should skip lines starting with `#`).
- **Per-batch trainee outcomes**: `generate_batch` returns `stats["trainee_outcomes"] = {"W":, "D":, "L":, "T":}` for non-self-play batches. Printed inline; engine batches get cumulative running totals.

### Active hparams in `__main__` (the validated recipe, unchanged since v10)
- `model_name="ppo_search_v11"`, `init_from="pretrained_conv.pt"` (bump both for v12)
- 400 batches, eval every 50
- `gamma=0.98`, `gae_lamb=0.95`, `entropy_weight=0.005`, `batch_size=32`, `opening_prob=0.6`
- `temperature=1.0`, `ppo_clip_ratio=0.2`, `target_kl=0.015`, `ppo_minibatch_size=256`
- `opponent_ratio=0.45`, `engine_ratio=0.1`, `engine_skill_level=0`, `engine_move_time=0.01`
- `step_penalty=0.001`, `draw_penalty=0.1`, `material_shaping_per_pawn=0.025`
- `lookahead_k=4`, `lookahead_alpha=0.3`, `trainee_search=True`, `distill_weight=0.025`
- `tablebase_terminate_prob=0.25`
- Optimizer: AdamW, lr=5e-5, cosine schedule to 10%

## Models folder

After the conservative cleanup on 2026-05-18, `models/` contains v4 (9 checkpoints 99–899), v6 (8 checkpoints 49–399), v8 (5 checkpoints 49–249), `pretrained-76-0.pt`, plus v9's checkpoints as they save. `_load_random_opponent` reads any `.pt` here — these are the curriculum.

**Strongest known models**: v11@399 (proven baseline, +114 Elo over v10@399), then v10@399 (+137 over v6@399), then v6@399. The opponent pool holds v4/v6/v8/v9/v10/v11 checkpoints plus `pretrained-76-0.pt`; mixed head architectures all load via `net_from_state_dict`.

## Open questions (still relevant even after v10's gain)

- **Search structure is the deep limit**: top-k by π + 1-step quiescence is much weaker than real MCTS / alpha-beta. Discussion in conversation reached: MCTS-lite (~1 week) or Stockfish-teacher distillation (~2× compute, gives up self-play purity). See conversation context if pivoting.
- **The distillation magnitude trap**: even outcome-filtered, distill gradient is unbounded while PPO actor is clip-truncated in checkpoint batches (clip frac 0.78+). The asymmetry is structural. If we keep distillation, also clipping the distill gradient (or per-state weight clipping by advantage magnitude) might help.
- **vs-random is saturated, and the engine W/D/L counter turned out insensitive too**: v10's cumulative vs-SF stayed flat (2W/16D/1320L) while the model gained +137 Elo cross-run. Neither proxy tracks real progress; H2H vs the previous baseline is the only trustworthy strength measurement.

## Files modified in working tree (uncommitted)

- `lookahead.py` (widened search + sampling-temperature log_pi)
- `train.py` (per-batch trainee_search, outcome-filtered distill, padding fix, engine W/D/L counter, k3 KL, CSV hparam line)
- Many supporting files (`models.py`, `pretrain_*`, `evaluate_*`) modified earlier; see `git diff`.
- Untracked: `lookahead.py`, `evaluate_vs_model.py`, `pretrain_from_puzzles.py`, `pretrain_from_tablebase.py`, `pretrain_and_train.sh`, `handover.md`.
- Deleted (legacy): `evaluate_model.py`, `mcts.py`, `search.py`.

Last working commit: `d5f16b8 Add MCTS training and strengthen PPO self-play pipeline` — pre-dates everything v5 onward. Probably worth committing the current working tree once v9's results are in.
