# Handover — ChessEngine PPO training

## Currently running

Nothing. v4 finished: 900 batches, final ckpt `models/ppo_search_v4_checkpoint_899.pt`. Next run will exercise the distillation feature (see code changes).

## Eval trajectory vs random (CSV `logs/ppo_search_v4_20260509_203042.csv`)

| batch | W/D/L |
|---|---|
| 0 | 89/11/0 |
| 100 | 89/11/0 |
| 200 | 92/8/0 |
| 300 | 92/8/0 |
| 400 | 97/3/0 ← peak |
| 500 | 92/8/0 |
| 600 | 91/9/0 |
| 700 | 92/8/0 |
| 800 | 88/12/0 |
| 900 | 89/11/0 |

Flat-with-a-bump. **Within-run h2h matrix on adjacent checkpoints (51 games each, temp=1)** confirms no statistically significant Elo gain: cumulative drift 99→899 is +15 wins net across 8 pairs, z≈1.2 (inside noise). The 599→699 transition lost 6 wins. See `memory/project_v4_within_run_flat.md`.

Search-on vs raw-π argmax on ckpt 899 (300 games vs random): search 91.7% vs raw 77.7% → +14.0pp, z≈4.86, p<0.0001. The value head + quiescence is genuinely picking better moves. Search is a real teacher.

## Code changes since project memory was last updated

### `lookahead.py`
- `select_moves_with_lookahead` now returns a **10-tuple**: `(chosen, log_pi, masks, root_values, states, log_b_chosen, kl_b_pi, pick_rank, topk_idx, log_b_topk)`. The last two are new — the search support and its log-probabilities — needed for the distillation loss. With temperature≤0, `log_b_topk` is a one-hot row on the argmax (entries other than the picked are `-1e9`). All `evaluate_vs_*.py` call sites use `idxs, *_ =` so they're unaffected.
- `_quiesce_ab` in-check branch fixes two bugs:
  - Stand-pat is illegal in check → skipped in the in-check path. Searches all legal evasions (captures MVV-LVA first, then quiet) instead of captures+checks-only.
  - In-check recursion floor at `depth <= 0` (was `<= -2`). Returns `net(state)` evaluation when depth runs out.

### `train.py`
- `generate_batch` got `trainee_search`, `trainee_top_k`, `trainee_alpha` params. When `trainee_search=True`: trainee uses `select_moves_with_lookahead` instead of `select_moves_from_policy`; records `log_b_chosen` as `old_log_prob` so PPO IS ratio is `π_new(a|s) / b(a|s)`. Accumulates per-rollout diagnostics (π↔search KL, disagree rate, mean rank, Δlogp@a).
- PPO loop now keeps **two** old log-prob tensors:
  - `pi_old_lp_t` — always log π_old, used for `approx_kl` early-stop diagnostic
  - `old_lp_t` — log b when `trainee_search=True`, else log π_old, used as IS denominator
  - This separation was load-bearing: the early-stop was firing on epoch 1 every batch in v1 because `(log_b - log_pi_new).mean()` measured KL(b‖π), not policy update drift.
- `generate_batch` got `tablebase_terminate_prob` (default 1.0 = current behavior). Per-game `tb_terminate[gid]` decided at game start; TB termination block skips games where it's False.
- `train_actor_critic` threads `tablebase_terminate_prob` to `generate_batch`.
- `__main__` block: `model_name="ppo_search_v4"`, `init_from="models/ppo_search_v3_checkpoint_199.pt"`, `ppo_clip_ratio=0.33`, `lookahead_k=4`, `lookahead_alpha=0.33`, `trainee_search=True`, `tablebase_terminate_prob=0.5`.
- **NEW: KL distillation toward search** (`distill_weight`, default 0.0 in the function signature; set to **0.08 in `__main__`** for the next run). When enabled, the PPO update adds `-Σ_a b(a) · log π_new(a)` over the top-k search support — soft cross-entropy = `KL(b‖π_new) − H(b)`. Since `b` is fixed, gradient is identical to KL. `generate_batch` now also returns `all_topk_idx` and `all_log_b_topk`; these are mirror-augmented (`mirror_perm[topk_idx]`, log_b_topk invariant). The per-batch log line gains a `Distill:` field when the term is active.
- **Why this was added**: within-run h2h matrix showed the 900-batch v4 run produced no measurable Elo gain (all adjacent pairs inside noise). Disagree rate stayed flat at 68-70% throughout the run — the PPO signal in self-play is zero-mean (zero-sum game, same network both colors) so π wasn't being pulled toward the search picks. Distillation is the outcome-independent gradient that breaks the symmetry. See `memory/project_v4_within_run_flat.md`.

### `evaluate_vs_engine.py`, `evaluate_vs_random.py`, `evaluate_vs_model.py`
- All call sites use `idxs, *_ = select_moves_with_lookahead(...)`.

## How to resume

1. **Check status**: `grep -E "Eval at batch" ppo_search_v4.log` and `pgrep -af train.py`.
2. **Wait for completion or pick a checkpoint**: `models/ppo_search_v4_checkpoint_*.pt` lands every 100 batches.
3. **Compare runs**: `python evaluate_vs_model.py --model-a models/ppo_search_v4_checkpoint_899.pt --model-b models/ppo_search_v3_checkpoint_199.pt` (head-to-head over 101 games).
4. **Sanity-check vs random**: `python evaluate_vs_random.py -m models/ppo_search_v4_checkpoint_899.pt -g 100`.

## Open questions / what to watch for

- **Distillation run (next)**: with `distill_weight=0.08`, watch the disagree-rate metric — it should *drop* over training as π is pulled toward b. If it stays at 68-70% like v4, distillation isn't biting (try larger weight). If it crashes to 0, weight is too aggressive and π is collapsing onto b. Mid-range (drifting to 30-50%) is the target.
- **Don't bump engine_ratio yet**: trainee currently loses ~100% of engine games. More engine batches would just feed pessimistic critic V≈-1, poisoning self-play baselines further. Revisit only once distillation produces decisive wins against Skill 0.
- π↔search KL trend: at start of v3 was ~0.42, drifted to 0.27–0.40 by b200. In v4 it sat flat at ~0.35 throughout — desired-signal decay did NOT happen, which is the disagree-rate story from a different angle. Distillation is the proposed fix.
- DataGen is dominated by serial per-board quiescence. True batched quiescence (gather all per-board search frontiers into one forward pass) is the next 2–3× speedup but is a real refactor.

## Files

- Modified: `train.py`, `evaluate_vs_*.py`, `models.py`, `pretrain_from_pgn.py`
- Added: `lookahead.py`, `evaluate_vs_model.py`, `pretrain_from_puzzles.py`, `pretrain_from_tablebase.py`, `pretrain_and_train.sh`
- Deleted (legacy): `evaluate_model.py`, `mcts.py`, `search.py`
