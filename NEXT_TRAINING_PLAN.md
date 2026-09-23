# v50: Stockfish-supervised search evaluation

Updated 2026-09-23. **Status: planned; implementation and training pending.**
Budget: about 24 hours on the current machine. All supervision sources are
permitted, including Stockfish, PGNs, and Syzygy.

## Decision and control

Start from `models/ppo_search_v26_checkpoint_299.pt`. Train its separate WDL head
on engine-analyzed positions and competing moves, keeping the shared trunk and
policy frozen. Test whether this improves playing strength through better search
values. This is the experiment's hypothesis, not an established Elo gain.

The current scalar critic learns shaped PPO returns, and the model already has
a detached WDL head suitable for a controlled evaluator comparison. See
[handover.md](handover.md) for checkpoint details and retained measurements.

Use quiescence search with `k=4`, `alpha=1.0`, `max_qdepth=2`, `check_budget=1`,
and temperature zero for the main comparison. Keep the control's original scalar
evaluator. Hold search settings fixed across candidates and measure inference
cost. If the new head adds material overhead, also compare at matched wall time.

## Implementation before the compute run

1. Add a resumable Stockfish labeling pipeline and a dedicated supervised trainer.
   Existing `train.py --policy-objective az` does not implement this plan.
2. Add explicit scalar/WDL evaluator selection and reuse the model's actual trunk
   path for feature caching. Preserve mate/draw handling outside the learned head.
3. Add an independent opening-suite input to the H2H harness. The current paired
   mode ignores `--games`; passing `--games 1000` cannot supply 1,000 fresh games.
4. Check label perspective, terminal outcomes, probability normalization, split
   isolation, and exact preservation of raw policy logits. Keep file reflection disabled.

## Data

Start with a 2,000-parent pilot. Subject to measured throughput, collect
30,000–60,000 parent positions and 4–8 candidate children per parent. Initial
sampling mix: 70% v26 games, 20% diverse PGN positions, 10% endgames. These are
starting settings to test, not measured optima. Sample across game phases and
outcomes, cap positions per source game, and deduplicate.

Include the policy favorite, current search move, Stockfish's best move, and
plausible tactical and quiet alternatives. Include positions at which quiescence
calls the evaluator. Keep siblings, duplicates, and their source games in one
partition; reserve separate development and final-evaluation openings. Preserve
move history or encoded repetition state, which a FEN alone cannot reconstruct.

Use full-strength Stockfish: Skill Level 20, strength limiting off, recorded
Threads/Hash settings, and fixed node budgets. Pilot approximately 20,000 nodes
for candidate discovery and 50,000–100,000 for child/MultiPV analysis. Benchmark
the actual Python client and prioritize extra analysis for unstable rankings.
Store engine/checkpoint/code identities, node counts, depths, principal variations,
centipawn or mate scores, WDL, legal actions, and label sources.

Express training WDL in the position's side-to-move perspective. Child ranking
must use the parent mover's perspective; reverse win/loss when converting between
them. Handle terminal results directly. Use Syzygy within its supported domain
with fifty-move and DTZ handling. Stockfish WDL is a soft teacher estimate, not
the probability that this network wins; see the
[Stockfish documentation](https://official-stockfish.github.io/docs/stockfish-wiki/Useful-data.html).

## Capacity pilot and training

The scalar value head has **17,092 parameters**. The primary arm uses the
**682,310-parameter WDL head**, including two trainable 192-channel 3×3
convolutions. Both eventually compress to one channel per square. Frozen features
limit what either head can learn.

Fit a small subset first to check optimization. Compare training and held-out
sibling ranking separately. If the existing WDL head underfits, test a 16- or
32-channel readout on the same pilot data. Choose using held-out ranking and
teacher-estimated loss from the selected move. If neither variant transfers,
stop this run; shared-trunk adaptation is a separate experiment because it also
changes the policy.

Primary training settings:

- Freeze trunk, policy, and original scalar head; keep frozen modules in eval mode.
- Cache trunk features through the same path as normal inference and use FP32.
- Initialize from the existing WDL head; start AdamW at LR `1e-4`, at most five
  shuffled epochs, with complete parent groups in minibatches.
- Use soft WDL cross-entropy plus sibling-ranking loss. Start with normalized
  loss weights 1.0 and 0.1; inspect gradient magnitudes. Exclude teacher near-ties
  from ranking and emphasize consequential errors.
- Stop on held-out ranking/regret, not training accuracy alone. Verify unchanged
  policy logits and, if time permits, repeat with a second shuffle seed.

Use `P(win) - P(loss)` for search. Development comparisons may include the new
value and one 50/50 blend with the original scalar. Freeze the selected setting
before final evaluation.

An optional companion trains only the policy head on the same teacher moves,
with a preservation loss toward v26. Start LR `1e-5`; review development play
around mean legal-policy KL 0.02 before taking larger steps. Screen value-only,
policy-only, and combined heads separately. Skip this companion if it reduces
the final evaluation reserve. No shared-trunk updates in this experiment.

## Compute allocation

| Hours | Deliverable |
|---|---|
| 0–2 | Record identities, check GPU/client operation, benchmark labeling and games, establish baseline |
| 2–10 | Generate and label data; save independent train/development/test partitions |
| 10–14 | Train the value head and measure held-out ranking/regret |
| 14–17 | Capacity comparison or optional policy companion; select one final candidate |
| 17–24 | Independent final games and conversion checks |

These are budget caps. Shrink the dataset or omit companion arms before taking
time from final evaluation. Do not launch a larger run if the pilot fails its
label, optimization, or held-out ranking checks.

## Evaluation and promotion

Target 1,000 candidate-versus-v26 games from 500 distinct held-out opening
positions, with colors reversed. Use independent standalone Stockfish skill-0,
10ms comparisons as a secondary check, interleaving candidate and baseline under
matched machine load. Add skill 1 only if time permits. Development games must
not enter the final comparison.

Report W/D/L, score, relative H2H Elo, elapsed time, and 95% confidence intervals
grouped by opening pair. Report checkmates separately from winning zeroing moves
in conversion tests, using a fixed defender. For the value-only arm, raw-policy
outputs must remain unchanged.

Promote only if the final H2H interval is above 50% and the independent engine
and conversion results show no material contradiction. Otherwise retain v26.
A lower supervised loss or a favorable small game sample is insufficient.
