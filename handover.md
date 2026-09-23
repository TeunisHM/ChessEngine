# Project handover

Updated 2026-09-23. Operational baseline and retained evidence are below;
[NEXT_TRAINING_PLAN.md](NEXT_TRAINING_PLAN.md) defines the next experiment and
[IMPROVEMENTS.md](IMPROVEMENTS.md) lists verified implementation issues.

## Baseline

Use `models/ppo_search_v26_checkpoint_299.pt` as the control and initializer.
The checkpoint contains 12,173,395 parameters, 16 residual blocks, 192 channels,
and one trailing attention layer.

| Component | Parameters | Structure |
|---|---:|---|
| Policy head | 346,441 | Convolutional move logits |
| Scalar value head | 17,092 | 192-to-1 convolution, 64-to-256-to-1 MLP |
| WDL head | 682,310 | Two 192-channel 3×3 convolutions, 1-channel readout, MLP to win/draw/loss logits |

`models.py` detaches trunk features for WDL prediction by default. Training only
that head can preserve policy outputs. Freezing the policy head while updating
the shared trunk does not preserve the policy.

## Retained measurements

The local `logs/paired/` files contain three 264-game runs per checkpoint:

| Checkpoint | Wins | Draws | Losses | Score |
|---|---:|---:|---:|---:|
| v26 | 367 | 119 | 306 | 53.85% |
| v33 | 347 | 127 | 318 | 51.83% |

Sources: `v26_paired_r1.log` through `v26_paired_r3.log`, and the corresponding
`v33` files. Score is `(wins + draws / 2) / games`. These recorded results do not
establish a v33 improvement or identify the effect of architecture alone.

`logs/optionB_gate.log` records 82/9/37 and 89/8/31 W/D/L for v26 Gumbel search.
Both settings execute the same algorithm at the evaluator's default temperature
zero; this is not evidence of a deterministic-candidates advantage.

Historical run logs and Git commits remain available. Unsupported causal
explanations, universal algorithm rankings, and predictions of future Elo have
been removed from the project notes.

## Training and evaluation behavior

- PPO's scalar critic fits discounted, shaped returns. Search uses its output
  as a negamax evaluation. These are different targets.
- `train.py --policy-objective az` fits search-policy targets and game outcomes
  while updating the shared network. It does not provide persistent replay or
  the proposed Stockfish sibling-label pipeline.
- File-reflection augmentation has been removed from RL and tablebase training
  because it does not preserve standard-chess castling.
- Gumbel search normally plays its Sequential Halving winner and returns a
  completed-Q distribution as its training target. Playing results for the
  winner do not directly measure the strength of the target policy.
- Small progress evaluations are diagnostics. Compare candidates with fixed
  search settings, separate development/final openings, and recorded W/D/L.
- The conversion suite can stop on a winning zeroing move. Its aggregate success
  rate is not a checkmate-completion rate; compare against a fixed defender.

## Next action

Implement the v50 data pipeline, evaluator selection, and held-out opening suite
before launching training. The primary arm trains the existing WDL head with the
trunk and policy frozen. A capacity pilot will determine whether a wider readout
is useful; parameter count alone does not answer whether frozen features suffice.
All supervision sources are allowed. Reserve final evaluation time within the
24-hour compute budget.
