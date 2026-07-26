# Chess Search Diagnostic

The raw policy outperforming the lookahead wrapper is not especially
surprising once the components are separated: the policy is considerably
stronger than the value head, while `lookahead.py` mostly reranks policy moves
using that weaker value head. It is not yet a robust chess search.

## Main diagnosis

### 1. The critic is not a valid negamax evaluator

Search assumes that a child's value can be negated:

```text
Q(s,a) = -V(child)
```

But the critic is trained on asymmetric, non-zero-sum returns: wins are
`+1.25`, losses `-1`, and draws are negative for both players
(`train.py:126`). Step penalties and shaping further break
`V(s) = -V(opponent_view(s))`.

Consequences include:

- Negating an opponent draw value of `-0.1` makes the draw look like `+0.1`
  for us.
- Longer games incur costs for both sides, but negating the opponent's future
  costs turns them into benefits.
- The critic estimates performance under a changing mixture of raw policy,
  searched policy, checkpoint opponents, and Stockfish—not objective minimax
  WDL.

Yet quiescence treats it as a bounded zero-sum evaluation
(`lookahead.py:89`). The value head is not actually bounded either; it has no
`tanh` or WDL output (`models.py:155`).

### 2. The latest critics have measurably regressed

Testing 512 random positions with exact Syzygy WDL labels produced:

| Model | Value MAE | 3-class WDL accuracy |
|---|---:|---:|
| `pretrained_big` | 0.307 | 90.6% |
| v19 | 0.505 | 75.8% |
| v20 | 0.501 | 68.8% |

V20 is significantly optimistic: exact draws average `+0.25`, while exact
losses average only `-0.30`. Thus the critic is compressed and biased precisely
where search needs reliable relative ordering.

On 192 nontrivial tablebase positions:

- Raw v19 selected an optimal-WDL move 96.4% of the time.
- Raw v20 selected one 95.3% of the time.
- Value reranking changed several moves but improved zero WDL outcomes.

The policy is already doing the job; the critic adds little.

### 3. The tactical reranker also failed to add signal

On 256 Lichess puzzles rated 1600–2200:

- The correct response was inside the widened candidate set about 89% of the
  time.
- Raw top-1 accuracy was about 31%.
- One-ply value reranking stayed flat or slightly declined.
- At `value_weight=2`, v20 fixed two raw errors but broke three raw successes.

Candidate recall is therefore not the primary bottleneck. The value ordering
is.

### 4. Quiescence barely uses the strong policy

The root uses policy top-k, but below that the search considers only captures
and checks and selects them entirely from critic values
(`lookahead.py:240`). It does not use policy priors at inner nodes and does not
explicitly search ordinary opponent replies.

Therefore:

- A quiet opponent refutation is represented only indirectly through `V`.
- Search strength is dominated by the critic rather than the policy.
- Tactical horizon effects remain despite the extra computation.

This is closer to "critic-based root reranking plus forcing-move cleanup" than
conventional lookahead.

## Concrete correctness problems

### Mate scores are inconsistent with value scaling

At `value_weight=2`, search can reject mate in one. Direct terminal moves
receive a fixed `+1`, while nonterminal learned values are multiplied by
`value_weight` (`lookahead.py:284` and `lookahead.py:298`).

With a test network returning opponent value `-0.8`, search chose `Qg8+`
instead of one of four immediate mates because `2 * 0.8 > 1`. Proven mates
should have dominant scores independent of critic scaling.

### Search discards repetition history

Every child uses `board.copy(stack=False)`. A move producing a third repetition
has repetition plane 1 with preserved history and 0 inside the searched copy.
This invalidates repetition/draw detection and creates a train/search input
mismatch.

### Timeout trajectories bootstrap from zero

Nonterminal max-ply games are treated as though their continuation value were
zero (`train.py:507`). Long games are common, so this contaminates the critic.

### Search-in-training PPO is not fully valid off-policy PPO

The search behavior has zero support outside its candidate set—and is a delta
when deterministic—while raw policy `pi` has positive probability there. The
`pi / b` ratio therefore cannot provide full importance correction, despite
the comment at `train.py:828`. PPO clipping is then applied around a behavior
distribution already different from `pi` (`train.py:865`). The v19 logs show
roughly 40% initial clip fraction in searched batches.

## Consumer-hardware issue

`quiesce_batched()` is only batched in name: it loops through boards and
performs a separate network call at every DFS node (`lookahead.py:168`). This
is particularly inefficient for a large GPU model.

The existing v19 logs show the practical result:

- Raw: 100 games in about 37 seconds.
- Search: 100 games in about 725 seconds.

That is roughly a 20x cost. An attempted full-quiescence diagnostic on only 24
positions also took over two minutes on CPU before being stopped.

## Recommended direction

### 1. Use raw policy for deployment until a paired test demonstrates a gain

Search should not be enabled merely because it performs more computation. Gate
it on repeatable paired-match evidence for the exact checkpoint and search
configuration.

### 2. Split the two value functions

Keep:

- `V_return` for PPO/GAE and shaped rewards.
- A separate search head trained only on zero-sum WDL, preferably three logits
  for win/draw/loss, using `P(win) - P(loss)` for search.

Train the search head from PGN outcomes, actual self-play outcomes, and
Syzygy. Keep a small replay fraction of supervised/tablebase positions during
RL so the search evaluator does not forget. The extra head is negligible
compared with the 11.5M-parameter trunk.

### 3. Replace recursive neural quiescence with batched selective two-ply search

A practical consumer-GPU structure would be:

```text
root:  top 4 policy moves + credible forcing moves
reply: opponent top 4 policy moves + forcing replies
leaf:  evaluate all approximately 16–30 grandchildren in one batch
score: choose the worst reply, blended with the root policy prior
```

This uses the strong policy at both plies, sees quiet replies, and keeps GPU
inference batched. It will likely be faster and stronger than hundreds of
batch-one quiescence calls.

### 4. If deeper search is desired, train a small search evaluator

Distill WDL into a small convolutional net or NNUE-like evaluator and run
alpha-beta on CPU. The large network can supply root priors while the small
evaluator handles thousands of nodes.

### 5. Fix the existing wrapper before further tuning

- Give proven mate dominant `+MATE`/`-MATE` scores at every depth.
- Preserve board history via push/pop or `copy(stack=True)`.
- Add a transposition/value cache.
- Batch leaf evaluations.
- Add SEE/delta pruning so losing captures are not all searched.
- Probe Syzygy directly at eligible leaves.
- Expose `max_qdepth` and check extension as evaluation ablations.

### 6. Repair evaluation methodology

`evaluate_vs_model.py` starts every game from the standard position, and its
default temperature is zero. With deterministic networks, 101 games can
therefore be repetitions of only two unique color assignments—not 101
independent observations.

Use 50–100 fixed opening positions, play each twice with colors reversed,
record PGNs, and compare raw versus search from exactly the same positions.
Log every search override and later classify it with a modest Stockfish
analysis.

Historical logs also show the effect is opponent-dependent: v19 search scored
much better than raw against weak Stockfish, while raw beat the pretrained
baseline more cleanly. That argues against an action-sign bug—the move
round-trip tests also pass—and strongly for an unreliable, poorly calibrated
wrapper whose occasional tactical gains do not consistently outweigh critic
mistakes.
