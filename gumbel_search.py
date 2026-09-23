"""Gumbel AlphaZero search (Danihelka, Guez, Schrittwieser & Silver, ICLR 2022,
"Policy Improvement by Planning with Gumbel").

Root:  sample m candidate actions without replacement via the Gumbel top-k trick
       on the policy logits, then spend a fixed budget of `sims` simulations on
       them with Sequential Halving, ranking survivors by
       g(a) + logits(a) + sigma(completedQ(a)) and cutting the worse half.
Below: the paper's deterministic selection rule
       argmax_a [ pi'(a) - N(a) / (1 + sum_b N(b)) ]
       with pi' = softmax(logits + sigma(completedQ)).
Leaf:  one evaluation with the frozen net. No rollout to terminal -- game
       outcomes only enter later, as value-head targets from finished games.

The return contract is identical to lookahead.select_moves_with_lookahead, so the
two are drop-in interchangeable (see search_backends.select_moves).

All boards are searched in lockstep: every simulation round contributes at most
one frontier node per board and they are evaluated in a single forward.
"""
import math
from typing import List, Optional, Sequence, Tuple

import chess
import numpy as np
import torch
import torch.nn.functional as F

from helper import board_to_tensor, legal_moves_mask, move_to_index

# sigma(q) = (c_visit + max_b N(b)) * c_scale * q.
# Keep the paper's setting separate from this implementation's default.
PAPER_C_VISIT = 50.0
C_VISIT = 1.0
C_SCALE = 1.0

_LOG_ZERO = -1e9


class _Node:
    """One tree node. `logits` are log-softmax over this node's legal actions
    only, so every array here is indexed by *local* action position."""

    __slots__ = ("board", "legal", "moves", "logits", "value",
                 "N", "W", "children", "terminal", "expanded")

    def __init__(self, board: chess.Board):
        self.board = board
        self.legal: List[int] = []
        self.moves: List[chess.Move] = []
        self.logits: Optional[np.ndarray] = None
        self.value: float = 0.0
        self.N: Optional[np.ndarray] = None
        self.W: Optional[np.ndarray] = None
        self.children: List[Optional["_Node"]] = []
        self.terminal = False
        self.expanded = False

    def set_legal(self):
        for mv in self.board.legal_moves:
            idx = move_to_index(mv, self.board)
            if idx is None:
                continue
            self.legal.append(idx)
            self.moves.append(mv)
        k = len(self.legal)
        self.N = np.zeros(k, dtype=np.float64)
        self.W = np.zeros(k, dtype=np.float64)
        self.children = [None] * k


def _log_softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max()
    return z - math.log(float(np.exp(z).sum()))


def _softmax(x: np.ndarray) -> np.ndarray:
    z = np.exp(x - x.max())
    return z / z.sum()


def _completed_q(node: _Node) -> Tuple[np.ndarray, float]:
    """completedQ(a) = q(a) where visited, else v_mix (paper eq. 'Completing Q').

    v_mix = ( v_hat + (sum_b N(b) / sum_{b:N>0} pi(b)) * sum_{b:N>0} pi(b) q(b) )
            / (1 + sum_b N(b))
    """
    visited = node.N > 0
    sum_n = float(node.N.sum())
    q = np.zeros_like(node.W)
    np.divide(node.W, node.N, out=q, where=visited)
    if visited.any():
        pi = _softmax(node.logits)
        pi_visited = float(pi[visited].sum())
        weighted = float((pi[visited] * q[visited]).sum())
        v_mix = (node.value + (sum_n / max(pi_visited, 1e-12)) * weighted) / (1.0 + sum_n)
    else:
        v_mix = node.value
    return np.where(visited, q, v_mix), sum_n


def _sigma(q: np.ndarray, node: _Node, c_visit: float, c_scale: float,
           rescale: bool = True) -> np.ndarray:
    """sigma(q) = (c_visit + max_b N(b)) * c_scale * q.

    q is min-max rescaled to [0,1] across the node's actions first, as the
    reference implementation does (mctx qtransform_completed_by_mix_value,
    rescale_values=True). Without it the multiplier -- which grows with the
    visit count -- is applied to raw [-1,1] values and pi' collapses to nearly
    one-hot, which in turn makes log b(chosen) unusable as a PPO denominator.
    """
    if rescale and q.size:
        lo, hi = float(q.min()), float(q.max())
        q = (q - lo) / (hi - lo) if hi - lo > 1e-8 else np.zeros_like(q)
    max_n = float(node.N.max()) if node.N.size else 0.0
    return (c_visit + max_n) * c_scale * q


def _improved_policy(node: _Node, c_visit: float, c_scale: float) -> Tuple[np.ndarray, float]:
    """pi' = softmax(logits + sigma(completedQ)) over this node's legal actions."""
    completed, sum_n = _completed_q(node)
    return _softmax(node.logits + _sigma(completed, node, c_visit, c_scale)), sum_n


def _select_child(node: _Node, c_visit: float, c_scale: float) -> int:
    """Non-root rule: argmax_a [ pi'(a) - N(a) / (1 + sum_b N(b)) ]."""
    improved, sum_n = _improved_policy(node, c_visit, c_scale)
    return int(np.argmax(improved - node.N / (1.0 + sum_n)))


class _SequentialHalving:
    """Visit schedule over the m Gumbel-sampled root candidates.

    Hands out one root action at a time so every board can advance by exactly
    one simulation per round, whatever its own candidate count.
    """

    def __init__(self, candidates: Sequence[int], sims: int):
        self.alive = list(candidates)
        self.sims = max(1, sims)
        self.phases = max(1, math.ceil(math.log2(len(self.alive)))) if len(self.alive) > 1 else 1
        self.queue: List[int] = []
        self._refill()

    def _refill(self):
        per_phase = self.sims / self.phases
        visits = max(1, int(per_phase // len(self.alive)))
        # Round-robin order so an interrupted phase still spreads its visits.
        self.queue = [a for _ in range(visits) for a in self.alive]

    def next_action(self, rank_factory) -> int:
        """rank_factory is only invoked at a phase boundary, so the root's
        completedQ is not recomputed on every simulation."""
        if not self.queue:
            if len(self.alive) > 1:
                rank_fn = rank_factory()
                self.alive = sorted(self.alive, key=lambda a: -rank_fn(a))[:max(1, len(self.alive) // 2)]
            self._refill()
        return self.queue.pop(0)

    def winner(self, rank_fn) -> int:
        return self.alive[0] if len(self.alive) == 1 else max(self.alive, key=rank_fn)


def _root_rank(node: _Node, perturbed: np.ndarray, c_visit: float, c_scale: float):
    """g(a) + logits(a) + sigma(completedQ(a)), evaluated lazily per call."""
    completed, _ = _completed_q(node)
    scores = perturbed + _sigma(completed, node, c_visit, c_scale)
    return lambda a: float(scores[a])


def _expand(net, nodes: List[_Node], device):
    """One batched forward for every frontier node discovered this round."""
    if not nodes:
        return
    states = torch.stack([board_to_tensor(n.board) for n in nodes]).to(device)
    logits, values = net(states)
    values = values.view(-1)
    logits_np = logits.float().cpu().numpy()
    values_np = values.float().cpu().numpy()
    for i, node in enumerate(nodes):
        node.logits = _log_softmax(logits_np[i][node.legal])
        node.value = float(values_np[i])
        node.expanded = True


def _backup(path: List[Tuple[_Node, int]], leaf_value: float):
    """Negamax backup: the value flips sign at every ply on the way up."""
    v = leaf_value
    for node, a in reversed(path):
        v = -v
        node.W[a] += v
        node.N[a] += 1


def _gumbel(k: int, device) -> np.ndarray:
    """Gumbel(0,1) noise drawn through torch so --seed controls it."""
    u = torch.rand(k).clamp_(min=1e-20, max=1.0 - 1e-7)
    return (-torch.log(-torch.log(u))).numpy().astype(np.float64)


@torch.inference_mode()
def select_moves_with_gumbel(
    net,
    boards: List[chess.Board],
    device,
    *,
    m: int = 16,
    sims: int = 32,
    temperature: float = 0.0,
    c_visit: float = C_VISIT,
    c_scale: float = C_SCALE,
    deterministic_candidates: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pick a move per board with Gumbel top-k root sampling + Sequential Halving.

    Returns the same 10-tuple as lookahead.select_moves_with_lookahead:
    (action_idx, log_pi, masks, root_values, states,
     log_b_chosen, kl_b_pi, pick_rank, topk_idx, log_b_topk).

    log_b_topk is the improved policy pi' over EVERY legal action (ordered by
    descending log pi), which is the Gumbel-AZ policy target. It is not
    restricted to the m sampled candidates: doing so both changes the target and
    can collapse its mass to ~0 when the searched actions all back up below
    v_mix.

    log_b_chosen is log pi'(chosen). In Sequential Halving mode this is not
    the action-selection log probability and cannot serve as a PPO denominator.
    With deterministic_candidates and positive temperature, the action is
    sampled from pi' instead. train.py rejects gumbel + ppo in either mode.

    temperature <= 0 disables the Gumbel perturbation entirely (deterministic
    play, matching the eval scripts' greedy default). Otherwise the logits are
    divided by the temperature before the noise is added, so the root sample is
    drawn from softmax(logits / T). With deterministic_candidates, root
    candidates use unperturbed logits at any temperature; positive temperature
    enables sampling the played move from pi'. At zero temperature the flag
    does not change the search or played move.

    c_visit / c_scale are what weigh the learned evaluation against the policy
    prior; sigma min-max rescales completedQ, so a uniform rescaling of the
    values themselves would leave the improved policy exactly unchanged.
    """
    n = len(boards)
    states = torch.stack([board_to_tensor(b) for b in boards]).to(device)
    masks = torch.stack([legal_moves_mask(b) for b in boards]).to(device)

    logits, root_values = net(states)
    masked = logits.masked_fill(~masks, _LOG_ZERO)
    log_pi = F.log_softmax(masked, dim=1)
    log_pi_cpu = log_pi.float().cpu().numpy()
    root_v_cpu = root_values.view(-1).float().cpu().numpy()

    roots: List[Optional[_Node]] = []
    plans: List[Optional[_SequentialHalving]] = []
    perturbations: List[Optional[np.ndarray]] = []

    for i, board in enumerate(boards):
        node = _Node(board)
        node.set_legal()
        if not node.legal:
            roots.append(None); plans.append(None)
            perturbations.append(None)
            continue
        node.logits = _log_softmax(log_pi_cpu[i][node.legal])
        # Same scaling as the leaves in _expand: the root's own value feeds
        # v_mix, so leaving it unscaled would mix weighted and unweighted
        # estimates in one completedQ. The value returned to the caller stays
        # raw -- that is the critic's output, not a search quantity.
        node.value = float(root_v_cpu[i])
        node.expanded = True
        if deterministic_candidates or temperature is None or temperature <= 1e-6:
            # Unperturbed top-m candidates. Temperature zero already takes
            # this path without the flag. Positive-temperature exploration
            # with deterministic candidates samples pi' below.
            perturbed = node.logits.copy()
        else:
            perturbed = node.logits / float(temperature) + _gumbel(len(node.legal), device)
        m_i = min(m, len(node.legal))
        # Gumbel top-k == sampling m actions without replacement from the policy.
        candidates = np.argsort(-perturbed)[:m_i].tolist()
        roots.append(node)
        plans.append(_SequentialHalving(candidates, sims))
        perturbations.append(perturbed)

    # --- simulations, in lockstep across boards -----------------------------
    for _ in range(max(1, sims)):
        pending: List[_Node] = []
        pending_paths: List[List[Tuple[_Node, int]]] = []
        for i in range(n):
            root = roots[i]
            if root is None:
                continue
            a = plans[i].next_action(
                lambda i=i: _root_rank(roots[i], perturbations[i], c_visit, c_scale))
            node, path = root, [(root, a)]
            leaf_value = None
            while True:
                child = node.children[a]
                if child is None:
                    nb = node.board.copy(stack=True)
                    nb.push(node.moves[a])
                    child = _Node(nb)
                    node.children[a] = child
                    if nb.is_game_over(claim_draw=False):
                        child.terminal = True
                        # Side to move at the leaf is the one that got mated.
                        child.value = -1.0 if nb.is_checkmate() else 0.0
                        leaf_value = child.value
                    else:
                        child.set_legal()
                        if not child.legal:          # defensive: no legal moves
                            child.terminal = True
                            child.value = 0.0
                            leaf_value = 0.0
                        else:
                            pending.append(child)
                            pending_paths.append(path)
                    break
                if child.terminal:
                    leaf_value = child.value
                    break
                node = child
                a = _select_child(node, c_visit, c_scale)
                path.append((node, a))
            if leaf_value is not None:
                _backup(path, leaf_value)

        _expand(net, pending, device)
        for node, path in zip(pending, pending_paths):
            _backup(path, node.value)

    # --- outputs ------------------------------------------------------------
    # The target spans EVERY legal action, not just the sampled candidates.
    # Restricting it changes the target even when well behaved, and when the
    # searched actions all back up below v_mix the improved policy puts nearly
    # all its mass on the UNSEARCHED ones -- renormalising what is left then
    # gives a row summing to ~1e-11, i.e. no learning signal at all.
    # improved is already a softmax over the legal actions, so taken whole it
    # needs no renormalisation.
    max_m = max(1, max((len(r.legal) if r is not None else 0 for r in roots), default=1))

    topk_idx = torch.zeros(n, max_m, dtype=torch.long)
    log_b_topk = torch.full((n, max_m), _LOG_ZERO, dtype=torch.float32)
    chosen = torch.zeros(n, dtype=torch.long)
    log_b_chosen = torch.zeros(n, dtype=torch.float32)
    kl_b_pi = torch.zeros(n, dtype=torch.float32)
    pick_rank = torch.zeros(n, dtype=torch.long)

    for i in range(n):
        root, plan = roots[i], plans[i]
        if root is None:
            continue
        rank = _root_rank(root, perturbations[i], c_visit, c_scale)
        winner_local = plan.winner(rank)

        improved, _ = _improved_policy(root, c_visit, c_scale)
        # Order by descending log pi so pick_rank keeps its meaning across
        # backends: 0 == the policy's own favourite.
        order = sorted(range(len(root.legal)), key=lambda a: -float(root.logits[a]))
        m_i = len(order)

        b = improved[order]
        log_b = np.log(np.clip(b, 1e-38, None))

        action_indices = [root.legal[a] for a in order]
        topk_idx[i, :m_i] = torch.tensor(action_indices, dtype=torch.long)
        log_b_topk[i, :m_i] = torch.tensor(log_b, dtype=torch.float32)

        if deterministic_candidates and temperature is not None and temperature > 1e-6:
            # Sample the completed-Q target over all legal actions. Use torch
            # so the training seed also controls move sampling.
            probs = torch.from_numpy(np.asarray(b / b.sum(), dtype=np.float64))
            pos = int(torch.multinomial(probs, 1).item())
            winner_local = order[pos]
        else:
            pos = order.index(winner_local)
        chosen[i] = root.legal[winner_local]
        log_b_chosen[i] = float(log_b[pos])
        pick_rank[i] = pos
        # KL(b || pi) over the considered support, measured against the
        # full-space masked log pi -- same definition as the quiescence backend.
        log_pi_cands = log_pi_cpu[i][action_indices]
        kl_b_pi[i] = float((b * (log_b - log_pi_cands)).sum())

    return (chosen.to(device), log_pi, masks, root_values.view(-1), states,
            log_b_chosen.to(device), kl_b_pi.to(device), pick_rank.to(device),
            topk_idx.to(device), log_b_topk.to(device))
