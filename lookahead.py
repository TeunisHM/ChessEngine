"""Top-k policy lookahead with quiescence at the leaves.

For each board:
    candidates = top_k by policy prior pi(a|s)
    score(a)   = -quiesce(s_a) + alpha * log pi(a|s)

quiesce(s) is a negamax search that extends through capture moves until no
captures remain (or max_qdepth is hit), returning the side-to-move's value
under stand-pat-V at quiet leaves.

temperature <= 0 -> argmax; otherwise sample softmax(score / temperature) over
the top_k.
"""
from typing import List, Optional, Tuple

import chess
import torch
import torch.nn.functional as F

from helper import board_to_tensor, index_to_move, legal_moves_mask, move_to_index

# Proven-mate score: dominant and independent of value_weight, so a forced mate
# always outranks any learned evaluation (fixes value_weight>1 rejecting mate).
MATE_SCORE = 1000.0


def _leaf_value(net, board: chess.Board, device, use_wdl: bool) -> float:
    """Side-to-move leaf evaluation in [-1,1]. use_wdl -> P(win)-P(loss) from the
    separate WDL head (zero-sum, calibrated); else the scalar value head."""
    state = board_to_tensor(board).unsqueeze(0).to(device)
    if use_wdl:
        _, _, wdl = net(state, with_wdl=True)
        p = wdl.softmax(-1)
        return float((p[0, 0] - p[0, 2]).item())
    _, v = net(state)
    return float(v.item())


@torch.inference_mode()
def select_moves_from_policy(
    net,
    boards: List[chess.Board],
    device,
    *,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample a move per board directly from the masked policy distribution.

    Returned `log_pi` is the log-softmax of `logits / temperature`, i.e. the
    log-probs *under the actual sampling distribution*. Caller's
    `log_pi[chosen]` is therefore a correct PPO old_log_prob — the IS ratio
    π_new_T(a) / π_old_T(a) refers to the same tempered policy on both sides.
    """
    states = torch.stack([board_to_tensor(b) for b in boards]).to(device)
    masks = torch.stack([legal_moves_mask(b) for b in boards]).to(device)

    logits, root_values = net(states)
    masked = logits.masked_fill(~masks, -1e9)

    if temperature is None or temperature <= 1e-6:
        # Argmax: behavior policy is a delta on the argmax; for downstream
        # consumers we still return log softmax at T=1 (the chosen-action
        # log-prob extracted from it is the T=1 log π for that action, which
        # under a delta behavior is a benign stand-in that won't affect
        # gradient direction).
        log_pi = F.log_softmax(masked, dim=1)
        chosen = log_pi.argmax(dim=1)
    else:
        log_pi = F.log_softmax(masked / float(temperature), dim=1)
        chosen = torch.distributions.Categorical(logits=log_pi).sample()

    return chosen, log_pi, masks, root_values.view(-1), states


_QPIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
}


def _ordered_captures(board: chess.Board):
    """MVV-LVA ordered capture list. Without ordering, alpha-beta barely prunes."""
    scored = []
    for move in board.legal_moves:
        if not board.is_capture(move):
            continue
        attacker_t = board.piece_type_at(move.from_square)
        if board.is_en_passant(move):
            victim_t = chess.PAWN
        else:
            victim_t = board.piece_type_at(move.to_square)
        v = _QPIECE_VALUES.get(victim_t, 0) if victim_t else 0
        a = _QPIECE_VALUES.get(attacker_t, 0) if attacker_t else 0
        scored.append((v * 10 - a, move))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [m for _, m in scored]


def _ordered_checks(board: chess.Board):
    """Non-capturing moves that give check. Captures handled separately."""
    return [m for m in board.legal_moves
            if not board.is_capture(m) and board.gives_check(m)]


def _eval_leaf_batch(net, boards: List[chess.Board], device, use_wdl: bool) -> torch.Tensor:
    """Side-to-move leaf value in [-1,1] for many boards in a few large forwards.

    Replaces per-leaf single-sample inference (latency-bound on iGPU) with
    chunked batch evaluation — same values as _leaf_value would produce,
    computed ~100x cheaper per board.
    """
    values = torch.empty(len(boards), device=device)
    chunk = 2048
    for start in range(0, len(boards), chunk):
        states = torch.stack(
            [board_to_tensor(b) for b in boards[start:start + chunk]]
        ).to(device)
        end = start + states.shape[0]
        if use_wdl:
            _, _, wdl = net(states, with_wdl=True)
            p = wdl.softmax(-1)
            values[start:end] = p[:, 0] - p[:, 2]
        else:
            _, v = net(states)
            values[start:end] = v.view(-1)
    return values


# Hard cap on net evaluations per quiesce level. Stand-pat fail-highs keep
# real trees far below this; hitting it means a pathological capture-storm,
# where we prefer correct-but-slow over wrong-fast.
_MAX_LEVEL_EVALS = 65536


@torch.inference_mode()
def quiesce_batched(
    net,
    boards: List[chess.Board],
    device,
    max_qdepth: int = 2,
    check_budget: int = 1,
    use_wdl: bool = False,
) -> torch.Tensor:
    """Per-board quiescence via level-synchronous batched negamax with windows.

    Semantics match the previous recursive fail-soft alpha-beta search exactly
    at each root: stand-pat cutoffs (value >= beta resolves), MVV-LVA capture
    ordering, check evasions searched without stand-pat, non-capturing checks
    extended while budget remains, mate scored MATE_SCORE-dominant, and all
    leaves evaluated by the net in batches (one forward per tree level).

    Returns one value per input board (side-to-move).
    """
    if not boards:
        return torch.empty(0, device=device)

    def new_node(board, alpha, beta, depth, cbudget, parent=None):
        return {
            "board": board, "alpha": alpha, "beta": beta,
            "depth": depth, "cbudget": cbudget,
            "parent": parent, "children": [], "open": 0,
            "best": -1e18, "done": False, "cancelled": False,
        }

    roots = [new_node(b, -1e9, 1e9, max_qdepth, check_budget) for b in boards]

    def resolve(node, value):
        # Iteratively settle this node and any ancestors it finishes.
        while node is not None and not node["cancelled"]:
            if node["done"]:
                return
            node["done"] = True
            node["best"] = value
            parent = node["parent"]
            if parent is None or parent["cancelled"] or parent["done"]:
                return
            contribution = -value
            if contribution > parent["best"]:
                parent["best"] = contribution
            if parent["best"] >= parent["beta"]:
                # Fail high: remaining siblings cannot improve the ancestor.
                for c in parent["children"]:
                    c["cancelled"] = True
                value = parent["best"]
                node = parent
            elif all(c["done"] or c["cancelled"] for c in parent["children"]):
                # Last sibling settled without fail-high: parent's exact max.
                value = parent["best"]
                node = parent
            else:
                return

    pending = list(roots)
    while pending:
        eval_boards: List[chess.Board] = []
        eval_nodes = []
        to_expand = []
        next_pending = []
        for node in pending:
            if node["cancelled"]:
                continue
            board = node["board"]
            if board.is_game_over(claim_draw=False):
                resolve(node, -MATE_SCORE if board.is_checkmate() else 0.0)
            elif board.is_check():
                if node["depth"] <= 0:
                    # In check at the recursion floor: no stand-pat exists.
                    eval_boards.append(board)
                    eval_nodes.append(node)
                else:
                    to_expand.append(node)
            else:
                eval_boards.append(board)   # stand-pat decides expand vs cutoff
                eval_nodes.append(node)

        if len(eval_boards) > _MAX_LEVEL_EVALS:
            raise RuntimeError(
                f"quiesce level exceeded {_MAX_LEVEL_EVALS} leaf evals; "
                "refusing to truncate (lower max_qdepth/check_budget)"
            )
        if eval_boards:
            vals = _eval_leaf_batch(net, eval_boards, device, use_wdl)
            for node, v in zip(eval_nodes, vals.tolist()):
                if node["board"].is_check():
                    resolve(node, v)
                    continue
                if v >= node["beta"]:
                    resolve(node, v)
                elif node["depth"] <= 0:
                    resolve(node, v)
                else:
                    node["alpha"] = max(node["alpha"], v)
                    node["best"] = v
                    to_expand.append(node)

        for node in to_expand:
            if node["cancelled"] or node["done"]:
                continue
            board = node["board"]
            children = []
            if board.is_check():
                # No stand-pat when in check: search all legal evasions.
                move_sets = (_ordered_captures(board),
                             (m for m in board.legal_moves if not board.is_capture(m)))
                child_depth, child_budget = node["depth"] - 1, node["cbudget"]
            else:
                captures = _ordered_captures(board)
                if node["cbudget"] > 0:
                    move_sets = (captures, _ordered_checks(board))
                    budgets = (node["cbudget"], node["cbudget"] - 1)
                else:
                    move_sets = (captures,)
                    budgets = (node["cbudget"],)
                child_depth = node["depth"] - 1
            for set_i, moves in enumerate(move_sets):
                cb = child_budget if board.is_check() else budgets[set_i]
                for move in moves:
                    child = board.copy(stack=True)
                    child.push(move)
                    cnode = new_node(child, -node["beta"], -node["alpha"],
                                     child_depth, cb, parent=node)
                    children.append(cnode)
            node["children"] = children
            node["open"] = len(children)
            if not children:
                resolve(node, node["best"])
            else:
                next_pending.extend(children)

        pending = next_pending

    return torch.tensor([r["best"] for r in roots], device=device, dtype=torch.float32)


@torch.inference_mode()
def select_moves_with_lookahead(
    net,
    boards: List[chess.Board],
    device,
    *,
    top_k: int = 8,
    alpha: float = 0.33,
    temperature: float = 0.0,
    max_qdepth: int = 2,
    value_weight: float = 1.0,
    use_wdl: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pick a move per board via *widened* candidate set + value quiescence.

    Candidate set per board = top-k by π  ∪  all legal captures  ∪  all legal
    non-capture checks. The widening ensures forcing tactical moves are always
    in the search support even when π undervalues them — addressing the
    structural limit where top-k by π alone can't escape the policy's blind
    spots.

    Returns (action_idx, log_pi, masks, root_values, states,
             log_b_chosen, kl_b_pi, pick_rank, topk_idx, log_b_topk),
    all on `device`.

    log_pi: masked log-softmax over the full action space.
    log_b_chosen: log probability that this search procedure assigned to the
        chosen action — i.e. the *behavior* policy log-prob. Use as PPO
        old_log_prob when sampling actions through this function so the IS
        ratio π_new(a|s) / b(a|s) is well-formed.
    kl_b_pi: per-state KL(b || π) over the candidate support; how much the
        search distribution disagrees with the raw policy at that state.
    pick_rank: 0-indexed rank of the chosen action in the candidate set,
        ordered by descending log π. (0 = highest-π candidate.)
    topk_idx: (n, max_cand) action indices that make up the search support,
        padded per row. Invalid/padding slots are scored at -inf so they
        cannot be selected.
    log_b_topk: (n, max_cand) log b over those indices; with temperature<=0,
        b is a delta on the chosen action (one-hot row).
    """
    states = torch.stack([board_to_tensor(b) for b in boards]).to(device)
    masks = torch.stack([legal_moves_mask(b) for b in boards]).to(device)

    logits, root_values = net(states)
    masked = logits.masked_fill(~masks, -1e9)
    log_pi = F.log_softmax(masked, dim=1)

    n = len(boards)
    k_base = min(top_k, log_pi.shape[1])

    # Per-board candidate set: top-k(π) ∪ captures ∪ non-capture checks.
    # Order each board's candidate list by descending log π so pick_rank
    # remains interpretable (0 = π's highest-prob candidate).
    log_pi_cpu = log_pi.detach().cpu()
    candidate_sets: List[List[int]] = []
    for i, board in enumerate(boards):
        topk_idx_i = log_pi_cpu[i].topk(k_base).indices.tolist()
        cands = set(topk_idx_i)
        for move in board.legal_moves:
            if board.is_capture(move) or board.gives_check(move):
                cands.add(move_to_index(move, board))
        cands_sorted = sorted(cands, key=lambda a: -float(log_pi_cpu[i, a]))
        candidate_sets.append(cands_sorted)

    max_k = max((len(c) for c in candidate_sets), default=k_base)
    if max_k == 0:
        max_k = k_base  # all-empty edge case (game over boards); pad zeros

    topk_idx = torch.zeros(n, max_k, dtype=torch.long, device=device)
    topk_logp = torch.full((n, max_k), -1e9, device=device)
    is_invalid = torch.ones(n, max_k, dtype=torch.bool, device=device)
    for i, cs in enumerate(candidate_sets):
        if not cs:
            continue
        cs_t = torch.tensor(cs, device=device, dtype=torch.long)
        topk_idx[i, :len(cs)] = cs_t
        topk_logp[i, :len(cs)] = log_pi[i].index_select(0, cs_t)
        is_invalid[i, :len(cs)] = False

    topk_idx_cpu = topk_idx.cpu().tolist()
    neg_v = torch.zeros(n, max_k, device=device)
    child_boards: List[chess.Board] = []
    child_rows: List[int] = []
    child_cols: List[int] = []

    for i, board in enumerate(boards):
        for j in range(max_k):
            if is_invalid[i, j]:
                continue
            action_idx = topk_idx_cpu[i][j]
            move = index_to_move(action_idx, board)
            if move is None or move not in board.legal_moves:
                is_invalid[i, j] = True
                continue
            child = board.copy(stack=True)
            child.push(move)
            if child.is_game_over(claim_draw=False):
                neg_v[i, j] = MATE_SCORE if child.is_checkmate() else 0.0
            else:
                child_boards.append(child)
                child_rows.append(i)
                child_cols.append(j)

    if child_boards:
        child_values = quiesce_batched(net, child_boards, device, max_qdepth=max_qdepth,
                                       use_wdl=use_wdl)
        rows_t = torch.tensor(child_rows, device=device, dtype=torch.long)
        cols_t = torch.tensor(child_cols, device=device, dtype=torch.long)
        # value_weight scales only net-derived quiescence values; ground-truth
        # terminal entries (checkmate/stalemate children) keep full weight, so
        # value_weight=0 ablates the learned evaluation but not mate detection.
        # Forced mates found in quiescence (|value| == MATE_SCORE) also bypass the
        # scaling, so a proven mate stays dominant regardless of value_weight.
        cv = -child_values.to(neg_v.dtype)
        cv = torch.where(cv.abs() >= MATE_SCORE, cv, cv * value_weight)
        neg_v.index_put_((rows_t, cols_t), cv)

    score = neg_v + alpha * topk_logp
    score = score.masked_fill(is_invalid, -1e9)

    if temperature is None or temperature <= 1e-6:
        sel = score.argmax(dim=1)
        # Deterministic behavior: b is one-hot on argmax.
        log_b_chosen = torch.zeros(n, device=device)
        # KL(δ_a || π) = -log π(a) on the support of δ.
        kl_b_pi = -topk_logp.gather(1, sel.view(-1, 1)).view(-1)
        # log_b_topk: one-hot row on `sel` (log 1 = 0 at sel, -inf elsewhere).
        # Use a large negative instead of -inf so downstream exp gives 0 cleanly.
        log_b_topk = torch.full((n, max_k), -1e9, device=device)
        log_b_topk.scatter_(1, sel.view(-1, 1), 0.0)
    else:
        log_b_topk = F.log_softmax(score / float(temperature), dim=1)
        sel = torch.distributions.Categorical(logits=log_b_topk).sample()
        log_b_chosen = log_b_topk.gather(1, sel.view(-1, 1)).view(-1)
        b_topk = log_b_topk.exp()
        log_diff = log_b_topk - topk_logp
        # 0·log0 = 0 by convention; mask the contribution where b ≈ 0.
        contrib = torch.where(b_topk > 1e-12, b_topk * log_diff, torch.zeros_like(b_topk))
        kl_b_pi = contrib.sum(dim=1)

    chosen = topk_idx.gather(1, sel.view(-1, 1)).view(-1)
    return (chosen, log_pi, masks, root_values.view(-1), states,
            log_b_chosen, kl_b_pi, sel, topk_idx, log_b_topk)
