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

from helper import board_to_tensor, index_to_move, legal_moves_mask


@torch.inference_mode()
def select_moves_from_policy(
    net,
    boards: List[chess.Board],
    device,
    *,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample a move per board directly from the masked policy distribution.

    Same return signature as select_moves_with_lookahead so call sites can swap.
    Used in PPO rollouts so the action distribution actually equals pi(a|s) and
    old_log_prob = log_pi[action] is a correct importance-sampling baseline.
    """
    states = torch.stack([board_to_tensor(b) for b in boards]).to(device)
    masks = torch.stack([legal_moves_mask(b) for b in boards]).to(device)

    logits, root_values = net(states)
    masked = logits.masked_fill(~masks, -1e9)
    log_pi = F.log_softmax(masked, dim=1)

    if temperature is None or temperature <= 1e-6:
        chosen = log_pi.argmax(dim=1)
    else:
        probs = F.softmax(masked / float(temperature), dim=1)
        chosen = torch.distributions.Categorical(probs=probs).sample()

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


@torch.inference_mode()
def _quiesce_ab(net, board: chess.Board, device,
                alpha: float, beta: float, depth: int,
                check_budget: int = 1) -> float:
    """Negamax alpha-beta quiescence with stand-pat cutoff. Side-to-move value.

    Extends through captures at every depth; non-capturing checks are explored
    while check_budget > 0, decremented after each check expansion. This
    surfaces forcing-check tactics and mate threats that would otherwise be
    invisible to a pure capture-only quiescence.

    When the side to move is in check, stand-pat is illegal: we must respond
    to the check. We skip the stand-pat baseline and search all legal evasions
    (captures first, then quiet evasions). A hard floor caps recursion so
    perpetual-check sequences can't run unbounded.
    """
    if board.is_game_over(claim_draw=False):
        return -1.0 if board.is_checkmate() else 0.0

    if board.is_check():
        # Recursion floor: fall back to the net's evaluation if we're stuck
        # deep in a check sequence. V here is approximate (net learns mostly
        # quiet positions) but bounded — better than infinite recursion.
        if depth <= 0:
            state = board_to_tensor(board).unsqueeze(0).to(device)
            _, v = net(state)
            return float(v.item())

        # No stand-pat — side to move can't refuse the check. Search all legal
        # evasions: captures first (often refute the check best), then quiet.
        for move_set in (_ordered_captures(board),
                         (m for m in board.legal_moves if not board.is_capture(m))):
            for move in move_set:
                child = board.copy(stack=False)
                child.push(move)
                score = -_quiesce_ab(net, child, device, -beta, -alpha,
                                     depth - 1, check_budget)
                if score >= beta:
                    return score
                if score > alpha:
                    alpha = score
        return alpha

    state = board_to_tensor(board).unsqueeze(0).to(device)
    _, v = net(state)
    stand_pat = float(v.item())

    # Stand-pat fail-high: side to move could just refuse to capture.
    if stand_pat >= beta:
        return stand_pat
    if stand_pat > alpha:
        alpha = stand_pat
    if depth <= 0:
        return alpha

    for move in _ordered_captures(board):
        child = board.copy(stack=False)
        child.push(move)
        score = -_quiesce_ab(net, child, device, -beta, -alpha,
                             depth - 1, check_budget)
        if score >= beta:
            return score
        if score > alpha:
            alpha = score

    if check_budget > 0:
        for move in _ordered_checks(board):
            child = board.copy(stack=False)
            child.push(move)
            score = -_quiesce_ab(net, child, device, -beta, -alpha,
                                 depth - 1, check_budget - 1)
            if score >= beta:
                return score
            if score > alpha:
                alpha = score

    return alpha


@torch.inference_mode()
def quiesce_batched(
    net,
    boards: List[chess.Board],
    device,
    max_qdepth: int = 2,
    check_budget: int = 1,
) -> torch.Tensor:
    """Per-board quiescence via alpha-beta DFS with stand-pat short-circuit and
    MVV-LVA capture ordering. Returns one value per input board (side-to-move).

    check_budget allows up to N non-capturing-check extensions per call chain;
    set to 0 to fall back to pure capture quiescence.
    """
    if not boards:
        return torch.empty(0, device=device)
    out = [_quiesce_ab(net, b, device, -1e9, 1e9, max_qdepth, check_budget)
           for b in boards]
    return torch.tensor(out, device=device, dtype=torch.float32)


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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pick a move per board via top-k policy + value quiescence at the children.

    Returns (action_idx, log_pi, masks, root_values, states,
             log_b_chosen, kl_b_pi, pick_rank, topk_idx, log_b_topk),
    all on `device`.

    log_pi: masked log-softmax over the full action space.
    log_b_chosen: log probability that this search procedure assigned to the
        chosen action — i.e. the *behavior* policy log-prob. Use as PPO
        old_log_prob when sampling actions through this function so the IS
        ratio π_new(a|s) / b(a|s) is well-formed.
    kl_b_pi: per-state KL(b || π) over the top-k support; how much the search
        distribution disagrees with the raw policy at that state.
    pick_rank: 0-indexed rank of the chosen action in π's top-k (0 = π's #1).
    topk_idx: (n, k) action indices that make up the search support.
    log_b_topk: (n, k) log b over those indices; with temperature<=0, b is a
        delta on the chosen action (one-hot row).
    """
    states = torch.stack([board_to_tensor(b) for b in boards]).to(device)
    masks = torch.stack([legal_moves_mask(b) for b in boards]).to(device)

    logits, root_values = net(states)
    masked = logits.masked_fill(~masks, -1e9)
    log_pi = F.log_softmax(masked, dim=1)

    n = len(boards)
    k = min(top_k, log_pi.shape[1])
    topk_logp, topk_idx = log_pi.topk(k, dim=1)

    topk_idx_cpu = topk_idx.cpu().tolist()
    is_invalid = torch.zeros(n, k, dtype=torch.bool, device=device)
    neg_v = torch.zeros(n, k, device=device)
    child_boards: List[chess.Board] = []
    child_rows: List[int] = []
    child_cols: List[int] = []

    for i, board in enumerate(boards):
        for j in range(k):
            action_idx = topk_idx_cpu[i][j]
            move = index_to_move(action_idx, board)
            if move is None or move not in board.legal_moves:
                is_invalid[i, j] = True
                continue
            child = board.copy(stack=False)
            child.push(move)
            if child.is_game_over(claim_draw=False):
                neg_v[i, j] = 1.0 if child.is_checkmate() else 0.0
            else:
                child_boards.append(child)
                child_rows.append(i)
                child_cols.append(j)

    if child_boards:
        child_values = quiesce_batched(net, child_boards, device, max_qdepth=max_qdepth)
        rows_t = torch.tensor(child_rows, device=device, dtype=torch.long)
        cols_t = torch.tensor(child_cols, device=device, dtype=torch.long)
        neg_v.index_put_((rows_t, cols_t), -child_values.to(neg_v.dtype))

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
        log_b_topk = torch.full((n, k), -1e9, device=device)
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
