import torch
import chess
from helper import board_to_tensor, index_to_move, legal_moves_mask

DEFAULT_SEARCH_DEPTH = 2


def search_select_move(board: chess.Board,
                       actor_critic_net,
                       logits: torch.Tensor,
                       device: str,
                       k: int = 5,
                       temperature: float = 1.0,
                       depth: int = DEFAULT_SEARCH_DEPTH,
                       legal_mask: torch.Tensor = None):
    """
    Lightweight fixed-depth search:
      1) Take top-k moves by policy prob at each node.
      2) Alternate max (current player) and min (opponent) layers up to `depth` plies.
      3) Score leaves with value head (for side to move at leaf) and terminal outcomes.
      4) Softmax scores (temperature) to pick a move; return move, log_prob, entropy.
    """
    if legal_mask is None:
        legal_mask = legal_moves_mask(board)
    mask = legal_mask.to(device)
    if mask.sum() == 0:
        return None, None, None

    was_training = actor_critic_net.training
    actor_critic_net.eval()

    value_cache = {}

    try:
        masked_logits = logits.masked_fill(~mask, -1e9)
        probs = torch.softmax(masked_logits, dim=0)

        sample_k = min(k, int(mask.sum().item()))

        temp_is_greedy = temperature is None or temperature <= 1e-6
        # When temperature == 1.0 the sampling distribution equals probs — short-circuit.
        if temp_is_greedy:
            candidate_indices = torch.topk(probs, k=sample_k).indices
        elif abs(float(temperature) - 1.0) < 1e-6:
            candidate_indices = torch.multinomial(probs, num_samples=sample_k, replacement=False)
        else:
            sample_probs = torch.softmax(masked_logits / float(temperature), dim=0)
            candidate_indices = torch.multinomial(sample_probs, num_samples=sample_k, replacement=False)

        candidate_moves = []
        candidate_scores = []
        candidate_indices_list = []

        root_turn = board.turn

        def _forward_policy_value(current_board: chess.Board):
            key = current_board._transposition_key() if hasattr(current_board, "_transposition_key") else current_board.fen()
            cached = value_cache.get(key)
            if cached is not None:
                return cached
            with torch.no_grad():
                state = board_to_tensor(current_board).unsqueeze(0).to(device)
                local_logits, val = actor_critic_net(state)
            value_cache[key] = (local_logits[0], val)
            return local_logits[0], val

        def _minimax(current_board: chess.Board, ply: int, maximizing: bool) -> float:
            if current_board.is_game_over():
                result = current_board.result()
                if result == "1-0":
                    return 1.0 if root_turn == chess.WHITE else -1.0
                elif result == "0-1":
                    return 1.0 if root_turn == chess.BLACK else -1.0
                else:
                    return 0.0

            if ply == 0:
                _, val = _forward_policy_value(current_board)
                val_item = val.item()
                if current_board.turn == root_turn:
                    return val_item
                else:
                    return -val_item

            local_mask = legal_moves_mask(current_board).to(device)
            if local_mask.sum() == 0:
                return 0.0

            local_logits, _ = _forward_policy_value(current_board)
            local_probs = torch.softmax(local_logits.masked_fill(~local_mask, -1e9), dim=0)
            local_topk = torch.topk(local_probs, k=min(k, int(local_mask.sum().item())))

            best_val = -float("inf") if maximizing else float("inf")

            for idx in local_topk.indices.tolist():
                mv = index_to_move(idx, current_board)
                if mv is None or mv not in current_board.legal_moves:
                    continue
                current_board.push(mv)
                val = _minimax(current_board, ply - 1, not maximizing)
                current_board.pop()

                if maximizing:
                    best_val = max(best_val, val)
                else:
                    best_val = min(best_val, val)

            return best_val if best_val != float("inf") and best_val != -float("inf") else 0.0

        for idx in candidate_indices.tolist():
            move = index_to_move(idx, board)
            if move is None or move not in board.legal_moves:
                continue

            board.push(move)
            score = _minimax(board, depth - 1, maximizing=False)
            board.pop()

            candidate_moves.append(move)
            candidate_scores.append(score)
            candidate_indices_list.append(idx)

        if not candidate_moves:
            return None, None, None

        scores_tensor = torch.tensor(candidate_scores, dtype=torch.float32, device=device)
        if temp_is_greedy:
            probs_sel = torch.zeros_like(scores_tensor)
            probs_sel[torch.argmax(scores_tensor)] = 1.0
        else:
            probs_sel = torch.softmax(scores_tensor / float(temperature), dim=0)

        dist_sel = torch.distributions.Categorical(probs=probs_sel)
        choice = dist_sel.sample()
        move = candidate_moves[choice.item()]

        # Gradient from the ORIGINAL policy distribution.
        chosen_policy_idx = candidate_indices_list[choice.item()]
        dist_policy = torch.distributions.Categorical(probs=probs)
        log_prob = dist_policy.log_prob(torch.tensor(chosen_policy_idx, device=device))
        entropy = dist_policy.entropy()

        return move, log_prob, entropy
    finally:
        if was_training:
            actor_critic_net.train()
