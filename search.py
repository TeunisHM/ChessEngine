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
                       depth: int = DEFAULT_SEARCH_DEPTH):
    """
    Lightweight fixed-depth search:
      1) Take top-k moves by policy prob at each node.
      2) Alternate max (current player) and min (opponent) layers up to `depth` plies.
      3) Score leaves with value head (for side to move at leaf) and terminal outcomes.
      4) Softmax scores (temperature) to pick a move; return move, log_prob, entropy.
    """
    mask = legal_moves_mask(board).to(device)
    if mask.sum() == 0:
        return None, None, None

    was_training = actor_critic_net.training
    actor_critic_net.eval()

    try:
        masked_logits = logits.masked_fill(~mask, -1e9)
        probs = torch.softmax(masked_logits, dim=0)

        sample_k = min(k, int(mask.sum().item()))

        # Sample candidate set with temperature on policy; fallback to greedy if temperature is effectively zero.
        policy_logits_temp = masked_logits if temperature is None or temperature <= 1e-6 else masked_logits / float(temperature)
        policy_probs = torch.softmax(policy_logits_temp, dim=0)
        if temperature is None or temperature <= 1e-6:
            candidate_indices = torch.topk(policy_probs, k=sample_k).indices
        else:
            candidate_indices = torch.multinomial(policy_probs, num_samples=sample_k, replacement=False)

        candidate_moves = []
        candidate_scores = []
        candidate_indices_list = []

        root_turn = board.turn 

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
                with torch.no_grad():
                    state = board_to_tensor(current_board).unsqueeze(0).to(device)
                    _, val = actor_critic_net(state)
                val_item = val.item()
                # If leaf node is opponent's turn, their "Good" is our "Bad"
                if current_board.turn == root_turn:
                    return val_item
                else:
                    return -val_item    

            local_mask = legal_moves_mask(current_board).to(device)
            if local_mask.sum() == 0:
                return 0.0

            with torch.no_grad():
                local_logits = actor_critic_net(
                    board_to_tensor(current_board).unsqueeze(0).to(device)
                )[0][0]
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
        if temperature is None or temperature <= 1e-6:
            probs_sel = torch.zeros_like(scores_tensor)
            probs_sel[torch.argmax(scores_tensor)] = 1.0
        else:
            probs_sel = torch.softmax(scores_tensor / float(temperature), dim=0)

        dist_policy = torch.distributions.Categorical(probs=probs)
        if not candidate_moves:
            move_idx = dist_policy.sample()
            move = index_to_move(move_idx.item(), board)
            if move is None or move not in board.legal_moves:
                return None, None, None
            return move, dist_policy.log_prob(move_idx), dist_policy.entropy()

        dist_sel = torch.distributions.Categorical(probs=probs_sel)
        choice = dist_sel.sample()
        move = candidate_moves[choice.item()]
        
        # IMPORTANT: Get gradient from the ORIGINAL Policy distribution
        chosen_policy_idx = candidate_indices_list[choice.item()]
        dist_policy = torch.distributions.Categorical(probs=probs) # Original Policy
        log_prob = dist_policy.log_prob(torch.tensor(chosen_policy_idx, device=device))
        entropy = dist_policy.entropy()

        return move, log_prob, entropy
    finally:
        if was_training:
            actor_critic_net.train()
