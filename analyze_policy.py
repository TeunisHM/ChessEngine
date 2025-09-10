import argparse
import os
from datetime import datetime
import chess
import torch
from helper import board_to_tensor, legal_moves_mask, index_to_move, PIECE_VALUES
from train import ActorCriticResNet
import time

@torch.no_grad()
def topk_policy_with_values(board, net, logits, k=5, gamma=0.995):
    """
    Returns a list of (move, prob, logit, value_score) for the top-K legal moves.

    - prob: softmax probability over legal moves
    - logit: masked logit for the move
    - value_score: immediate reward (captures) + gamma * (-V(s')) if non-terminal, or terminal reward
    """
    device = next(net.parameters()).device

    mask = legal_moves_mask(board).to(logits.device)
    if mask.sum().item() == 0:
        return []

    masked_logits = logits.masked_fill(mask == 0, -1e9)
    probs = torch.softmax(masked_logits, dim=0)

    k_eff = min(k, int(mask.sum().item()))
    topk = torch.topk(masked_logits, k=k_eff)

    entries = []
    for idx in topk.indices.tolist():
        move = index_to_move(idx, board)
        if move is None or move not in board.legal_moves:
            continue

        # Immediate reward (align with training capture bonus)
        immediate = 0.0
        if board.is_capture(move):
            if board.is_en_passant(move):
                captured_sq = chess.square(
                    chess.square_file(move.to_square),
                    chess.square_rank(move.from_square),
                )
            else:
                captured_sq = move.to_square
            captured_piece = board.piece_at(captured_sq)
            if captured_piece is not None:
                immediate += PIECE_VALUES.get(captured_piece.piece_type, 0.0) / 20.0

        board.push(move)
        if board.is_game_over():
            result = board.result()
            mover_is_white = not board.turn  # flipped after push
            if result == "1-0":
                score = immediate + (2.5 if mover_is_white else -1.0)
            elif result == "0-1":
                score = immediate + (2.5 if not mover_is_white else -1.0)
            else:
                score = immediate - 0.5
        else:
            state_next = board_to_tensor(board).unsqueeze(0).to(device)
            _, v_next = net(state_next)
            score = immediate + gamma * (-v_next.item())
        board.pop()

        entries.append((move, probs[idx].item(), masked_logits[idx].item(), score))

    # Sort by value score (desc), tie-break by prob
    entries.sort(key=lambda t: (t[3], t[1]), reverse=True)
    return entries


@torch.no_grad()
def play_and_explain(model_path=None, k=5, num_moves=200, sample=False, seed=0, gamma=0.995):
    """
    Plays a single self-play game (policy controls both sides) and prints per-move diagnostics:
    - current value V(s)
    - top-K policy moves: SAN, prob, masked logit, value lookahead score
    - chosen move (sample or argmax of policy probs)

    If model_path is None, initializes a fresh ActorCriticResNet.
    """
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = ActorCriticResNet().to(device)
    if model_path and os.path.exists(model_path):
        print(f"Loading model: {model_path}")
        state = torch.load(model_path, map_location=device)
        net.load_state_dict(state)
    else:
        print("Using randomly initialized network (no model file provided/found).")

    net.eval()
    board = chess.Board()

    ply = 0
    while not board.is_game_over() and ply < num_moves:
        side = "White" if board.turn == chess.WHITE else "Black"
        state = board_to_tensor(board).unsqueeze(0).to(device)
        logits, v = net(state)
        logits = logits[0]
        v = v.item()

        # Mask and probs for info
        mask_cpu = legal_moves_mask(board)
        n_legal = int(mask_cpu.sum().item())
        if n_legal == 0:
            print("No legal moves; aborting.")
            break
        mask = mask_cpu.to(logits.device)
        masked_logits = logits.masked_fill(mask == 0, -1e9)
        probs = torch.softmax(masked_logits, dim=0)

        # Collect top-K with values
        topk_entries = topk_policy_with_values(board, net, logits, k=k, gamma=gamma)

        print(f"\nMove {ply+1} ({side} to move)")
        print(board)
        print(f"Value V(s): {v:+.4f}")
        print("Top-{} candidates (SAN | prob | logit | lookahead score):".format(len(topk_entries)))
        for move, p, lg, sc in topk_entries:
            try:
                san = board.san(move)
            except Exception:
                san = move.uci()
            print(f"  {san:8s} | {p:7.4f} | {lg:7.3f} | {sc:+7.3f}")

        # Summaries: sum of shown probs and number of legal moves
        sum_topk = sum(p for _, p, _, _ in topk_entries)
        print(f"Sum prob (top-{len(topk_entries)}): {sum_topk:.4f} | Legal moves: {n_legal}")

        # Choose policy move (sample or argmax)
        if sample:
            dist = torch.distributions.Categorical(probs)
            act_idx = dist.sample().item()
        else:
            act_idx = int(torch.argmax(probs).item())
        move = index_to_move(act_idx, board)
        if move is None or move not in board.legal_moves:
            # fallback to best by value lookahead, else first legal
            move = topk_entries[0][0] if topk_entries else next(iter(board.legal_moves))

        try:
            chosen_san = board.san(move)
        except Exception:
            chosen_san = move.uci()
        print(f"Chosen: {chosen_san}\n")

        board.push(move)
        ply += 1
        time.sleep(1.5)
        
    print("\nFinal position:")
    print(board)
    print(f"Result: {board.result()}")
    outcome = board.outcome()
    if outcome is not None:
        term = outcome.termination
        term_name = getattr(term, 'name', str(term))
        winner = outcome.winner
        if winner is None:
            winner_str = "None"
        else:
            winner_str = "White" if winner else "Black"
        print(f"Termination: {term_name}")
        print(f"Winner: {winner_str}")


def main():
    parser = argparse.ArgumentParser(description="Analyze policy decisions with top-K lookahead")
    parser.add_argument("--model", type=str, default=None, help="Path to .pt model file")
    parser.add_argument("--k", type=int, default=5, help="Top-K candidates to display")
    parser.add_argument("--moves", type=int, default=800, help="Max number of plies to play")
    parser.add_argument("--sample", action="store_true", help="Sample from policy instead of argmax")
    parser.add_argument("--gamma", type=float, default=0.995, help="Discount for lookahead value")
    args = parser.parse_args()

    play_and_explain(model_path=args.model, k=args.k, num_moves=args.moves, sample=args.sample, gamma=args.gamma)


if __name__ == "__main__":
    main()
