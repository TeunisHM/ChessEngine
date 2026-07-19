import argparse
import os
import random
from time import perf_counter

import chess
import torch

if torch.version.hip is not None:
    os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")

from helper import index_to_move
from lookahead import select_moves_with_lookahead
from models import net_from_state_dict


def evaluate_vs_random(actor_critic_net,
                       num_games: int = 100,
                       show_progress: bool = True,
                       device: str = "cpu",
                       temperature: float = 0.0,
                       lookahead_k: int = 8,
                       lookahead_alpha: float = 0.33,
                       max_qdepth: int = 2):
    """Evaluate the policy against a random opponent. temperature<=0 is greedy."""
    was_training = actor_critic_net.training
    actor_critic_net.eval()

    results = []
    white_wins = black_wins = ties = 0
    policy_white_wins = policy_black_wins = 0
    game_lengths = []

    with torch.inference_mode():
        start_t = perf_counter()
        for i in range(num_games):
            board = chess.Board()
            is_policy_white = i % 2 == 0
            move_count = 0

            while not board.is_game_over():
                policy_turn = (board.turn == chess.WHITE) == is_policy_white
                if policy_turn:
                    idxs, *_ = select_moves_with_lookahead(
                        actor_critic_net, [board], device,
                        top_k=lookahead_k, alpha=lookahead_alpha,
                        temperature=temperature,
                        max_qdepth=max_qdepth,
                    )
                    move = index_to_move(int(idxs[0].item()), board)
                    if move is None or move not in board.legal_moves:
                        move = random.choice(list(board.legal_moves))
                else:
                    move = random.choice(list(board.legal_moves))

                board.push(move)
                move_count += 1

            game_lengths.append(move_count)
            result = board.result()
            outcome = 0
            if result == "1-0":
                white_wins += 1
                if is_policy_white:
                    outcome = 1
                    policy_white_wins += 1
                else:
                    outcome = -1
            elif result == "0-1":
                black_wins += 1
                if not is_policy_white:
                    outcome = 1
                    policy_black_wins += 1
                else:
                    outcome = -1
            else:
                ties += 1
            results.append(outcome)

            if show_progress:
                done = i + 1
                frac = done / max(1, num_games)
                bar_len = 30
                filled = int(frac * bar_len)
                bar = "=" * filled + "-" * (bar_len - filled)
                elapsed = perf_counter() - start_t
                rate = done / elapsed if elapsed > 0 else 0.0
                eta = (num_games - done) / rate if rate > 0 else 0.0
                print(
                    f"\rEvaluating [{bar}] {done}/{num_games} | "
                    f"{elapsed:5.1f}s elapsed | ETA {eta:5.1f}s",
                    end="", flush=True,
                )

    if was_training:
        actor_critic_net.train()
    if show_progress:
        print()

    wins = results.count(1)
    losses = results.count(-1)

    print("\n--- Evaluation Results ---")
    print(f"Policy vs Random: {wins} Wins / {ties} Draws / {losses} Losses")
    print(f"Policy as White Wins: {policy_white_wins}, Policy as Black Wins: {policy_black_wins}")
    print("--------------------------\n")

    return {
        "white_wins": white_wins,
        "black_wins": black_wins,
        "draws": ties,
        "wins": wins,
        "losses": losses,
        "policy_white_wins": policy_white_wins,
        "policy_black_wins": policy_black_wins,
        "avg_game_length": sum(game_lengths) / len(game_lengths) if game_lengths else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Load a model and evaluate vs random.")
    parser.add_argument("--model", "-m", type=str, required=True, help="Path to .pt model file")
    parser.add_argument("--games", "-g", type=int, default=100, help="Number of games to play")
    parser.add_argument("--device", type=str, default=None, help="Device override: cpu or cuda")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature for policy moves; <=0 is greedy.")
    parser.add_argument("--lookahead-k", type=int, default=8,
                        help="Top-k policy candidates considered by 1-ply value lookahead.")
    parser.add_argument("--lookahead-alpha", type=float, default=0.33,
                        help="Weight on log pi(a) in the lookahead score: -V(child) + alpha*log pi.")
    parser.add_argument("--max-qdepth", type=int, default=2,
                        help="Max quiescence search depth for capture extensions.")
    args = parser.parse_args()

    device = (
        torch.device(args.device) if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    if not os.path.exists(args.model):
        raise SystemExit(f"Model file not found: {args.model}")

    print(f"Loading model: {args.model}")
    net = net_from_state_dict(torch.load(args.model, map_location=device), device)
    net.eval()

    evaluate_vs_random(
        net,
        num_games=args.games,
        show_progress=True,
        device=str(device),
        temperature=args.temperature,
        lookahead_k=args.lookahead_k,
        lookahead_alpha=args.lookahead_alpha,
        max_qdepth=args.max_qdepth,
    )


if __name__ == "__main__":
    main()
