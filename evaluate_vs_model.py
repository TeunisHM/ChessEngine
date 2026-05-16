import argparse
import os
from time import perf_counter

import chess
import torch

from helper import index_to_move
from lookahead import select_moves_with_lookahead
from models import ActorCriticResNet, load_actor_critic_state_dict


def play_game(net_a, net_b, device, *, a_is_white: bool,
              lookahead_k: int, lookahead_alpha: float, temperature: float):
    board = chess.Board()
    while not board.is_game_over():
        a_turn = (board.turn == chess.WHITE) == a_is_white
        net = net_a if a_turn else net_b
        idxs, *_ = select_moves_with_lookahead(
            net, [board], device,
            top_k=lookahead_k, alpha=lookahead_alpha,
            temperature=temperature,
        )
        move = index_to_move(int(idxs[0].item()), board)
        if move is None or move not in board.legal_moves:
            break
        board.push(move)
    return board.result()


def main():
    parser = argparse.ArgumentParser(description="Pit model A vs model B.")
    parser.add_argument("--model-a", required=True)
    parser.add_argument("--model-b", required=True)
    parser.add_argument("--games", type=int, default=101)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--lookahead-k", type=int, default=8)
    parser.add_argument("--lookahead-alpha", type=float, default=0.33)
    args = parser.parse_args()

    device = (
        torch.device(args.device) if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    nets = {}
    for path in (args.model_a, args.model_b):
        if not os.path.exists(path):
            raise SystemExit(f"Model file not found: {path}")
        n = ActorCriticResNet().to(device)
        load_actor_critic_state_dict(n, torch.load(path, map_location=device))
        n.eval()
        nets[path] = n

    net_a, net_b = nets[args.model_a], nets[args.model_b]

    a_wins = b_wins = draws = 0
    a_white_wins = a_black_wins = 0
    start = perf_counter()
    with torch.inference_mode():
        for i in range(args.games):
            a_is_white = i % 2 == 0
            result = play_game(
                net_a, net_b, str(device),
                a_is_white=a_is_white,
                lookahead_k=args.lookahead_k,
                lookahead_alpha=args.lookahead_alpha,
                temperature=args.temperature,
            )
            if result == "1-0":
                if a_is_white:
                    a_wins += 1; a_white_wins += 1
                else:
                    b_wins += 1
            elif result == "0-1":
                if not a_is_white:
                    a_wins += 1; a_black_wins += 1
                else:
                    b_wins += 1
            else:
                draws += 1

            done = i + 1
            elapsed = perf_counter() - start
            rate = done / elapsed if elapsed > 0 else 0
            eta = (args.games - done) / rate if rate > 0 else 0
            bar_len = 30
            filled = int(done / args.games * bar_len)
            bar = "=" * filled + "-" * (bar_len - filled)
            print(
                f"\r[{bar}] {done}/{args.games} | A:{a_wins} D:{draws} B:{b_wins} | "
                f"{elapsed:5.1f}s | ETA {eta:5.1f}s",
                end="", flush=True,
            )
    print()
    print("\n--- Model vs Model ---")
    print(f"A = {args.model_a}")
    print(f"B = {args.model_b}")
    print(f"A wins: {a_wins} (white {a_white_wins}, black {a_black_wins}) | "
          f"Draws: {draws} | B wins: {b_wins}")
    print("----------------------\n")


if __name__ == "__main__":
    main()
