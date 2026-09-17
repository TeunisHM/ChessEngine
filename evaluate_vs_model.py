import argparse
import os
from time import perf_counter

import chess
import torch

if torch.version.hip is not None:
    os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")

from helper import board_to_tensor, index_to_move, legal_moves_mask
from search_backends import add_search_args, search_config, select_moves
from models import net_from_state_dict


def _raw_policy_index(net, board, device, temperature: float) -> int:
    """Sample a canonical move index straight from the masked policy (no search)."""
    state = board_to_tensor(board).unsqueeze(0).to(device)
    logits, _ = net(state)
    masked = logits[0].masked_fill(~legal_moves_mask(board).to(device), -1e9)
    if temperature > 0:
        dist = torch.distributions.Categorical(logits=masked / temperature)
        return int(dist.sample().item())
    return int(masked.argmax().item())


def play_game(net_a, net_b, device, *, a_is_white: bool,
              lookahead_k: int, lookahead_alpha: float, temperature: float,
              raw_a: bool = False, raw_b: bool = False,
              max_qdepth: int = 2,
              search_cfg: "dict | None" = None,
              start_board: "chess.Board | None" = None):
    board = chess.Board() if start_board is None else start_board.copy()
    while not board.is_game_over():
        a_turn = (board.turn == chess.WHITE) == a_is_white
        net = net_a if a_turn else net_b
        if raw_a if a_turn else raw_b:
            idx = _raw_policy_index(net, board, device, temperature)
        else:
            idxs, *_ = select_moves(
                net, [board], device,
                top_k=lookahead_k, alpha=lookahead_alpha,
                temperature=temperature, max_qdepth=max_qdepth,
                **(search_cfg or {}),
            )
            idx = int(idxs[0].item())
        move = index_to_move(idx, board)
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
    parser.add_argument("--raw-a", action="store_true",
                        help="Model A plays the raw policy (no lookahead).")
    parser.add_argument("--raw-b", action="store_true",
                        help="Model B plays the raw policy (no lookahead).")
    parser.add_argument("--max-qdepth", type=int, default=2,
                        help="Max forcing-move plies quiescence extends below "
                             "each candidate (default 2). Higher = deeper "
                             "tactical horizon along captures/checks.")
    parser.add_argument("--paired-openings", action="store_true",
                        help="Play every opening in the book twice with colors "
                             "reversed (removes opening+color noise); ignores --games.")
    add_search_args(parser)
    args = parser.parse_args()

    device = (
        torch.device(args.device) if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    nets = {}
    for path in (args.model_a, args.model_b):
        if not os.path.exists(path):
            raise SystemExit(f"Model file not found: {path}")
        n = net_from_state_dict(torch.load(path, map_location=device), device)
        n.eval()
        nets[path] = n

    net_a, net_b = nets[args.model_a], nets[args.model_b]

    # Build the game plan: (a_is_white, start_board). Paired openings play each
    # line twice with colors reversed, removing opening+color as noise sources.
    plan = []
    if args.paired_openings:
        from helper import OPENINGS
        for san_line in OPENINGS.values():
            b = chess.Board()
            try:
                for san in san_line:
                    b.push_san(san)
            except Exception:
                continue
            plan.append((True, b)); plan.append((False, b))
    else:
        plan = [(i % 2 == 0, None) for i in range(args.games)]

    a_wins = b_wins = draws = 0
    a_white_wins = a_black_wins = 0
    total = len(plan)
    start = perf_counter()
    with torch.inference_mode():
        for i, (a_is_white, start_board) in enumerate(plan):
            result = play_game(
                net_a, net_b, str(device),
                a_is_white=a_is_white,
                lookahead_k=args.lookahead_k,
                lookahead_alpha=args.lookahead_alpha,
                temperature=args.temperature,
                raw_a=args.raw_a,
                raw_b=args.raw_b,
                max_qdepth=args.max_qdepth,
                search_cfg=search_config(args),
                start_board=start_board,
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
            eta = (total - done) / rate if rate > 0 else 0
            bar_len = 30
            filled = int(done / total * bar_len)
            bar = "=" * filled + "-" * (bar_len - filled)
            print(
                f"\r[{bar}] {done}/{total} | A:{a_wins} D:{draws} B:{b_wins} | "
                f"{elapsed:5.1f}s | ETA {eta:5.1f}s",
                end="", flush=True,
            )
    print()
    print("\n--- Model vs Model ---")
    mode_a = "raw" if args.raw_a else "search"
    mode_b = "raw" if args.raw_b else "search"
    print(f"A = {args.model_a} [{mode_a}]")
    print(f"B = {args.model_b} [{mode_b}]")
    print(f"A wins: {a_wins} (white {a_white_wins}, black {a_black_wins}) | "
          f"Draws: {draws} | B wins: {b_wins}")
    print("----------------------\n")


if __name__ == "__main__":
    main()
