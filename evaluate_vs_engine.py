import argparse
import os
from typing import Dict, Optional

import chess
import chess.engine
import torch

from helper import index_to_move
from lookahead import select_moves_with_lookahead
from models import ActorCriticResNet, load_actor_critic_state_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model against a UCI chess engine."
    )
    parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Path to the .pt model file.",
    )
    parser.add_argument(
        "--engine-path",
        default='./stockfish/stockfish',
        help="Path to a UCI engine executable. If omitted, tries STOCKFISH_PATH env, local ./stockfish, PATH, and common system locations.",
    )
    parser.add_argument(
        "--engine-move-time",
        type=float,
        default=0.1,
        help="Seconds allowed per engine move (e.g., 0.1 = 100ms).",
    )
    parser.add_argument(
        "--engine-skill-level",
        type=int,
        default=0,
        help="Optional engine skill level (0-20 for Stockfish).",
    )
    parser.add_argument(
        "--games",
        "-g",
        type=int,
        default=100,
        help="Number of games to play.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override: cpu or cuda. Defaults to auto-detect.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for policy moves; <=0 is greedy.",
    )
    parser.add_argument(
        "--lookahead-k",
        type=int,
        default=8,
        help="Top-k policy candidates considered by 1-ply value lookahead.",
    )
    parser.add_argument(
        "--lookahead-alpha",
        type=float,
        default=0.33,
        help="Weight on log pi(a) in the lookahead score: -V(child) + alpha*log pi.",
    )
    return parser.parse_args()

def _create_engine(path: str, skill: Optional[int]) -> chess.engine.SimpleEngine:
    engine = chess.engine.SimpleEngine.popen_uci(path)
    if skill is not None:
        try:
            engine.configure({"Skill Level": int(skill)})
        except Exception as exc:  # pragma: no cover
            print(f"[WARN] Could not set engine skill level: {exc}")
    return engine


def _play_game(
    net: ActorCriticResNet,
    engine: chess.engine.SimpleEngine,
    device: torch.device,
    game_index: int,
    temperature: float,
    engine_move_time: float,
    lookahead_k: int = 8,
    lookahead_alpha: float = 0.33,
) -> Dict[str, int]:
    board = chess.Board()
    is_policy_white = (game_index % 2 == 0)
    move_count = 0

    while not board.is_game_over():
        policy_turn = (board.turn == chess.WHITE and is_policy_white) or (
            board.turn == chess.BLACK and not is_policy_white
        )

        if policy_turn:
            idxs, *_ = select_moves_with_lookahead(
                net, [board], device,
                top_k=lookahead_k, alpha=lookahead_alpha,
                temperature=temperature,
            )
            move = index_to_move(int(idxs[0].item()), board)
            if move is None or move not in board.legal_moves:
                move = next(iter(board.legal_moves))
        else:
            try:
                limit = chess.engine.Limit(time=max(engine_move_time, 0.01))
                result = engine.play(board, limit=limit)
                move = result.move
            except Exception as exc:  # pragma: no cover - runtime safety
                print(f"[WARN] Engine failed to move: {exc}")
                move = next(iter(board.legal_moves))

        board.push(move)
        move_count += 1

    outcome = board.result()
    return {
        "result": outcome,
        "policy_as_white": int(is_policy_white),
        "plies": move_count,
    }


def main() -> None:
    args = parse_args()
    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    if not os.path.exists(args.model):
        print(f"[ERROR] Model file not found: {args.model}")
        raise SystemExit(1)

    print(f"[INFO] Loading model from {args.model}")
    net = ActorCriticResNet().to(device)
    state = torch.load(args.model, map_location=device)
    load_actor_critic_state_dict(net, state)
    net.eval()

    engine_path = args.engine_path
    try:
        engine = _create_engine(engine_path, args.engine_skill_level)
    except Exception as exc:
        print(f"[ERROR] Could not launch engine at '{engine_path}': {exc}")
        raise SystemExit(1)

    wins = losses = draws = 0
    policy_white_wins = policy_black_wins = 0
    game_lengths = []

    try:
        for g in range(args.games):
            stats = _play_game(
                net=net,
                engine=engine,
                device=device,
                game_index=g,
                temperature=args.temperature,
                engine_move_time=args.engine_move_time,
                lookahead_k=args.lookahead_k,
                lookahead_alpha=args.lookahead_alpha,
            )
            game_lengths.append(stats["plies"])
            result = stats["result"]
            if result == "1-0":
                wins += 1 if stats["policy_as_white"] else 0
                losses += 1 if not stats["policy_as_white"] else 0
                policy_white_wins += 1 if stats["policy_as_white"] else 0
            elif result == "0-1":
                wins += 1 if not stats["policy_as_white"] else 0
                losses += 1 if stats["policy_as_white"] else 0
                policy_black_wins += 1 if not stats["policy_as_white"] else 0
            else:
                draws += 1

            done = g + 1
            avg_len = sum(game_lengths) / max(1, len(game_lengths))
            print(
                f"\rGames {done}/{args.games} | Wins {wins} Draws {draws} Losses {losses} | Avg plies {avg_len:.1f}",
                end="",
                flush=True,
            )
        print()
    finally:
        engine.quit()

    print("\n--- Evaluation vs Engine ---")
    print(f"Wins: {wins} | Draws: {draws} | Losses: {losses}")
    print(f"Policy as White Wins: {policy_white_wins} | Policy as Black Wins: {policy_black_wins}")
    print(f"Avg plies per game: {sum(game_lengths) / max(1, len(game_lengths)):.1f}")


if __name__ == "__main__":
    main()