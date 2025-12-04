import argparse
import os
from typing import Dict, Optional

import chess
import chess.engine
import torch

from train import ActorCriticResNet, pick_move_topk_value
from helper import board_to_tensor


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
        default="./stockfish/stockfish",
        help="Path to a UCI engine executable (default: /usr/bin/stockfish from RPM install).",
    )
    parser.add_argument(
        "--engine-move-time",
        type=float,
        default=0.01,
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
        default=50,
        help="Number of games to play.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override: cpu or cuda. Defaults to auto-detect.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=3,
        help="Number of top policy moves to evaluate with the value head.",
    )
    parser.add_argument(
        "--policy-temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for policy selection within top-k; <=0 for greedy.",
    )
    return parser.parse_args()


def _create_engine(path: str, skill: Optional[int]) -> chess.engine.SimpleEngine:
    engine = chess.engine.SimpleEngine.popen_uci(path)
    if skill is not None:
        try:
            engine.configure({"Skill Level": int(skill)})
        except Exception as exc:  # pragma: no cover - best-effort only
            print(f"[WARN] Could not set engine skill level: {exc}")
    return engine


def _play_game(
    net: ActorCriticResNet,
    engine: chess.engine.SimpleEngine,
    device: torch.device,
    game_index: int,
    topk: int,
    policy_temperature: float,
    engine_move_time: float,
) -> Dict[str, int]:
    board = chess.Board()
    is_policy_white = (game_index % 2 == 0)
    move_count = 0

    while not board.is_game_over():
        policy_turn = (board.turn == chess.WHITE and is_policy_white) or (
            board.turn == chess.BLACK and not is_policy_white
        )

        if policy_turn:
            state = board_to_tensor(board).unsqueeze(0).to(device)
            policy_logits, _ = net(state)
            move = pick_move_topk_value(
                board,
                net,
                policy_logits[0],
                k=max(1, topk),
                device=device,
                temperature=policy_temperature if policy_temperature > 0 else None,
            )
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
    net.load_state_dict(state)
    net.eval()

    try:
        engine = _create_engine(args.engine_path, args.engine_skill_level)
    except Exception as exc:
        print(f"[ERROR] Could not launch engine at '{args.engine_path}': {exc}")
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
                topk=args.topk,
                policy_temperature=args.policy_temperature,
                engine_move_time=args.engine_move_time,
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

            # Progress line
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
