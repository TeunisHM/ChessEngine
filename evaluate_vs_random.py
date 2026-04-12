import random
import chess
import torch
from typing import Optional
from time import perf_counter

from helper import board_to_tensor, legal_moves_mask
from search import search_select_move, DEFAULT_SEARCH_DEPTH


try:
    inference_mode = torch.inference_mode
except AttributeError:
    inference_mode = torch.no_grad


def evaluate_vs_random(actor_critic_net,
                      game_num,
                      num_games: int = 100,
                      writer=None,
                      show_progress: bool = True,
                      device: str = "cpu",
                      search_temperature: float = 0.01,
                      search_k: int = 5,
                      search_depth: int = DEFAULT_SEARCH_DEPTH):
    """
    Evaluates the actor-critic model's policy against a random opponent.
    """
    entropies = []
    actor_critic_net.eval()

    results = []
    white_wins = black_wins = ties = policy_white_wins = policy_black_wins = 0
    game_lengths = []

    print(f"evaluating model with k={search_k} and depth={search_depth}")

    with inference_mode():
        start_t = perf_counter()
        for i in range(num_games):
            board = chess.Board()
            is_policy_white = i % 2 == 0
            move_count = 0

            while not board.is_game_over():
                if (board.turn == chess.WHITE and is_policy_white) or (
                    board.turn == chess.BLACK and not is_policy_white
                ):
                    state = board_to_tensor(board).unsqueeze(0).to(device)
                    policy_logits, _ = actor_critic_net(state)
                    move, _, entropy = search_select_move(
                        board=board,
                        actor_critic_net=actor_critic_net,
                        logits=policy_logits[0],
                        device=device,
                        k=search_k,
                        temperature=search_temperature,
                        depth=search_depth,
                    )
                    if entropy is not None:
                        entropies.append(float(entropy.item()) if hasattr(entropy, "item") else float(entropy))
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
                    f"\rEvaluating [{bar}] {done}/{num_games} | {elapsed:5.1f}s elapsed | ETA {eta:5.1f}s",
                    end="",
                    flush=True,
                )

    actor_critic_net.train()

    if show_progress:
        print()

    wins = results.count(1)
    losses = results.count(-1)

    print("\n--- Evaluation Results ---")
    print(f"Policy vs Random: {wins} Wins / {ties} Draws / {losses} Losses")
    print(f"Policy as White Wins: {policy_white_wins}, Policy as Black Wins: {policy_black_wins}")
    print(f"--------------------------\n")

    if writer:
        writer.add_scalar("Eval/Wins", wins, game_num)
        writer.add_scalar("Eval/Losses", losses, game_num)
        writer.add_scalar("Eval/Draws", ties, game_num)
        writer.add_scalar("Eval/PolicyWhiteWins", policy_white_wins, game_num)
        writer.add_scalar("Eval/PolicyBlackWins", policy_black_wins, game_num)
        writer.add_scalar("Eval/AvgGameLength", sum(game_lengths) / len(game_lengths), game_num)
        writer.add_scalar("Eval/AvgEntropy", sum(entropies) / len(entropies), game_num)

    return {
        "white_wins": white_wins,
        "black_wins": black_wins,
        "draws": ties,
        "wins": wins,
        "losses": losses,
        "policy_white_wins": policy_white_wins,
        "policy_black_wins": policy_black_wins,
        "avg_game_length": sum(game_lengths) / len(game_lengths) if game_lengths else 0,
        "avg_entropy": sum(entropies) / len(entropies) if entropies else 0,
    }

if __name__ == "__main__":
    evaluate_vs_random()