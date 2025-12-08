import argparse
import os
import torch

from models import ActorCriticResNet
from evaluate_vs_random import evaluate_vs_random
from search import DEFAULT_SEARCH_DEPTH


class NoOpWriter:
    """should probably just make writer optional in function that is used"""
    def add_scalar(self, *args, **kwargs):
        pass

def main():
    parser = argparse.ArgumentParser(description="Load a model and evaluate vs random.")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--model", "-m", type=str, help="Path to .pt model file")
    mode_group.add_argument(
        "--zero-policy",
        action="store_true",
        help="Use a zero-initialized network (uniform policy logits) instead of loading a model",
    )
    parser.add_argument("--games", "-g", type=int, default=100, help="Number of games to play")
    parser.add_argument("--device", type=str, default=None, help="Device override: cpu or cuda")
    parser.add_argument(
        "--search-k",
        type=int,
        default=5,
        help="Top-k policy moves to explore in the lightweight search selector.",
    )
    parser.add_argument(
        "--search-temperature",
        type=float,
        default=0.01,
        help="Temperature over search scores; <=0 makes selection greedy.",
    )
    parser.add_argument(
        "--search-depth",
        type=int,
        default=DEFAULT_SEARCH_DEPTH,
        help="Search depth (plies) for the lightweight selector (>=1).",
    )
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.zero_policy:
        print("Creating zero-initialized ActorCriticResNet for uniform policy evaluation.")
        net = ActorCriticResNet().to(device)
        with torch.no_grad():
            for param in net.parameters():
                param.zero_()
    else:
        if not os.path.exists(args.model):
            print(f"Model file not found: {args.model}")
            raise SystemExit(1)

        net = ActorCriticResNet().to(device)
        print(f"Loading model: {args.model}")
        state = torch.load(args.model, map_location=device)
        net.load_state_dict(state)
    net.eval()

    writer = NoOpWriter()
    stats = evaluate_vs_random(net, game_num=0, num_games=args.games, writer=writer, show_progress=True, search_k=args.search_k, search_depth=args.search_depth)

    print("\nSummary:")
    print(f"Wins: {stats['wins']}  Draws: {stats['draws']}  Losses: {stats['losses']}")
    print(f"Policy as White Wins: {stats['policy_white_wins']} | Policy as Black Wins: {stats['policy_black_wins']}")
    print(f"Avg game length: {stats['avg_game_length']:.2f} | Avg entropy: {stats['avg_entropy']:.4f}")


if __name__ == "__main__":
    main()
