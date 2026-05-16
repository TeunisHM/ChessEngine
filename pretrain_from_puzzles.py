"""Supervised pretraining on Lichess puzzles.

Each puzzle's `Moves` field is the alternating sequence
[opponent_blunder, your_response, opponent_reply, your_response, ...]
applied to FEN. Positions where it's the solver's turn (after odd-indexed
moves have been applied) become training samples whose policy target is the
indicated correct response and whose value target is +1 (you have a winning
tactic).

Filters allow a curriculum: e.g. start with --themes-required mateIn1 and a
low --max-rating, then re-run with broader themes / higher rating using
--init-model on the prior checkpoint.
"""
import argparse
import csv
import os
import sys
from typing import List, Optional, Tuple

import chess
import torch
from torch.utils.data import Dataset

from helper import board_to_tensor, move_to_index
from models import ActorCriticResNet, load_actor_critic_state_dict
from pretrain_from_pgn import PGNSupervisedDataset, supervised_pretrain

csv.field_size_limit(sys.maxsize)

_TimeBar = PGNSupervisedDataset._TimeBar


class PuzzleSupervisedDataset(Dataset):
    """Lichess puzzle CSV → (state, correct_move_idx, value_target=+1)."""

    def __init__(
        self,
        csv_path: str,
        max_puzzles: Optional[int] = None,
        max_positions: Optional[int] = None,
        max_rating: Optional[int] = None,
        min_rating: Optional[int] = None,
        themes_required: Optional[List[str]] = None,
    ) -> None:
        self.fens: List[str] = []
        self.policy_targets: List[int] = []
        self.value_targets: List[float] = []
        self._load(
            csv_path,
            max_puzzles=max_puzzles,
            max_positions=max_positions,
            max_rating=max_rating,
            min_rating=min_rating,
            themes_required=set(themes_required) if themes_required else None,
        )

    def _load(self, csv_path, max_puzzles, max_positions, max_rating,
              min_rating, themes_required):
        bar = _TimeBar(label="[LOAD puzzles]", total=max_positions)
        n_puzzles = 0
        n_positions = 0

        with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if max_puzzles is not None and n_puzzles >= max_puzzles:
                    break
                if max_positions is not None and n_positions >= max_positions:
                    break

                try:
                    rating = int(row.get("Rating", "0") or 0)
                except ValueError:
                    rating = 0
                if min_rating is not None and rating < min_rating:
                    continue
                if max_rating is not None and rating > max_rating:
                    continue

                themes = (row.get("Themes") or "").split()
                if themes_required and not themes_required.intersection(themes):
                    continue

                fen = (row.get("FEN") or "").strip()
                moves_str = (row.get("Moves") or "").strip()
                if not fen or not moves_str:
                    continue

                try:
                    board = chess.Board(fen)
                except Exception:
                    continue

                moves = moves_str.split()
                drop = False
                for i, uci in enumerate(moves):
                    try:
                        move = chess.Move.from_uci(uci)
                    except Exception:
                        drop = True
                        break
                    if move not in board.legal_moves:
                        drop = True
                        break

                    # Odd i: solver's turn before this move; emit sample.
                    if i % 2 == 1:
                        try:
                            move_idx = move_to_index(move, board)
                        except Exception:
                            drop = True
                            break
                        self.fens.append(board.fen())
                        self.policy_targets.append(move_idx)
                        self.value_targets.append(1.0)
                        n_positions += 1
                        bar.update(n_positions)
                        if max_positions is not None and n_positions >= max_positions:
                            board.push(move)
                            break

                    board.push(move)

                if not drop:
                    n_puzzles += 1

        bar.finish()
        print(f"[INFO] Loaded {n_puzzles} puzzles | {n_positions} solver positions.")

    def __len__(self) -> int:
        return len(self.fens)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, float]:
        board = chess.Board(self.fens[idx])
        state_tensor = board_to_tensor(board)
        return state_tensor, self.policy_targets[idx], self.value_targets[idx]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Supervised pretraining on Lichess puzzles."
    )
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to lichess_db_puzzle.csv")
    parser.add_argument("--max-puzzles", type=int, default=None)
    parser.add_argument("--max-positions", type=int, default=500_000)
    parser.add_argument("--max-rating", type=int, default=None,
                        help="Skip puzzles harder than this rating.")
    parser.add_argument("--min-rating", type=int, default=None)
    parser.add_argument("--themes-required", type=str, default=None,
                        help="Comma-separated themes; puzzles must include at least one "
                             "(e.g. 'mateIn1,mateIn2,endgame').")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--value-loss-weight", type=float, default=1.5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--init-model", type=str, default=None,
                        help="Optional .pt file to initialize weights from.")
    parser.add_argument("--output-model", type=str, default="puzzle_pretrained.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = (
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    themes = [t.strip() for t in args.themes_required.split(",")] if args.themes_required else None

    dataset = PuzzleSupervisedDataset(
        args.csv,
        max_puzzles=args.max_puzzles,
        max_positions=args.max_positions,
        max_rating=args.max_rating,
        min_rating=args.min_rating,
        themes_required=themes,
    )
    if len(dataset) == 0:
        print("[ERROR] No samples passed the filters.")
        return

    model = ActorCriticResNet()
    if args.init_model:
        if os.path.exists(args.init_model):
            print(f"[INFO] Loading initial weights from {args.init_model}")
            state_dict = torch.load(args.init_model, map_location="cpu")
            load_actor_critic_state_dict(model, state_dict)
        else:
            print(f"[WARN] init-model path not found: {args.init_model}")

    supervised_pretrain(
        model=model,
        dataset=dataset,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        value_loss_weight=args.value_loss_weight,
        num_workers=args.num_workers,
    )

    torch.save(model.state_dict(), args.output_model)
    print(f"[INFO] Saved to {args.output_model}")


if __name__ == "__main__":
    main()
