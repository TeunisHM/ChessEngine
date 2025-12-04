import argparse
import os
from typing import Iterable, List, Optional, Tuple
from time import perf_counter

import chess
import chess.pgn
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from helper import board_to_tensor, move_to_index
from train import ActorCriticResNet


def _result_to_value(result: str) -> Optional[float]:
    """Map PGN result string to a scalar outcome from White's perspective."""
    if result == "1-0":
        return 1.0
    if result == "0-1":
        return -1.0
    if result in ("1/2-1/2", "1/2"):
        return 0.0
    return None


class PGNSupervisedDataset(Dataset):
    """Load PGN games and expose (state, policy_target, value_target) samples."""

    class _TimeBar:
        """Lightweight progress/time bar to avoid external deps."""

        def __init__(self, label: str, total: Optional[int] = None, length: int = 30, min_interval: float = 0.3) -> None:
            self.label = label
            self.total = total
            self.length = length
            self.min_interval = min_interval
            self.start = perf_counter()
            self.last_print = self.start
            self.current = 0

        def update(self, value: int, force: bool = False) -> None:
            self.current = value
            now = perf_counter()
            if (
                not force
                and now - self.last_print < self.min_interval
                and (self.total is None or value < self.total)
            ):
                return

            elapsed = now - self.start
            rate = value / elapsed if elapsed > 0 else 0.0

            if self.total:
                frac = min(1.0, value / self.total)
                filled = int(frac * self.length)
                bar = "=" * filled + "-" * (self.length - filled)
                eta = (self.total - value) / rate if rate > 0 else 0.0
                msg = (
                    f"\r{self.label} [{bar}] {value}/{self.total} | "
                    f"{elapsed:5.1f}s elapsed | ETA {eta:5.1f}s | {rate:.1f}/s"
                )
            else:
                msg = f"\r{self.label} {value} | {elapsed:5.1f}s elapsed | {rate:.1f}/s"

            print(msg, end="", flush=True)
            self.last_print = now

        def finish(self) -> None:
            self.update(self.current, force=True)
            print()

    def __init__(
        self,
        pgn_paths: Iterable[str],
        max_games: Optional[int] = None,
        max_positions: Optional[int] = None,
        min_result: float = -1.0,
    ) -> None:
        self.fens: List[str] = []
        self.policy_targets: List[int] = []
        self.value_targets: List[float] = []
        self._load_games(
            list(pgn_paths),
            max_games=max_games,
            max_positions=max_positions,
            min_result=min_result,
        )

    def _load_games(
        self,
        pgn_paths: List[str],
        max_games: Optional[int],
        max_positions: Optional[int],
        min_result: float,
    ) -> None:
        games_loaded = 0
        pos_loaded = 0
        progress = self._TimeBar(label="[LOAD positions]", total=max_positions)

        for path in pgn_paths:
            if max_games is not None and games_loaded >= max_games:
                break
            if not os.path.exists(path):
                print(f"[WARN] PGN file not found: {path}")
                continue

            with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                while True:
                    if max_games is not None and games_loaded >= max_games:
                        break
                    if max_positions is not None and pos_loaded >= max_positions:
                        break

                    game = chess.pgn.read_game(handle)
                    if game is None:
                        break

                    result = _result_to_value(game.headers.get("Result", ""))
                    if result is None:
                        continue

                    board = game.board()
                    drop_game = False
                    for move in game.mainline_moves():
                        try:
                            fen = board.fen()
                            move_idx = move_to_index(move, board)
                        except Exception as exc:
                            print(f"[WARN] Skipping move due to encoding issue: {exc}")
                            drop_game = True
                            break

                        current_player = 1.0 if board.turn == chess.WHITE else -1.0
                        value_target = result * current_player

                        if value_target >= min_result:
                            self.fens.append(fen)
                            self.policy_targets.append(move_idx)
                            self.value_targets.append(value_target)
                            pos_loaded += 1
                            progress.update(pos_loaded)
                            if max_positions is not None and pos_loaded >= max_positions:
                                board.push(move)
                                break

                        board.push(move)

                    if not drop_game:
                        games_loaded += 1

                    if max_positions is not None and pos_loaded >= max_positions:
                        break

        progress.finish()

        print(
            f"[INFO] Loaded {games_loaded} games | {len(self.fens)} positions for supervised pretraining."
        )

    def __len__(self) -> int:
        return len(self.fens)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, float]:
        board = chess.Board(self.fens[idx])
        state_tensor = board_to_tensor(board)
        return state_tensor, self.policy_targets[idx], self.value_targets[idx]


def collate_fn(
    batch: List[Tuple[torch.Tensor, int, float]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    states = torch.stack([sample[0] for sample in batch], dim=0)
    policy_targets = torch.tensor([sample[1] for sample in batch], dtype=torch.long)
    value_targets = torch.tensor([sample[2] for sample in batch], dtype=torch.float32)
    return states, policy_targets, value_targets

def supervised_pretrain(
    model: ActorCriticResNet,
    dataset: PGNSupervisedDataset,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    value_loss_weight: float,
    num_workers: int,
) -> None:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()

    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        total_policy_loss = 0.0
        total_value_loss = 0.0

        correct_moves = 0
        total_moves = 0
        epoch_bar = PGNSupervisedDataset._TimeBar(
            label=f"[Epoch {epoch}/{epochs}]",
            total=len(loader),
        )

        for batch_idx, (states, policy_targets, value_targets) in enumerate(loader, start=1):
            states = states.to(device)
            policy_targets = policy_targets.to(device)
            value_targets = value_targets.to(device)

            optimizer.zero_grad()
            policy_logits, state_values = model(states)
            policy_loss = policy_criterion(policy_logits, policy_targets)
            value_loss = value_criterion(state_values.squeeze(-1), value_targets)
            loss = policy_loss + value_loss_weight * value_loss
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                # Get the index of the highest logit
                pred_moves = torch.argmax(policy_logits, dim=1)
                correct = (pred_moves == policy_targets).sum().item()
                correct_moves += correct
                total_moves += states.size(0)

            total_policy_loss += policy_loss.item() * states.size(0)
            total_value_loss += value_loss.item() * states.size(0)
            epoch_bar.update(batch_idx)

        epoch_bar.finish()

        dataset_size = len(dataset)
        avg_policy = total_policy_loss / max(1, dataset_size)
        avg_value = total_value_loss / max(1, dataset_size)
        accuracy = correct_moves / max(1, total_moves)

        print(
            f"[Epoch {epoch}/{epochs}] "
            f"Policy Loss: {avg_policy:.4f} | Value Loss: {avg_value:.4f} "
            f"Acc: {accuracy*100:.2f}%"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Supervised pretraining on PGN games before RL fine-tuning."
    )
    parser.add_argument(
        "--pgn",
        nargs="+",
        required=True,
        help="Paths to PGN files containing training games.",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Maximum number of games to load across all PGNs.",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=2500000,
        help="Maximum number of positions to keep for training.",
    )
    parser.add_argument(
        "--min-result",
        type=float,
        default=-0.01,
        help="Minimum eventual outcome for the side to move (-1 loss, 0 draw, 1 win).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Supervised batch size.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument(
        "--value-loss-weight",
        type=float,
        default=0.5,
        help="Relative weight for the value regression term.",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cpu or cuda device string. Defaults to auto-detect.",
    )
    parser.add_argument(
        "--init-model",
        type=str,
        default=None,
        help="Optional .pt file to initialize weights before pretraining.",
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default="actor_critic_chess_resnet_pgn.pt",
        help="Where to save the pretrained weights.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = (
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    dataset = PGNSupervisedDataset(
        args.pgn,
        max_games=args.max_games,
        max_positions=args.max_positions,
        min_result=args.min_result,
    )
    if len(dataset) == 0:
        print("[ERROR] No training samples were loaded from the PGNs.")
        return

    model = ActorCriticResNet()
    if args.init_model:
        if os.path.exists(args.init_model):
            print(f"[INFO] Loading initial weights from {args.init_model}")
            state_dict = torch.load(args.init_model, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
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
    print(f"[INFO] Saved pretrained weights to {args.output_model}")


if __name__ == "__main__":
    main()
