import argparse
import os
import random as _random
from time import perf_counter
from typing import List, Optional, Tuple

import chess
import chess.syzygy
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

if torch.version.hip is not None:
    os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")

from helper import (
    board_to_tensor,
    mirror_action_index,
    mirror_board_tensor,
    move_to_index,
)
from models import (
    ActorCriticResNet,
    DEFAULT_NUM_FILTERS,
    DEFAULT_NUM_RESIDUAL_BLOCKS,
    load_actor_critic_state_dict,
)


_PIECE_CHOICES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]


def _random_endgame_board(rng: _random.Random, n_extra: int) -> Optional[chess.Board]:
    """Random valid 2+n_extra-piece position. None on invalid/terminal/adjacent kings."""
    squares = list(range(64))
    rng.shuffle(squares)

    wk, bk = squares[0], squares[1]
    if chess.square_distance(wk, bk) <= 1:
        return None

    board = chess.Board.empty()
    board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))

    placed = 0
    for sq in squares[2:]:
        if placed == n_extra:
            break
        piece_type = rng.choice(_PIECE_CHOICES)
        color = rng.choice([chess.WHITE, chess.BLACK])
        if piece_type == chess.PAWN:
            r = chess.square_rank(sq)
            if r == 0 or r == 7:
                continue
        board.set_piece_at(sq, chess.Piece(piece_type, color))
        placed += 1

    if placed != n_extra:
        return None

    board.turn = rng.choice([chess.WHITE, chess.BLACK])
    if not board.is_valid():
        return None
    if board.is_game_over():
        return None
    return board


def _label_position(
    board: chess.Board, tb: chess.syzygy.Tablebase
) -> Optional[Tuple[List[int], float]]:
    """Return (best_move_indices, value_target) from side-to-move POV, or None on TB miss."""
    try:
        wdl = tb.probe_wdl(board)
    except (chess.syzygy.MissingTableError, KeyError):
        return None

    if wdl >= 2:
        value = 1.0
    elif wdl <= -2:
        value = -1.0
    else:
        value = 0.0

    best_score: Optional[int] = None
    best_moves: List[chess.Move] = []
    for move in board.legal_moves:
        board.push(move)
        try:
            child_wdl = tb.probe_wdl(board)
        except (chess.syzygy.MissingTableError, KeyError):
            board.pop()
            return None
        board.pop()
        score = -child_wdl
        if best_score is None or score > best_score:
            best_score = score
            best_moves = [move]
        elif score == best_score:
            best_moves.append(move)

    indices: List[int] = []
    for m in best_moves:
        try:
            indices.append(move_to_index(m, board))
        except Exception:
            continue
    if not indices:
        return None

    return indices, value


class _TimeBar:
    def __init__(self, label: str, total: Optional[int] = None,
                 length: int = 30, min_interval: float = 0.3) -> None:
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


class TablebaseDataset(Dataset):
    """Rejection-sampled random 3-5 piece positions labeled from Syzygy."""

    def __init__(self, tb_path: str, n_samples: int, seed: int = 0) -> None:
        self.fens: List[str] = []
        self.best_indices: List[List[int]] = []
        self.values: List[float] = []
        rng = _random.Random(seed)
        n_extra_cycle = (1, 2, 3)

        tb = chess.syzygy.open_tablebase(tb_path)
        bar = _TimeBar(label="[SAMPLE TB positions]", total=n_samples)
        attempts = 0
        try:
            ci = 0
            while len(self.fens) < n_samples:
                attempts += 1
                n_extra = n_extra_cycle[ci % 3]
                ci += 1
                board = _random_endgame_board(rng, n_extra)
                if board is None:
                    continue
                labeled = _label_position(board, tb)
                if labeled is None:
                    continue
                indices, value = labeled
                self.fens.append(board.fen())
                self.best_indices.append(indices)
                self.values.append(value)
                bar.update(len(self.fens))
        finally:
            bar.finish()
            tb.close()
        accept_rate = len(self.fens) / max(1, attempts)
        print(
            f"[INFO] Sampled {len(self.fens)} positions from {attempts} attempts "
            f"(accept rate {accept_rate:.2%})."
        )

    def __len__(self) -> int:
        return len(self.fens)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, float]:
        board = chess.Board(self.fens[idx])
        state = board_to_tensor(board)
        action_idx = _random.choice(self.best_indices[idx])
        value = self.values[idx]
        if _random.random() < 0.5:
            state = mirror_board_tensor(state)
            action_idx = mirror_action_index(action_idx)
        return state, action_idx, value


def collate_fn(
    batch: List[Tuple[torch.Tensor, int, float]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    states = torch.stack([s[0] for s in batch], 0)
    actions = torch.tensor([s[1] for s in batch], dtype=torch.long)
    values = torch.tensor([s[2] for s in batch], dtype=torch.float32)
    return states, actions, values


def supervised_pretrain(
    model: ActorCriticResNet,
    dataset: TablebaseDataset,
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
        total_p = 0.0
        total_v = 0.0
        correct = 0
        total = 0
        bar = _TimeBar(label=f"[Epoch {epoch}/{epochs}]", total=len(loader))

        for bi, (states, actions, values) in enumerate(loader, start=1):
            states = states.to(device)
            actions = actions.to(device)
            values = values.to(device)

            optimizer.zero_grad()
            logits, vhat = model(states)
            ploss = policy_criterion(logits, actions)
            vloss = value_criterion(vhat.squeeze(-1), values)
            loss = ploss + value_loss_weight * vloss
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred = torch.argmax(logits, dim=1)
                correct += (pred == actions).sum().item()
                total += states.size(0)

            total_p += ploss.item() * states.size(0)
            total_v += vloss.item() * states.size(0)
            bar.update(bi)

        bar.finish()
        ds = max(1, len(dataset))
        print(
            f"[Epoch {epoch}/{epochs}] "
            f"Policy: {total_p/ds:.4f} Value: {total_v/ds:.4f} "
            f"Acc: {100*correct/max(1,total):.2f}%"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Supervised pretraining on Syzygy 3-5 piece tablebase positions."
    )
    p.add_argument("--tablebase", default="syzygy", help="Path to Syzygy directory.")
    p.add_argument("--n-samples", type=int, default=200000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--value-loss-weight", type=float, default=1.5)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--se", action="store_true",
        help="Build the net with squeeze-excitation blocks in the residual tower.",
    )
    p.add_argument(
        "--num-filters", type=int, default=DEFAULT_NUM_FILTERS,
        help="Residual tower channel width.",
    )
    p.add_argument(
        "--num-residual-blocks", type=int, default=DEFAULT_NUM_RESIDUAL_BLOCKS,
        help="Number of residual blocks in the tower.",
    )
    p.add_argument("--init-model", default="pretrained.pt",
                   help="Optional .pt to initialize from.")
    p.add_argument("--output-model", default="pretrained_tb.pt")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = (
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if not os.path.isdir(args.tablebase):
        print(f"[ERROR] Tablebase dir not found: {args.tablebase}")
        return

    dataset = TablebaseDataset(args.tablebase, args.n_samples, seed=args.seed)
    if len(dataset) == 0:
        print("[ERROR] No samples generated.")
        return

    model = ActorCriticResNet(
        use_se=args.se,
        num_filters=args.num_filters,
        num_residual_blocks=args.num_residual_blocks,
    )
    if args.init_model and os.path.exists(args.init_model):
        print(f"[INFO] Loading initial weights from {args.init_model}")
        state_dict = torch.load(args.init_model, map_location="cpu")
        load_actor_critic_state_dict(model, state_dict)
    elif args.init_model:
        print(f"[WARN] init-model not found: {args.init_model}; fresh weights")

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
