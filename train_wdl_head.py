"""Train ONLY the separate WDL head on objective game outcomes, trunk frozen.

Cheap gate for the Path-A design: does a detached WDL head (its own adapter on
frozen shared features) reach usable calibration? If diagnose_wdl.py --head wdl
hits ~80%, wire it into search + RL; if it plateaus low, the shared features lack
clean WDL info and we need a small trunk LR + policy-preservation term instead.
"""
import argparse
import os
from time import perf_counter

import chess
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

if torch.version.hip is not None:
    os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")

from helper import board_to_tensor
from models import net_from_state_dict
from pretrain_from_pgn import PGNSupervisedDataset


def _cls(vt: float) -> int:
    # PGN value_target sign -> class: 0 win, 1 draw, 2 loss (mover POV)
    return 0 if vt > 0 else (2 if vt < 0 else 1)


class _View(Dataset):
    def __init__(self, ds):
        self.fens, self.vt = ds.fens, ds.value_targets
    def __len__(self):
        return len(self.fens)
    def __getitem__(self, i):
        return self.fens[i], _cls(self.vt[i])


def _collate(batch):
    states = torch.stack([board_to_tensor(chess.Board(f)) for f, _ in batch])
    cls = torch.tensor([c for _, c in batch], dtype=torch.long)
    return states, cls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pgn", nargs="+", default=["lichess_elite_2025-07.pgn"])
    ap.add_argument("--max-positions", type=int, default=120000)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    net = net_from_state_dict(torch.load(args.init, map_location=dev), dev)
    for p in net.parameters():
        p.requires_grad = False
    for p in net.wdl_head.parameters():
        p.requires_grad = True

    print(f"[data] loading up to {args.max_positions} PGN positions ...")
    ds = PGNSupervisedDataset(args.pgn, max_positions=args.max_positions, value_discount=0.99)
    loader = DataLoader(_View(ds), batch_size=args.batch_size, shuffle=True,
                        collate_fn=_collate, num_workers=4, drop_last=True)

    opt = torch.optim.AdamW(net.wdl_head.parameters(), lr=args.lr)
    for ep in range(args.epochs):
        t0 = perf_counter()
        tot = 0.0; n = 0; correct = 0; total = 0
        for states, cls in loader:
            states, cls = states.to(dev), cls.to(dev)
            _, _, wdl = net(states, with_wdl=True)   # trunk detached inside forward
            loss = F.cross_entropy(wdl, cls)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); n += 1
            correct += (wdl.argmax(-1) == cls).sum().item(); total += cls.numel()
        print(f"[epoch {ep+1}/{args.epochs}] CE {tot/max(n,1):.4f} "
              f"train-acc {correct/max(total,1)*100:.1f}% | {perf_counter()-t0:.1f}s")

    torch.save(net.state_dict(), args.out)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
