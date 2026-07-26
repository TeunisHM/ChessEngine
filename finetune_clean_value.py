"""Recalibrate an RL checkpoint's value head on clean game outcomes.

Modes:
  (default)          re-init value head, train it only (trunk frozen).
  --unfreeze-trunk   also fine-tune trunk on value MSE (recovers discrimination).
  --joint            train policy head (CE on PGN moves) AND value head (MSE on
                     clean side-to-move outcomes) with the trunk unfrozen, so the
                     recalibrated model stays playable. This is the Path-B recipe:
                     one clean value head, no second WDL head.

Calibration is measured separately by diagnose_wdl.py.
"""
import argparse
import os
from time import perf_counter

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

if torch.version.hip is not None:
    os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")

import chess
from helper import board_to_tensor
from models import net_from_state_dict
from pretrain_from_pgn import PGNSupervisedDataset


def _trunk(net, x):
    return net.transformer(net.residual_tower(net.stem(x)))


def _reinit(module):
    for m in module.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)


def _collate(batch):
    states = torch.stack([board_to_tensor(chess.Board(fen)) for fen, _, _ in batch])
    pols = torch.tensor([p for _, p, _ in batch], dtype=torch.long)
    vals = torch.tensor([v for _, _, v in batch], dtype=torch.float32)
    return states, pols, vals


class _View(torch.utils.data.Dataset):
    def __init__(self, ds):
        self.fens, self.pols, self.vals = ds.fens, ds.policy_targets, ds.value_targets
    def __len__(self):
        return len(self.fens)
    def __getitem__(self, i):
        return self.fens[i], self.pols[i], self.vals[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pgn", nargs="+", default=["lichess_elite_2025-07.pgn"])
    ap.add_argument("--max-positions", type=int, default=80000)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3, help="LR for the (re-init) value head")
    ap.add_argument("--unfreeze-trunk", action="store_true", default=False)
    ap.add_argument("--joint", action="store_true", default=False,
                    help="also train policy head (CE) with trunk unfrozen; keeps model playable")
    ap.add_argument("--trunk-lr", type=float, default=1e-4, help="LR for trunk (and policy head in --joint)")
    ap.add_argument("--policy-weight", type=float, default=1.0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    net = net_from_state_dict(torch.load(args.init, map_location=dev), dev)
    train_trunk = args.unfreeze_trunk or args.joint

    for p in net.parameters():
        p.requires_grad = False
    _reinit(net.value_head)
    for p in net.value_head.parameters():
        p.requires_grad = True
    param_groups = [{"params": net.value_head.parameters(), "lr": args.lr}]

    if train_trunk:
        trunk_params = [p for m in (net.stem, net.residual_tower, net.transformer) for p in m.parameters()]
        for p in trunk_params:
            p.requires_grad = True
        param_groups.append({"params": trunk_params, "lr": args.trunk_lr})
    if args.joint:
        for p in net.policy_head.parameters():
            p.requires_grad = True
        param_groups.append({"params": net.policy_head.parameters(), "lr": args.trunk_lr})
    mode = "JOINT policy+value (unfrozen)" if args.joint else \
           ("value + unfrozen trunk" if args.unfreeze_trunk else "value only (frozen trunk)")
    print(f"[mode] {mode}")
    net.train()

    print(f"[data] loading up to {args.max_positions} PGN positions ...")
    ds = PGNSupervisedDataset(args.pgn, max_positions=args.max_positions, value_discount=0.99)
    loader = DataLoader(_View(ds), batch_size=args.batch_size, shuffle=True,
                        collate_fn=_collate, num_workers=4, drop_last=True)

    opt = torch.optim.AdamW(param_groups)
    for ep in range(args.epochs):
        t0 = perf_counter()
        v_tot = p_tot = 0.0
        n = 0
        for states, pols, vals in loader:
            states, pols, vals = states.to(dev), pols.to(dev), vals.to(dev)
            ctx = torch.enable_grad() if train_trunk else torch.no_grad()
            with ctx:
                feats = _trunk(net, states)
            pred_v = net.value_head(feats).view(-1)
            v_loss = F.mse_loss(pred_v, vals)
            if args.joint:
                logits = net.policy_head(feats)
                p_loss = F.cross_entropy(logits, pols)
                loss = v_loss + args.policy_weight * p_loss
            else:
                p_loss = torch.zeros((), device=dev)
                loss = v_loss
            opt.zero_grad(); loss.backward(); opt.step()
            v_tot += v_loss.item(); p_tot += float(p_loss); n += 1
        msg = f"[epoch {ep+1}/{args.epochs}] value MSE {v_tot/max(n,1):.4f}"
        if args.joint:
            msg += f" | policy CE {p_tot/max(n,1):.4f}"
        print(msg + f" | {perf_counter()-t0:.1f}s")

    torch.save(net.state_dict(), args.out)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
