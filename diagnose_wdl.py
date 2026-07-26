"""Measure a model's value-head calibration against exact Syzygy WDL labels.

Samples random <=5-man endgame positions (side-to-move POV), probes Syzygy for
ground-truth win/draw/loss, and reports the value head's MAE and 3-class WDL
accuracy. Reproduces the diagnostic.md table so Path-B experiments have a
repeatable yardstick.
"""
import argparse
import os
import random as _random

import chess
import chess.syzygy
import torch

if torch.version.hip is not None:
    os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")

from helper import board_to_tensor
from models import net_from_state_dict
from pretrain_from_tablebase import _random_endgame_board


def _wdl_to_class_value(wdl: int) -> float:
    """Syzygy wdl (-2..2) -> value in {-1,0,+1}, side-to-move POV (|wdl|>=2 decisive)."""
    if wdl >= 2:
        return 1.0
    if wdl <= -2:
        return -1.0
    return 0.0


def _value_to_class(v: float, band: float) -> int:
    if v > band:
        return 1
    if v < -band:
        return -1
    return 0


def main():
    ap = argparse.ArgumentParser(description="Value-head WDL calibration vs Syzygy.")
    ap.add_argument("--model", "-m", required=True)
    ap.add_argument("--n", type=int, default=512, help="labeled positions to test")
    ap.add_argument("--seed", type=int, default=1401)
    ap.add_argument("--band", type=float, default=0.5,
                    help="dead-zone half-width for 3-class thresholding of the value scalar")
    ap.add_argument("--head", choices=["value", "wdl"], default="value",
                    help="value = scalar value head (thresholded); wdl = 3-logit WDL head (argmax)")
    ap.add_argument("--tablebase", default="syzygy")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = _random.Random(args.seed)
    tb = chess.syzygy.open_tablebase(args.tablebase)
    net = net_from_state_dict(torch.load(args.model, map_location=dev), dev)
    net.eval()

    boards, targets = [], []
    attempts = 0
    while len(boards) < args.n and attempts < args.n * 200:
        attempts += 1
        n_extra = rng.choice([1, 2, 3])          # 3..5 man
        b = _random_endgame_board(rng, n_extra)
        if b is None:
            continue
        try:
            wdl = tb.probe_wdl(b)
        except (chess.syzygy.MissingTableError, KeyError):
            continue
        boards.append(b)
        targets.append(_wdl_to_class_value(wdl))

    tgt = torch.tensor(targets)
    with torch.inference_mode():
        states = torch.stack([board_to_tensor(b) for b in boards]).to(dev)
        vals, cls_list = [], []
        for i in range(0, len(states), 1024):
            chunk = states[i:i + 1024]
            if args.head == "wdl":
                _, _, wdl = net(chunk, with_wdl=True)
                p = wdl.softmax(-1)
                vals.append((p[:, 0] - p[:, 2]).cpu())          # P(win) - P(loss)
                # class from argmax: 0=win->+1, 1=draw->0, 2=loss->-1
                cls_list.append((1 - wdl.argmax(-1)).cpu())
            else:
                _, v = net(chunk)
                vals.append(v.view(-1).cpu())
        vals = torch.cat(vals)

    mae = (vals - tgt).abs().mean().item()
    if args.head == "wdl":
        pred_cls = torch.cat(cls_list)
    else:
        pred_cls = torch.tensor([_value_to_class(float(v), args.band) for v in vals])
    true_cls = tgt.to(torch.long)
    acc = (pred_cls == true_cls).float().mean().item()

    # per-class mean predicted value, to expose optimism/compression
    def mean_v(cls):
        mask = true_cls == cls
        return float(vals[mask].mean()) if mask.any() else float("nan")

    print(f"model: {args.model}")
    print(f"n={len(boards)}  band=±{args.band}")
    print(f"Value MAE (vs ±1/0): {mae:.3f}")
    print(f"3-class WDL accuracy: {acc*100:.1f}%")
    print(f"mean predicted value  | true win: {mean_v(1):+.3f}  draw: {mean_v(0):+.3f}  loss: {mean_v(-1):+.3f}")
    n_w = int((true_cls == 1).sum()); n_d = int((true_cls == 0).sum()); n_l = int((true_cls == -1).sum())
    print(f"label balance         | win: {n_w}  draw: {n_d}  loss: {n_l}")


if __name__ == "__main__":
    main()
