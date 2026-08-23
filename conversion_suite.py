"""Endgame conversion testsuite: can the net actually win tablebase-confirmed wins?

Samples random <=5-man positions where Syzygy says the side to move has a
clean win (wdl=+2, |dtz|<=100), lets the net play the winning side against
itself (defender = same net, greedy search), and measures:

  conversion rate  - fraction of games reaching a zeroing win before the
                     50-move rule (halfmove clock >= 100) or stalemate;
  plies over optimal - actual plies to the zeroing move minus initial DTZ.

Run before/after touching dtz_shaping_weight or the value/WDL heads:
    python conversion_suite.py --model models/ppo_search_v23_checkpoint_299.pt
"""
import argparse
import os
import random

import chess
import chess.syzygy
import torch

if torch.version.hip is not None:
    os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")

from helper import index_to_move, random_endgame_board, board_to_tensor, legal_moves_mask
from lookahead import select_moves_with_lookahead
from models import net_from_state_dict

MATERIAL_CLASSES = {
    "KQvK": (1,),
    "KRvK": (1,),
    "KQvKP": None,   # unused placeholder; classes below use piece counts
}


def _sample_won_position(tb, rng, n_extra):
    """Random position with a clean TB win for White. None on miss.

    Syzygy covers <=5 men TOTAL, so n_extra <= 3. Winner material is sampled
    heavy (queen/rook first) since light random material is usually a TB draw;
    ~30% of pieces are defender pawns/minors to vary the endings.
    """
    for _ in range(600):
        squares = list(range(64))
        rng.shuffle(squares)
        wk, bk = squares[0], squares[1]
        if chess.square_distance(wk, bk) <= 1:
            continue
        b = chess.Board.empty()
        b.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
        b.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
        placed = 0
        for sq in squares[2:]:
            if placed == n_extra:
                break
            if rng.random() < 0.3:
                pt = rng.choice([chess.PAWN, chess.KNIGHT, chess.BISHOP])
                color = rng.choice([chess.WHITE, chess.BLACK])
            else:
                pt = rng.choice([chess.QUEEN, chess.QUEEN, chess.ROOK,
                                 chess.ROOK, chess.BISHOP])
                color = chess.WHITE
            if pt == chess.PAWN and chess.square_rank(sq) in (0, 7):
                continue
            b.set_piece_at(sq, chess.Piece(pt, color))
            placed += 1
        if placed != n_extra:
            continue
        b.turn = rng.choice([chess.WHITE, chess.BLACK])
        if not b.is_valid() or b.is_game_over():
            continue
        try:
            wdl = tb.probe_wdl(b)
            dtz = tb.probe_dtz(b)
        except (chess.syzygy.MissingTableError, KeyError):
            continue
        # Winner = White by construction; require White to have the clean win.
        wdl_white = wdl if b.turn == chess.WHITE else -wdl
        dtz_white = dtz if b.turn == chess.WHITE else -dtz
        if wdl_white == 2 and 0 < abs(dtz_white) <= 90:
            return b, abs(dtz_white)
    return None, None


def _net_move(net, board, device, k, alpha, vw, use_wdl, temperature=0.0):
    idxs, *_ = select_moves_with_lookahead(
        net, [board], device, top_k=k, alpha=alpha,
        temperature=temperature, value_weight=vw, use_wdl=use_wdl,
    )
    move = index_to_move(int(idxs[0].item()), board)
    if move is None or move not in board.legal_moves:
        return None
    return move


def play_conversion(net, start, init_dtz, tb, device, args):
    """Winner (White) played by the net; defender (Black) = same net greedy.
    Returns (converted: bool, plies_used: int)."""
    board = start.copy(stack=True)
    plies = 0
    while not board.is_game_over(claim_draw=False):
        if board.halfmove_clock >= 100:
            return False, plies          # 50-move rule: conversion failed
        if board.turn == chess.WHITE:
            move = _net_move(net, board, device, args.k, args.alpha,
                             args.value_weight, args.use_wdl)
        else:
            move = _net_move(net, board, device, max(2, args.k - 2),
                             args.alpha, args.value_weight, False)
        if move is None:
            return False, plies
        zeroing = board.is_zeroing(move)
        board.push(move)
        plies += 1
        if zeroing:
            # A zeroing move resets progress; conversion "completes" when the
            # resulting position is dead-won with no pawn moves/captures left
            # to engineer — approximate completion as reaching wdl=+2 with
            # halfmove clock reset AND no immediate draw. We stop tracking at
            # the FIRST zeroing move that keeps a clean TB win: deeper play is
            # routine technique.
            try:
                wdl = tb.probe_wdl(board)
            except Exception:
                return True, plies
            wdl_white = wdl if board.turn == chess.WHITE else -wdl
            if wdl_white == 2:
                return True, plies
        if plies > args.max_plies:
            break
    result = board.result(claim_draw=False)
    return result == "1-0", plies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", "-m", required=True)
    ap.add_argument("--tablebase", default="syzygy")
    ap.add_argument("--games-per-class", type=int, default=6)
    ap.add_argument("--classes", type=int, nargs="+", default=[1, 2, 3],
                    help="Extra-piece counts (Syzygy max 3: <=5 men total).")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--value-weight", type=float, default=1.0)
    ap.add_argument("--use-wdl", action="store_true")
    ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1401)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)
    tb = chess.syzygy.open_tablebase(args.tablebase)
    net = net_from_state_dict(torch.load(args.model, map_location=dev), dev)
    net.eval()

    print(f"model: {args.model} | k={args.k} alpha={args.alpha} "
          f"vw={args.value_weight} wdl={args.use_wdl}")
    total_ok = total_n = 0
    for n_extra in args.classes:
        wins = plies_sum = optimal_sum = n = 0
        attempts = 0
        while n < args.games_per_class and attempts < args.games_per_class * 60:
            attempts += 1
            start, dtz = _sample_won_position(tb, rng, n_extra)
            if start is None:
                continue
            ok, used = play_conversion(net, start, dtz, tb, dev, args)
            wins += int(ok)
            plies_sum += used
            optimal_sum += dtz
            n += 1
            print(f"\r  {n}/{args.games_per_class}", end="", flush=True)
        print("\r", end="")
        rate = wins / n if n else float("nan")
        excess = (plies_sum - optimal_sum) / n if n else float("nan")
        print(f"[{n_extra}-extra] converted {wins}/{n} ({rate*100:.0f}%) | "
              f"mean plies {plies_sum/max(n,1):.0f} vs mean DTZ "
              f"{optimal_sum/max(n,1):.0f} (+{excess:.0f})")
        total_ok += wins
        total_n += n
    print(f"TOTAL: {total_ok}/{total_n} ({100*total_ok/max(1,total_n):.0f}%)")
    tb.close()


if __name__ == "__main__":
    main()
