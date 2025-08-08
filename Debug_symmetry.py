import chess
import torch
from helper import board_to_tensor

def show_plane_diffs(board):
    t_orig = board_to_tensor(board)

    # Mirror the board
    mirrored = board.mirror()      # a mirrored Board object
    # Two sensible possibilities to compare:
    # A) mirror without toggling turn
    mirrored_a = mirrored
    # B) mirror with toggling turn (what you tried earlier)
    mirrored_b = mirrored.copy()
    mirrored_b.turn = not board.turn

    t_a = board_to_tensor(mirrored_a)
    t_b = board_to_tensor(mirrored_b)

    print("Original FEN:", board.fen())
    print("Mirrored FEN:", mirrored_a.fen())
    print("Mirrored (toggled-turn) FEN:", mirrored_b.fen())
    print()

    def per_plane_report(t1, t2, label):
        diffs = ((t1 - t2).abs()).sum(dim=(1,2)).tolist()
        print(f"Comparison: original vs {label}")
        for i, d in enumerate(diffs):
            if d > 1e-6:
                print(f"  Plane {i:02d}: L1 diff = {d:.6f} | sum_orig = {t1[i].sum().item():.6f} | sum_other = {t2[i].sum().item():.6f}")
        if all(d <= 1e-6 for d in diffs):
            print("  -> All planes equal!")
        print()

    per_plane_report(t_orig, t_a, "mirror (no toggle)")
    per_plane_report(t_orig, t_b, "mirror (with toggled turn)")

if __name__ == "__main__":
    # starting position
    b = chess.Board()
    print("=== STARTING POSITION ===")
    show_plane_diffs(b)

    # random positions
    import random
    for i in range(10):
        b = chess.Board()
        for _ in range(random.randint(5,30)):
            if b.is_game_over(): break
            b.push(random.choice(list(b.legal_moves)))
        print(f"\n=== RANDOM POS {i+1} ===")
        show_plane_diffs(b)