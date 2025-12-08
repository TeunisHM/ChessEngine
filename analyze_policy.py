import argparse
import os
from datetime import datetime
import chess
import torch
from typing import Optional
from helper import board_to_tensor, legal_moves_mask, index_to_move, PIECE_VALUES
from models import ActorCriticResNet
import time


def load_network(model_path: Optional[str], device: torch.device) -> ActorCriticResNet:
    """
    Initialize an ActorCriticResNet and (optionally) load weights from disk.
    """
    net = ActorCriticResNet().to(device)
    if model_path and os.path.exists(model_path):
        print(f"Loading model: {model_path}")
        state = torch.load(model_path, map_location=device)
        net.load_state_dict(state)
    else:
        if model_path:
            print(f"Model file '{model_path}' not found; using randomly initialized network.")
        else:
            print("Using randomly initialized network (no model file provided).")
    net.eval()
    return net

@torch.no_grad()
def topk_policy_with_values(board, net, logits, k=5, gamma=0.995):
    """
    Returns a list of (move, prob, logit, value_score) for the top-K legal moves.

    - prob: softmax probability over legal moves
    - logit: masked logit for the move
    - value_score: immediate reward (captures) + gamma * (-V(s')) if non-terminal, or terminal reward
    """
    device = next(net.parameters()).device

    mask = legal_moves_mask(board).to(logits.device)
    if mask.sum().item() == 0:
        return []

    masked_logits = logits.masked_fill(~mask, -1e9)
    probs = torch.softmax(masked_logits, dim=0)

    k_eff = min(k, int(mask.sum().item()))
    topk = torch.topk(masked_logits, k=k_eff)

    entries = []
    for idx in topk.indices.tolist():
        move = index_to_move(idx, board)
        if move is None or move not in board.legal_moves:
            continue

        # Immediate reward (align with training capture bonus)
        immediate = 0.0
        if board.is_capture(move):
            if board.is_en_passant(move):
                captured_sq = chess.square(
                    chess.square_file(move.to_square),
                    chess.square_rank(move.from_square),
                )
            else:
                captured_sq = move.to_square
            captured_piece = board.piece_at(captured_sq)
            if captured_piece is not None:
                immediate += PIECE_VALUES.get(captured_piece.piece_type, 0.0) / 20.0

        board.push(move)
        if board.is_game_over():
            result = board.result()
            mover_is_white = not board.turn  # flipped after push
            if result == "1-0":
                score = immediate + (2.5 if mover_is_white else -1.0)
            elif result == "0-1":
                score = immediate + (2.5 if not mover_is_white else -1.0)
            else:
                score = immediate - 0.5
        else:
            state_next = board_to_tensor(board).unsqueeze(0).to(device)
            _, v_next = net(state_next)
            score = immediate + gamma * (-v_next.item())
        board.pop()

        entries.append((move, probs[idx].item(), masked_logits[idx].item(), score))

    # Sort by value score (desc), tie-break by prob
    entries.sort(key=lambda t: (t[3], t[1]), reverse=True)
    return entries


def _prob_mask_stats(probs: torch.Tensor, mask: torch.Tensor) -> dict:
    """
    Compute summary statistics for a policy distribution under a legal-move mask.
    """
    illegal = probs.masked_select(~mask)
    legal = probs.masked_select(mask)
    illegal_mass = float(illegal.sum().item()) if illegal.numel() else 0.0
    illegal_max = float(illegal.max().item()) if illegal.numel() else 0.0
    legal_mass = float(legal.sum().item()) if legal.numel() else 0.0
    legal_count = int(mask.sum().item())
    return {
        "illegal_mass": illegal_mass,
        "illegal_max": illegal_max,
        "legal_mass": legal_mass,
        "legal_count": legal_count,
        "num_illegal": int(mask.numel() - legal_count),
    }


@torch.no_grad()
def run_mask_audit(net: ActorCriticResNet, board: chess.Board, label: str = "Mask audit"):
    """
    Evaluate the policy on `board` and report probability mass assigned to illegal moves.
    """
    device = next(net.parameters()).device
    state = board_to_tensor(board).unsqueeze(0).to(device)
    logits, _ = net(state)
    mask = legal_moves_mask(board).to(device)
    legal_count = int(mask.sum().item())
    if legal_count == 0:
        print(f"{label}: no legal moves available; skipping audit.")
        return

    masked_logits = logits[0].masked_fill(~mask, -1e9)
    probs = torch.softmax(masked_logits, dim=0)
    stats = _prob_mask_stats(probs, mask)
    print(
        f"{label}: legal moves {stats['legal_count']}, "
        f"legal prob mass {stats['legal_mass']:.6f}, "
        f"illegal prob mass {stats['illegal_mass']:.3e} "
        f"(max illegal {stats['illegal_max']:.3e})"
    )


@torch.no_grad()
def play_and_explain(net: Optional[ActorCriticResNet] = None,
                     model_path: Optional[str] = None,
                     k: int = 5,
                     num_moves: int = 200,
                     sample: bool = False,
                     seed: int = 0,
                     gamma: float = 0.995,
                     selection: str = "policy"):
    """
    Plays a single self-play game (policy controls both sides) and prints per-move diagnostics:
    - current value V(s)
    - top-K policy moves: SAN, prob, masked logit, value lookahead score
    - chosen move (sampled from policy or selected deterministically)

    Args:
        selection: 'policy' picks the policy argmax, 'value' picks the highest lookahead score.

    If model_path is None, initializes a fresh ActorCriticResNet.
    """
    torch.manual_seed(seed)
    if net is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = load_network(model_path, device)
    else:
        device = next(net.parameters()).device
        net.eval()
        if model_path:
            print("Warning: `net` provided; `model_path` argument will be ignored.")

    board = chess.Board()

    ply = 0
    while not board.is_game_over() and ply < num_moves:
        side = "White" if board.turn == chess.WHITE else "Black"
        state = board_to_tensor(board).unsqueeze(0).to(device)
        logits, v = net(state)
        logits = logits[0]
        v = v.item()

        # Mask and probs for info
        mask = legal_moves_mask(board).to(logits.device)
        n_legal = int(mask.sum().item())
        if n_legal == 0:
            print("No legal moves; aborting.")
            break
        masked_logits = logits.masked_fill(~mask, -1e9)
        probs = torch.softmax(masked_logits, dim=0)
        mask_stats = _prob_mask_stats(probs, mask)

        # Collect top-K with values
        topk_entries = topk_policy_with_values(board, net, logits, k=k, gamma=gamma)

        print(f"\nMove {ply+1} ({side} to move)")
        print(board)
        print(f"Value V(s): {v:+.4f}")
        print("Top-{} candidates (SAN | prob | logit | lookahead score):".format(len(topk_entries)))
        for move, p, lg, sc in topk_entries:
            try:
                san = board.san(move)
            except Exception:
                san = move.uci()
            print(f"  {san:8s} | {p:7.4f} | {lg:7.3f} | {sc:+7.3f}")

        # Summaries: sum of shown probs and number of legal moves
        sum_topk = sum(p for _, p, _, _ in topk_entries)
        print(f"Sum prob (top-{len(topk_entries)}): {sum_topk:.4f} | Legal moves: {n_legal}")
        print(
            "Mask audit: legal mass {legal:.6f} over {count} moves, illegal mass {illegal:.3e} (max {illegal_max:.3e})"
            .format(
                legal=mask_stats["legal_mass"],
                count=mask_stats["legal_count"],
                illegal=mask_stats["illegal_mass"],
                illegal_max=mask_stats["illegal_max"],
            )
        )

        policy_argmax_idx = int(torch.argmax(probs).item())
        policy_argmax_move = index_to_move(policy_argmax_idx, board)

        # Choose policy move (sample or deterministic selection)
        move = None
        if sample:
            dist = torch.distributions.Categorical(probs)
            sampled_idx = dist.sample().item()
            move = index_to_move(sampled_idx, board)
        else:
            if selection not in {"policy", "value"}:
                raise ValueError(f"Unsupported selection mode '{selection}'. Use 'policy' or 'value'.")

            if selection == "policy":
                move = policy_argmax_move
            else:  # selection == "value"
                move = topk_entries[0][0] if topk_entries else None

        if move is None or move not in board.legal_moves:
            # fallback hierarchy: top value candidate, policy argmax, then first legal
            move = (topk_entries[0][0] if topk_entries else None) or (
                policy_argmax_move
            )
            if move is None or move not in board.legal_moves:
                move = next(iter(board.legal_moves))

        try:
            chosen_san = board.san(move)
        except Exception:
            chosen_san = move.uci()
        print(f"Chosen: {chosen_san}\n")

        board.push(move)
        ply += 1
        time.sleep(1.5)
        
    print("\nFinal position:")
    print(board)
    print(f"Result: {board.result()}")
    outcome = board.outcome()
    if outcome is not None:
        term = outcome.termination
        term_name = getattr(term, 'name', str(term))
        winner = outcome.winner
        if winner is None:
            winner_str = "None"
        else:
            winner_str = "White" if winner else "Black"
        print(f"Termination: {term_name}")
        print(f"Winner: {winner_str}")


def _stm_plane_sequence(board: chess.Board, plies: int = 2):
    """
    Helper to collect STM plane tensors over a short move sequence.
    """
    snapshots = []
    moves = []
    working_board = board.copy()

    snapshots.append(board_to_tensor(working_board).clone())
    for _ in range(plies):
        try:
            move = next(iter(working_board.legal_moves))
        except StopIteration:
            break
        moves.append(move.uci())
        working_board.push(move)
        snapshots.append(board_to_tensor(working_board).clone())
    return snapshots, moves


def check_stm_plane_consistency():
    """
    Print diagnostics confirming the STM plane is constant under canonicalization.
    """
    snapshots, moves = _stm_plane_sequence(chess.Board())
    print("STM plane audit:")
    deviations = []
    for idx, tensor in enumerate(snapshots):
        unique_vals = torch.unique(tensor[16]).tolist()
        print(f"  Ply {idx}: plane16 unique values {unique_vals}")
        if unique_vals != [1.0]:
            deviations.append((idx, unique_vals))
    if moves:
        print(f"  Moves applied: {moves}")
    if deviations:
        print("  Warning: STM plane deviates from canonical constant 1.0:", deviations)
    else:
        print("  STM plane remains constant at 1.0 (expected with canonical perspective).")


@torch.no_grad()
def perspective_unit_test(net: ActorCriticResNet, tolerance: float = 1e-3):
    """
    Compare policy outputs on a position and its color-flipped counterpart.
    """
    device = next(net.parameters()).device
    fen = "rnbq1rk1/pp1pppbp/2p2np1/2P5/2P1P3/2N1BN2/PP3PPP/R2QKB1R w KQ - 1 9"
    board = chess.Board(fen)
    board_flipped = board.mirror()

    def _policy(board_obj: chess.Board):
        state = board_to_tensor(board_obj).unsqueeze(0).to(device)
        logits, _ = net(state)
        mask = legal_moves_mask(board_obj).to(device)
        masked_logits = logits[0].masked_fill(~mask, -1e9)
        probs = torch.softmax(masked_logits, dim=0)
        return probs.detach().cpu(), mask.detach().cpu()

    probs, mask = _policy(board)
    probs_flip, mask_flip = _policy(board_flipped)

    legal_mismatch = int(torch.sum(mask ^ mask_flip).item())
    shared_mask = mask & mask_flip
    shared_count = int(shared_mask.sum().item())

    if shared_count > 0:
        diffs = torch.abs(probs[shared_mask] - probs_flip[shared_mask])
        max_diff = float(diffs.max().item())
        mean_diff = float(diffs.mean().item())
    else:
        max_diff = float("nan")
        mean_diff = float("nan")

    total_l1 = float(torch.abs(probs - probs_flip).sum().item())
    within_tolerance = shared_count > 0 and max_diff <= tolerance

    print("Perspective unit test:")
    print(f"  FEN: {fen}")
    print(f"  Shared canonical legal moves: {shared_count}")
    print(f"  Mask mismatches: {legal_mismatch}")
    if shared_count > 0:
        print(f"  Max prob diff on shared moves: {max_diff:.3e} (mean {mean_diff:.3e})")
        print(f"  Within tolerance {tolerance}: {'yes' if within_tolerance else 'no'}")
    print(f"  Total L1 diff over all moves: {total_l1:.3e}")


@torch.no_grad()
def value_flip_test(net: ActorCriticResNet):
    """
    Evaluate how the value head responds when only the side-to-move flips.
    """
    device = next(net.parameters()).device
    test_fens = [
        ("Startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("Symmetric middlegame", "r2q1rk1/pp1bbppp/2nppn2/2p5/2P1PP2/2NBBN2/PP3P1P/R2Q1RK1 w - - 0 10"),
    ]
    print("Value flip test:")
    for label, fen in test_fens:
        board = chess.Board(fen)
        board_flipped = board.copy()
        board_flipped.turn = not board.turn

        state = board_to_tensor(board).unsqueeze(0).to(device)
        _, v = net(state)
        state_flip = board_to_tensor(board_flipped).unsqueeze(0).to(device)
        _, v_flip = net(state_flip)

        v_item = float(v.item())
        v_flip_item = float(v_flip.item())
        delta = v_flip_item - v_item
        print(
            f"  {label}: V(stm)={v_item:+.4f}, V(flip)={v_flip_item:+.4f}, delta={delta:+.4f}"
        )


def run_diagnostics_suite(net: ActorCriticResNet):
    """
    Execute the requested diagnostics in a single pass.
    """
    print("\n=== Diagnostics Suite ===")
    check_stm_plane_consistency()
    run_mask_audit(net, chess.Board(), label="Mask audit (start position)")
    perspective_unit_test(net)
    value_flip_test(net)


def main():
    parser = argparse.ArgumentParser(description="Analyze policy decisions with top-K lookahead")
    parser.add_argument("--model", type=str, default=None, help="Path to .pt model file")
    parser.add_argument("--k", type=int, default=5, help="Top-K candidates to display")
    parser.add_argument("--moves", type=int, default=800, help="Max number of plies to play")
    parser.add_argument("--sample", action="store_true", help="Sample from policy instead of argmax")
    parser.add_argument("--select", choices=["policy", "value"], default="policy",
                        help="Deterministic selection rule when not sampling")
    parser.add_argument("--gamma", type=float, default=0.995, help="Discount for lookahead value")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for action sampling")
    parser.add_argument("--diagnostics", action="store_true", help="Run diagnostic checks before self-play")
    parser.add_argument("--diagnostics-only", action="store_true", help="Run diagnostics and skip self-play")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_network(args.model, device)

    if args.diagnostics or args.diagnostics_only:
        run_diagnostics_suite(net)
    if args.diagnostics_only:
        return

    play_and_explain(
        net=net,
        model_path=None,
        k=args.k,
        num_moves=args.moves,
        sample=args.sample,
        seed=args.seed,
        gamma=args.gamma,
        selection=args.select,
    )

if __name__ == "__main__":
    main()
