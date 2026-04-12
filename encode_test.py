import chess
import random
import torch
from helper import move_to_index, index_to_move, board_to_tensor, BOARD_TENSOR_PLANES


# --------------------------------------------------------
# 1) Move encoding/decoding reversibility test
# --------------------------------------------------------
def test_move_encoding_on_board(board):
    errors = []
    for move in list(board.legal_moves):
        try:
            idx = move_to_index(move, board)
            decoded = index_to_move(idx, board)
        except Exception as e:
            errors.append((move, None, f"Exception during encode/decode: {e}"))
            continue

        if decoded not in board.legal_moves:
            errors.append((move, decoded, "Decoded move is not legal in this position"))
        elif move.uci() != decoded.uci():
            errors.append((move, decoded, "Decoded move differs in UCI string"))

    return errors


# --------------------------------------------------------
# 2) Board-to-tensor symmetry test
# --------------------------------------------------------
def test_board_tensor_symmetry(board):
    """
    Checks that flipping the board and switching the turn
    gives the same tensor as the opponent would see
    in the mirrored position.
    """
    tensor_current = board_to_tensor(board)

    flipped_board = board.mirror()
    flipped_board.turn = not board.turn
    tensor_flipped = board_to_tensor(flipped_board)

    return torch.allclose(tensor_current, tensor_flipped, atol=1e-3)


def test_tensor_shape():
    tensor = board_to_tensor(chess.Board())
    return tensor.shape == (BOARD_TENSOR_PLANES, 8, 8)


def test_en_passant_plane():
    board = chess.Board()
    for san in ["e4", "Nf6", "e5", "d5"]:
        board.push_san(san)

    tensor = board_to_tensor(board)
    expected = torch.zeros((8, 8), dtype=torch.float32)
    expected[chess.square_rank(chess.D6), chess.square_file(chess.D6)] = 1.0

    return torch.equal(tensor[18], expected)


def test_repetition_plane():
    board = chess.Board()
    for san in ["Nf3", "Nf6", "Ng1", "Ng8"]:
        board.push_san(san)

    tensor = board_to_tensor(board)
    return bool(torch.all(tensor[19] == 1.0))


# --------------------------------------------------------
# Test Runner
# --------------------------------------------------------
def run_tests(num_random_positions=100):
    all_move_errors = []
    symmetry_failures = 0
    feature_failures = []

    print("Testing starting position...")
    board = chess.Board()
    errs = test_move_encoding_on_board(board)
    if errs:
        print(f"  Found {len(errs)} move encoding errors in starting position")
        all_move_errors.extend(errs)
    if not test_board_tensor_symmetry(board):
        print("  Symmetry test FAILED in starting position")
        symmetry_failures += 1
    if not test_tensor_shape():
        print("  Tensor shape test FAILED")
        feature_failures.append("tensor shape")
    if not test_en_passant_plane():
        print("  En passant plane test FAILED")
        feature_failures.append("en passant plane")
    if not test_repetition_plane():
        print("  Repetition plane test FAILED")
        feature_failures.append("repetition plane")

    print(f"\nTesting {num_random_positions} random positions...")
    for i in range(num_random_positions):
        board = chess.Board()
        for _ in range(random.randint(5, 30)):
            if board.is_game_over():
                break
            board.push(random.choice(list(board.legal_moves)))

        errs = test_move_encoding_on_board(board)
        if errs:
            print(f"  Move encoding errors in random board {i+1}")
            all_move_errors.extend(errs)

        if not test_board_tensor_symmetry(board):
            print(f"  Symmetry test FAILED in random board {i+1}")
            symmetry_failures += 1

    if not all_move_errors and symmetry_failures == 0 and not feature_failures:
        print("\nAll tests passed.")
    else:
        print(
            f"\nFound {len(all_move_errors)} move encoding errors, "
            f"{symmetry_failures} symmetry failures, and {len(feature_failures)} feature failures."
        )
        for orig, decoded, reason in all_move_errors[:20]:
            print(f"  Move: {orig} -> {decoded} | {reason}")
        for feature in feature_failures:
            print(f"  Feature failure: {feature}")


if __name__ == "__main__":
    run_tests()
