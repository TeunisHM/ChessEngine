
import unittest
import chess
import torch
from helper import (
    BOARD_TENSOR_PLANES,
    board_to_tensor,
    index_to_move,
    legal_moves_mask,
    move_to_index,
)

class TestHelperFunctions(unittest.TestCase):

    def test_board_to_tensor_initial_pos(self):
        """Tests the tensor representation from the starting position."""
        board = chess.Board()
        tensor = board_to_tensor(board)

        # --- Shape and Type ---
        self.assertEqual(tensor.shape, (BOARD_TENSOR_PLANES, 8, 8))
        self.assertEqual(tensor.dtype, torch.float32)

        # --- Player's Turn (White) ---
        # White's pieces on planes 0-5
        self.assertTrue(torch.all(tensor[0, 1, :] == 1.0))  # White Pawns on rank 2 (index 1)
        self.assertEqual(tensor[3, 0, 0], 1.0)  # White Rook on a1
        self.assertEqual(tensor[3, 0, 7], 1.0)  # White Rook on h1
        
        # Opponent's pieces on planes 6-11
        self.assertTrue(torch.all(tensor[6, 6, :] == 1.0)) # Black Pawns on rank 7 (index 6)
        self.assertEqual(tensor[9, 7, 0], 1.0)  # Black Rook on a8 (plane 6+3)
        self.assertEqual(tensor[9, 7, 7], 1.0)  # Black Rook on h8 (plane 6+3)

        # --- Metadata ---
        self.assertTrue(torch.all(tensor[12, :, :] == 1.0)) # White K-side castling
        self.assertTrue(torch.all(tensor[13, :, :] == 1.0)) # White Q-side castling
        self.assertTrue(torch.all(tensor[14, :, :] == 1.0)) # Black K-side castling
        self.assertTrue(torch.all(tensor[15, :, :] == 1.0)) # Black Q-side castling
        self.assertTrue(torch.all(tensor[16, :, :] == 1.0)) # Turn is White
        self.assertAlmostEqual(tensor[17, 0, 0], 1 / 100.0) # Move counter

    def test_board_to_tensor_black_perspective(self):
        """Tests the tensor representation from Black's perspective."""
        board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
        tensor = board_to_tensor(board)

        # --- Player's Turn (Black) ---
        # Black's pieces (now on planes 0-5) should be on their "home" ranks from their view
        self.assertTrue(torch.all(tensor[0, 1, :] == 1.0))  # Black Pawns on what looks like rank 2 (flipped from rank 7)
        self.assertEqual(tensor[3, 0, 0], 1.0)  # Black Rook on a8 (flipped to rank 1)
        
        # Opponent's pieces (White, now on planes 6-11)
        # White pawns on what looks like rank 7 (flipped from rank 2), except for the e-pawn
        expected_white_pawn_rank = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0])
        self.assertTrue(torch.all(tensor[6, 6, :] == expected_white_pawn_rank))
        # White Pawn on e4 (rank 3) is now on rank 4 (7-3) from black's perspective
        self.assertEqual(tensor[6, 4, 4], 1.0)

        # --- Metadata ---
        self.assertTrue(torch.all(tensor[16, :, :] == 1.0)) # Canonical STM plane stays 1.0

    def test_move_encoding_decoding_roundtrip(self):
        """
        Tests that for any legal move in a variety of positions,
        move_to_index -> index_to_move gives back the original move.
        """
        fen_strings = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", # Start
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", # Complex
            "8/P7/8/k7/8/8/8/K7 w - - 0 1", # White promotion
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", # White promotion capture
            "k7/8/8/8/8/8/p7/K7 b - - 0 1", # Black promotion
            "4k3/8/8/8/8/8/4p3/4K3 b - - 0 1", # Black pawn push
            "r1bqkbnr/pp1ppppp/2n5/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", # Castling available
            "r3k2r/ppp2ppp/2n5/2b5/2B5/2P2N2/P1P2PPP/R2QK2R b KQkq - 0 12" # Black's turn
        ]

        for fen in fen_strings:
            board = chess.Board(fen)
            for move in board.legal_moves:
                with self.subTest(fen=fen, move=move.uci()):
                    try:
                        idx = move_to_index(move, board)
                        self.assertIsNotNone(idx, f"move_to_index returned None for {move.uci()}")
                        
                        decoded_move = index_to_move(idx, board)
                        self.assertIsNotNone(decoded_move, f"index_to_move returned None for index {idx} from move {move.uci()}")
                        
                        # We must check if the decoded move is legal, as the decoder can
                        # produce pseudo-legal moves that are not actually legal.
                        self.assertTrue(decoded_move in board.legal_moves, f"Decoded move {decoded_move.uci()} is not legal. Original: {move.uci()}")

                        self.assertEqual(move, decoded_move)

                    except Exception as e:
                        self.fail(f"Failed on FEN: '{fen}' with move: {move.uci()}. Error: {e}")

    def test_legal_moves_mask(self):
        """Tests that the legal move mask is correct."""
        board = chess.Board("r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
        
        mask = legal_moves_mask(board)
        num_legal_moves = board.legal_moves.count()

        # Check that the number of enabled bits in the mask equals the number of legal moves
        self.assertEqual(torch.sum(mask).item(), num_legal_moves)

        # Check that every legal move corresponds to a True value in the mask
        for move in board.legal_moves:
            idx = move_to_index(move, board)
            self.assertTrue(mask[idx], f"Mask is False for legal move {move.uci()}")

        # Check a known illegal move
        illegal_move = chess.Move.from_uci("a1a2") # Blocked
        illegal_idx = move_to_index(illegal_move, board)
        self.assertFalse(mask[illegal_idx], f"Mask is True for illegal move {illegal_move.uci()}")

    def test_move_indices_match_mirror(self):
        """Canonical move indices should be identical after mirroring the board."""
        fen_strings = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R b KQ - 1 8",
            "8/P7/8/k7/8/8/8/K7 w - - 0 1",
            "k7/8/8/8/8/8/p7/K7 b - - 0 1",
            "4k3/8/8/8/8/8/4p3/4K3 b - - 0 1",
            "r1bqkbnr/pp1ppppp/2n5/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
            "r3k2r/ppp2ppp/2n5/2b5/2B5/2P2N2/P1P2PPP/R2QK2R b KQkq - 0 12",
        ]

        for fen in fen_strings:
            board = chess.Board(fen)
            mirror_board = board.mirror()
            with self.subTest(fen=fen):
                for move in list(board.legal_moves):
                    idx = move_to_index(move, board)
                    mirrored_move = chess.Move(
                        chess.square_mirror(move.from_square),
                        chess.square_mirror(move.to_square),
                        promotion=move.promotion,
                    )
                    self.assertTrue(
                        mirror_board.is_legal(mirrored_move),
                        f"Mirrored move {mirrored_move.uci()} not legal in mirrored board for original move {move.uci()}",
                    )
                    idx_mirror = move_to_index(mirrored_move, mirror_board)
                    self.assertEqual(
                        idx,
                        idx_mirror,
                        f"Canonical index mismatch for move {move.uci()} (mirror {mirrored_move.uci()}) in FEN '{fen}'",
                    )

if __name__ == '__main__':
    unittest.main()
