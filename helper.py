import chess
import torch
import random

PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0  # not used since game ends before king capture
}

OPENINGS = {
    # Open Games (1.e4 e5)
    "italian": ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5"],
    "scotch": ["e4", "e5", "Nf3", "Nc6", "d4", "exd4", "Nxd4"],
    "sicilian": ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3"],
    "french": ["e4", "e6", "d4", "d5", "Nc3"],
    "caro": ["e4", "c6", "d4", "d5", "Nc3", "dxe4", "Nxe4"],
    "ruy_lopez": ["e4", "e5", "Nf3", "Nc6", "Bb5"],
    "vienna": ["e4", "e5", "Nc3"],
    "kings_gambit": ["e4", "e5", "f4"],
    "philidor": ["e4", "e5", "Nf3", "d6"],
    "petrov": ["e4", "e5", "Nf3", "Nf6"],
    "four_knights": ["e4", "e5", "Nf3", "Nc6", "Nc3", "Nf6"],
    "three_knights": ["e4", "e5", "Nf3", "Nc6", "Nc3"],
    "pirc": ["e4", "d6", "d4", "Nf6", "Nc3", "g6"],
    "modern": ["e4", "g6", "d4", "Bg7"],
    "alehine": ["e4", "Nf6"],

    # Semi-Open (1.e4 but Black replies asymmetrically)
    "sicilian_dragon": ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "g6"],
    "sicilian_najdorf": ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "a6"],
    "sicilian_scheveningen": ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "e6"],
    "sicilian_accelerated_dragon": ["e4", "c5", "Nf3", "Nc6", "d4", "cxd4", "Nxd4", "g6"],
    "sicilian_alapin": ["e4", "c5", "c3"],
    "sicilian_closed": ["e4", "c5", "Nc3"],
    "caro_advance": ["e4", "c6", "d4", "d5", "e5"],
    "french_advance": ["e4", "e6", "d4", "d5", "e5"],
    "french_exchange": ["e4", "e6", "d4", "d5", "exd5", "exd5"],
    "french_tarrasch": ["e4", "e6", "d4", "d5", "Nd2"],
    "scandinavian": ["e4", "d5"],
    "nimzowitsch_defense": ["e4", "Nc6"],

    # Closed/Openings starting with 1.d4
    "queens_gambit": ["d4", "d5", "c4"],
    "queens_gambit_declined": ["d4", "d5", "c4", "e6"],
    "queens_gambit_accepted": ["d4", "d5", "c4", "dxc4"],
    "london": ["d4", "d5", "Nf3", "Nf6", "Bf4"],
    "colle": ["d4", "d5", "Nf3", "Nf6", "e3"],
    "trompowsky": ["d4", "Nf6", "Bg5"],
    "veresov": ["d4", "Nf6", "Nc3", "d5", "Bg5"],
    "catalan": ["d4", "Nf6", "c4", "e6", "g3"],
    "benoni": ["d4", "Nf6", "c4", "c5", "d5", "e6"],
    "benko": ["d4", "Nf6", "c4", "c5", "d5", "b5"],
    "grunfeld": ["d4", "Nf6", "c4", "g6", "Nc3", "d5"],
    "kings_indian": ["d4", "Nf6", "c4", "g6", "Nc3", "Bg7", "e4", "d6"],
    "nimzo_indian": ["d4", "Nf6", "c4", "e6", "Nc3", "Bb4"],
    "queens_indian": ["d4", "Nf6", "c4", "e6", "Nf3", "b6"],
    "bogo_indian": ["d4", "Nf6", "c4", "e6", "Nf3", "Bb4+"],
    "slav": ["d4", "d5", "c4", "c6"],
    "semi_slav": ["d4", "d5", "c4", "c6", "Nc3", "Nf6", "Nf3", "e6"],
    "tarrasch_defense": ["d4", "d5", "c4", "e6", "Nc3", "c5"],
    "dutch": ["d4", "f5"],

    # Flank openings
    "english": ["c4", "e5"],
    "english_symmetric": ["c4", "c5"],
    "english_four_knights": ["c4", "Nf6", "Nc3", "Nc6", "Nf3", "e5"],
    "reti": ["Nf3", "d5", "c4"],
    "kings_fianchetto": ["Nf3", "d5", "g3", "Nf6", "Bg2", "g6"],
    "bird": ["f4", "d5", "Nf3"],
    "larsen_attack": ["b3", "e5"],
    "benoni_fianchetto": ["d4", "Nf6", "c4", "c5", "d5", "g6"],
    "old_indian": ["d4", "Nf6", "c4", "d6", "Nc3", "e5"],

    # Weird/trap/fun
    "grobs": ["g4", "d5"],
    "vang_t_": ["d4", "e5"],
    "polish": ["b4", "d5"],
    "kings_indian_attack": ["Nf3", "Nf6", "d3", "d5", "g3", "g6", "Bg2", "Bg7", "O-O", "O-O", "Nbd2", "Nbd7", "e4", "e5"],
    "albin_countergambit": ["d4", "d5", "c4", "e5"],
    "budapest_gambit": ["d4", "Nf6", "c4", "e5"],
    "englund_gambit": ["d4", "e5"],
    "latvian_gambit": ["e4", "e5", "Nf3", "f5"],
    "elephant_gambit": ["e4", "e5", "Nf3", "d5"],
    "center_game": ["e4", "e5", "d4", "exd4", "Qxd4"],
    "vienna_gambit": ["e4", "e5", "Nc3", "Nf6", "f4"],
}

DIRECTIONS = [
    (0, 1),    # up
    (1, 1),    # up-right
    (1, 0),    # right
    (1, -1),   # down-right
    (0, -1),   # down
    (-1, -1),  # down-left
    (-1, 0),   # left
    (-1, 1)    # up-left
]

KNIGHT_DIRS = [
    (1, 2), (2, 1), (2, -1), (1, -2),
    (-1, -2), (-2, -1), (-2, 1), (-1, 2)
]

PROMOTION_PIECES = ['n', 'r', 'q']  # Promotion piece order (standard)

def square_to_coords(square):
    return chess.square_file(square), chess.square_rank(square)

def get_move_plane(move: chess.Move):
    """
    Calculates the action plane for a given move in the AlphaZero-style representation.
    """
    # Underpromotions (to Rook, Bishop, or Knight)
    if move.promotion and move.promotion in [chess.KNIGHT, chess.BISHOP, chess.ROOK]:
        promo_piece = move.promotion
        from_file = chess.square_file(move.from_square)
        to_file = chess.square_file(move.to_square)
        
        direction = to_file - from_file  # -1 for left, 0 for fwd, 1 for right
        
        if direction == -1: # Capture left
            plane_base = 64
        elif direction == 0: # Push forward
            plane_base = 67
        elif direction == 1: # Capture right
            plane_base = 70
        else:
            raise ValueError("Invalid promotion direction")
            
        # N, B, R
        promo_offset = [chess.KNIGHT, chess.BISHOP, chess.ROOK].index(promo_piece)
        return plane_base + promo_offset

    # Knight moves
    from_rank, from_file = chess.square_rank(move.from_square), chess.square_file(move.from_square)
    to_rank, to_file = chess.square_rank(move.to_square), chess.square_file(move.to_square)
    
    d_rank, d_file = abs(to_rank - from_rank), abs(to_file - from_file)
    if (d_rank == 2 and d_file == 1) or (d_rank == 1 and d_file == 2):
        # It's a knight move
        # We can define a fixed mapping for the 8 knight moves
        delta = (to_rank - from_rank, to_file - from_file)
        knight_moves_map = {
            (2, 1): 0, (1, 2): 1, (-1, 2): 2, (-2, 1): 3,
            (-2, -1): 4, (-1, -2): 5, (1, -2): 6, (2, -1): 7
        }
        return 56 + knight_moves_map[delta]

    # Sliding moves (including queen promotions)
    # This covers Rooks, Bishops, and Queens
    dr, df = to_rank - from_rank, to_file - from_file
    dist = max(abs(dr), abs(df))
    
    # Normalize direction to a single step
    step_dr, step_df = dr // dist, df // dist
    
    direction_map = {
        (1, 0): 0, (1, 1): 1, (0, 1): 2, (-1, 1): 3,
        (-1, 0): 4, (-1, -1): 5, (0, -1): 6, (1, -1): 7
    }
    direction_idx = direction_map[(step_dr, step_df)]
    
    return (direction_idx * 7) + (dist - 1)

def move_to_index(move: chess.Move, board: chess.Board) -> int:
    """
    Converts an absolute move from the board into its canonical index.
    This function now REQUIRES the board to know the perspective.
    """
    player = board.turn
    
    # --- Canonicalize the move ---
    # If the player is Black, we "flip" the move to see it from White's perspective.
    from_square = move.from_square
    to_square = move.to_square
    
    if player == chess.BLACK:
        from_square = chess.square_mirror(from_square)
        to_square = chess.square_mirror(to_square)
        
    # Now, from_square and to_square are the move's squares as if White were making an analogous move.
    
    # Create a temporary move object for this canonical move
    canonical_move = chess.Move(from_square, to_square, promotion=move.promotion)
    
    # Use the same plane logic as before, which works on this canonical move
    plane = get_move_plane(canonical_move)
    
    # The final index is based on the canonical from_square and the plane
    return from_square * 73 + plane

def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """
    Converts a canonical index back into an absolute, legal move for the board.
    """
    player = board.turn
    
    # --- Deconstruct the canonical action ---
    canonical_from_square = index // 73
    plane = index % 73
    
    # Decode the move as if White were making it
    # (This part is complex, let's use a helper for clarity)
    canonical_move = decode_plane_to_move(canonical_from_square, plane, board)
    
    if canonical_move is None:
        return None

    # --- Translate back to the absolute board ---
    # If the player is Black, we must "un-flip" the canonical move
    if player == chess.BLACK:
        from_square = chess.square_mirror(canonical_move.from_square)
        to_square = chess.square_mirror(canonical_move.to_square)
        return chess.Move(from_square, to_square, promotion=canonical_move.promotion)
    else:
        # If White, the canonical move is the real move
        return canonical_move


def decode_plane_to_move(from_square: int, plane: int, board: chess.Board) -> chess.Move:
    """
    Helper function to decode a canonical from_square and plane into a move.
    This function always thinks it's creating a move for White.
    """
    from_rank = chess.square_rank(from_square)
    from_file = chess.square_file(from_square)

    # Underpromotions
    if plane >= 64:
        promo_map = {0: chess.KNIGHT, 1: chess.BISHOP, 2: chess.ROOK}
        if 64 <= plane < 67: to_file_offset, piece_idx = -1, plane - 64
        elif 67 <= plane < 70: to_file_offset, piece_idx = 0, plane - 67
        else: to_file_offset, piece_idx = 1, plane - 70
        
        to_file = from_file + to_file_offset
        promo_piece = promo_map[piece_idx]
        return chess.Move(from_square, chess.square(to_file, 7), promotion=promo_piece)

    # Knight moves
    elif 56 <= plane < 64:
        knight_map = {
            0: (2, 1), 1: (1, 2), 2: (-1, 2), 3: (-2, 1),
            4: (-2, -1), 5: (-1, -2), 6: (1, -2), 7: (2, -1)
        }
        dr, df = knight_map[plane - 56]
        return chess.Move(from_square, chess.square(from_file + df, from_rank + dr))

    # Sliding moves (and Queen promotions)
    else:
        direction_map = {
            0: (1, 0), 1: (1, 1), 2: (0, 1), 3: (-1, 1),
            4: (-1, 0), 5: (-1, -1), 6: (0, -1), 7: (1, -1)
        }
        direction_idx = plane // 7
        distance = (plane % 7) + 1
        dr, df = direction_map[direction_idx]
        
        to_rank = from_rank + dr * distance
        to_file = from_file + df * distance
        to_square = chess.square(to_file, to_rank)

        # We need to check the original board to see if it's a pawn
        # This requires translating the from_square back if it's black's turn
        original_from_square = from_square if board.turn == chess.WHITE else chess.square_mirror(from_square)
        
        if board.piece_type_at(original_from_square) == chess.PAWN and from_rank == 6:
            return chess.Move(from_square, to_square, promotion=chess.QUEEN)
        else:
            return chess.Move(from_square, to_square)

def piece_planes(board: chess.Board) -> torch.Tensor:
    # Your existing 12x8x8 encoding here
    tensor = torch.zeros((12, 8, 8), dtype=torch.float32)
    player = board.turn

    # Planes 0-11: Piece Positions (own pieces 0-5, opp pieces 6-11) ---
    # Rotate 180 deg for Black to make the net always see "player" at bottom
    for square, piece in board.piece_map().items():
        if player == chess.BLACK:
            square = chess.square_mirror(square)

        rank = chess.square_rank(square)
        file = chess.square_file(square)

        if piece.color == player:
            plane = piece.piece_type - 1
        else:
            plane = piece.piece_type - 1 + 6

        tensor[plane, rank, file] = 1.0
    return tensor  

def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """
    Converts the board state to a canonical tensor representation (18, 8, 8).
    The board is always viewed from the perspective of the current player.
    """
    tensor = torch.zeros((18, 8, 8), dtype=torch.float32)
    player = board.turn

    # --- Planes 0-11: Piece Positions (own pieces 0-5, opp pieces 6-11) ---
    # Rotate 180 deg for Black to make the net always see "player" at bottom
    for square, piece in board.piece_map().items():
        if player == chess.BLACK:
            square = chess.square_mirror(square)

        rank = chess.square_rank(square)
        file = chess.square_file(square)

        if piece.color == player:
            plane = piece.piece_type - 1
        else:
            plane = piece.piece_type - 1 + 6

        tensor[plane, rank, file] = 1.0

    # --- Planes 12-15: Castling Rights RELATIVE TO CURRENT PLAYER ---
    # planes 12 = our KS, 13 = our QS, 14 = opp KS, 15 = opp QS
    if player == chess.WHITE:
        our_ks = board.has_kingside_castling_rights(chess.WHITE)
        our_qs = board.has_queenside_castling_rights(chess.WHITE)
        opp_ks = board.has_kingside_castling_rights(chess.BLACK)
        opp_qs = board.has_queenside_castling_rights(chess.BLACK)
    else:
        our_ks = board.has_kingside_castling_rights(chess.BLACK)
        our_qs = board.has_queenside_castling_rights(chess.BLACK)
        opp_ks = board.has_kingside_castling_rights(chess.WHITE)
        opp_qs = board.has_queenside_castling_rights(chess.WHITE)

    if our_ks: tensor[12, :, :] = 1.0
    if our_qs: tensor[13, :, :] = 1.0
    if opp_ks: tensor[14, :, :] = 1.0
    if opp_qs: tensor[15, :, :] = 1.0

    # --- Plane 16: PLAYER-TO-MOVE indicator (canonical perspective -> always 1.0) ---
    tensor[16, :, :] = 1.0

    # --- Plane 17: Total Move Count (scaled) ---
    move_count_scaled = min(board.fullmove_number / 100.0, 1.0)
    tensor[17, :, :] = move_count_scaled

    return tensor

def legal_moves_mask(board: chess.Board) -> torch.Tensor:
    mask = torch.zeros(4672, dtype=torch.bool)
    for move in board.legal_moves:
        try:
            # Pass the board to correctly canonicalize the move
            idx = move_to_index(move, board)
            if idx is not None:
                mask[idx] = 1
        except (ValueError, KeyError) as e:
            print(f"Could not encode legal move: {move.uci()} for board {board.fen()}. Error: {e}")
    return mask

def eval_material(board):
    score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = PIECE_VALUES[piece.piece_type]
            score += value if piece.color == chess.WHITE else -value
    return score

if __name__ == "__main__":
    import chess

    def is_playable_from_start(san_line):
        b = chess.Board()
        for san in san_line:
            try:
                b.push_san(san)
            except Exception:
                return False
        return True

    CLEAN_OPENINGS = {k:v for k,v in OPENINGS.items() if is_playable_from_start(v)}
    BAD = {k:v for k,v in OPENINGS.items() if k not in CLEAN_OPENINGS}
    if BAD:
        print("Skipping non-alternating/illegal opening lines:")
        for k,v in BAD.items():
            print(f"  {k}: {v}")

