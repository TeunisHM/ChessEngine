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

def piece_plane_index(piece, perspective):
    """
    Calculates the plane index for a piece from a specific perspective.
    Perspective should be the color of the player whose turn it is.
    """
    # Planes 0-5 are for the current player's pieces
    if piece.color == perspective:
        return piece.piece_type - 1  # PAWN=1 -> 0, KNIGHT=2 -> 1, etc.
    # Planes 6-11 are for the opponent's pieces
    else:
        return piece.piece_type - 1 + 6

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

def move_to_index(move: chess.Move) -> int:
    """Converts a move to its index in the 4672-action space."""
    plane = get_move_plane(move)
    return move.from_square * 73 + plane

def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """
    Converts an index to a chess.Move object.
    CRITICAL: This function now requires the board state to handle promotions correctly.
    """
    from_square = index // 73
    plane = index % 73
    
    from_rank = chess.square_rank(from_square)

    # Underpromotions
    if plane >= 64:
        promo_map = {0: chess.KNIGHT, 1: chess.BISHOP, 2: chess.ROOK}
        if 64 <= plane < 67: # Capture left
            promo_piece = promo_map[plane - 64]
            to_file = chess.square_file(from_square) - 1
        elif 67 <= plane < 70: # Push forward
            promo_piece = promo_map[plane - 67]
            to_file = chess.square_file(from_square)
        else: # Capture right
            promo_piece = promo_map[plane - 70]
            to_file = chess.square_file(from_square) + 1
        
        to_rank = 7 if board.turn == chess.WHITE else 0
        return chess.Move(from_square, chess.square(to_file, to_rank), promotion=promo_piece)

    # Knight moves
    elif 56 <= plane < 64:
        knight_map = {
            0: (2, 1), 1: (1, 2), 2: (-1, 2), 3: (-2, 1),
            4: (-2, -1), 5: (-1, -2), 6: (1, -2), 7: (2, -1)
        }
        dr, df = knight_map[plane - 56]
        to_rank = from_rank + dr
        to_file = chess.square_file(from_square) + df
        return chess.Move(from_square, chess.square(to_file, to_rank))

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
        to_file = chess.square_file(from_square) + df * distance
        
        # Check for queen promotion
        is_pawn = board.piece_type_at(from_square) == chess.PAWN
        is_promo_rank = (from_rank == 6 and board.turn == chess.WHITE) or \
                        (from_rank == 1 and board.turn == chess.BLACK)
                        
        if is_pawn and is_promo_rank:
            return chess.Move(from_square, chess.square(to_file, to_rank), promotion=chess.QUEEN)
        else:
            return chess.Move(from_square, chess.square(to_file, to_rank)) 

def board_to_tensor(board):
    """
    Converts the board state to a canonical tensor representation (18, 8, 8).
    The board is always viewed from the perspective of the current player.
    """
    # 18 planes: 6 for player pieces, 6 for opponent, 4 for castling, 1 for turn, 1 for move count
    tensor = torch.zeros((18, 8, 8), dtype=torch.float32)
    
    current_player = board.turn
    
    # --- Planes 0-11: Piece Positions (from current player's perspective) ---
    for square, piece in board.piece_map().items():
        rank, file = chess.square_rank(square), chess.square_file(square)
        
        # Flip the board if the current player is Black
        if current_player == chess.BLACK:
            rank = 7 - rank
        
        plane = piece_plane_index(piece, current_player)
        tensor[plane, rank, file] = 1.0

    # --- Planes 12-15: Castling Rights ---
    # These are also from the perspective of the current player
    if board.has_castling_rights(current_player):
        if board.has_kingside_castling_rights(current_player):
            tensor[12, :, :] = 1.0 # Player can castle kingside
        if board.has_queenside_castling_rights(current_player):
            tensor[13, :, :] = 1.0 # Player can castle queenside
            
    opponent = not current_player
    if board.has_castling_rights(opponent):
        if board.has_kingside_castling_rights(opponent):
            tensor[14, :, :] = 1.0 # Opponent can castle kingside
        if board.has_queenside_castling_rights(opponent):
            tensor[15, :, :] = 1.0 # Opponent can castle queenside

    # --- Plane 16: Turn Indicator ---
    # This is now less critical since the board is canonical, but can still be useful.
    # It's always 1.0 to indicate "it is my turn to move".
    tensor[16, :, :] = 1.0

    # --- Plane 17: Total Move Count ---
    # Helps the network learn about game progression. Can be normalized.
    # A simple way is to scale it to be between 0 and 1.
    move_count_scaled = min(board.fullmove_number / 100.0, 1.0)
    tensor[17, :, :] = move_count_scaled
    
    return tensor

def legal_moves_mask(board):
    mask = torch.zeros(4672)
    for move in board.legal_moves:
        idx = move_to_index(move)
        if idx is not None:
            mask[idx] = 1
    #assert mask.sum().item() == len(legal_moves), "Mismatch between mask and legal moves"
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
    idx = move_to_index(chess.Move.from_uci("d8c7"))
    print(idx)
    print(index_to_move(idx))  