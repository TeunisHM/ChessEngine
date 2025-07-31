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

def move_to_index(move: chess.Move) -> int:
    from_sq = move.from_square
    to_sq = move.to_square
    fx, fy = square_to_coords(from_sq)
    tx, ty = square_to_coords(to_sq)
    dx = tx - fx
    dy = ty - fy

    # 1. Handle promotions
    if move.promotion:
        # piece = board.piece_at(from_sq)
        # if piece is None or piece.piece_type != chess.PAWN:
        #     return None

        # Pawn promotions must be one rank forward
        if dy not in [1, -1]:
            return None
        color = 1 if dy > 0 else -1
        offset_x = dx

        promo_piece = chess.piece_symbol(move.promotion).lower()
        if promo_piece not in PROMOTION_PIECES:
            return None
        promo_idx = PROMOTION_PIECES.index(promo_piece)

        if offset_x == 0:     # Forward
            direction = 0
        elif offset_x == -1:  # Left capture
            direction = 1
        elif offset_x == 1:   # Right capture
            direction = 2
        else:
            return None

        move_type = 56 + direction * 3 + promo_idx  # 56–64
        return from_sq * 73 + move_type

    # 2. Handle sliding (queen-like) moves
    for dir_idx, (dx_step, dy_step) in enumerate(DIRECTIONS):
        for dist in range(1, 8):
            if fx + dx_step * dist == tx and fy + dy_step * dist == ty:
                move_type = dir_idx * 7 + (dist - 1)  # 0–55
                return from_sq * 73 + move_type

    # 3. Handle knight moves
    for i, (kx, ky) in enumerate(KNIGHT_DIRS):
        if fx + kx == tx and fy + ky == ty:
            move_type = 65 + i  # 65–72
            return from_sq * 73 + move_type
    return None  # unsupported move

def index_to_move(index: int) -> chess.Move:
    if index < 0 or index >= 4672:
        return None
    
    from_sq = index // 73
    move_type = index % 73
    fx, fy = square_to_coords(from_sq)

    # 0–55: queen-like sliding moves
    if move_type < 56:
        dir_idx = move_type // 7
        dist = (move_type % 7) + 1
        dx, dy = DIRECTIONS[dir_idx]
        tx = fx + dx * dist
        ty = fy + dy * dist
        if 0 <= tx < 8 and 0 <= ty < 8:
            to_sq = chess.square(tx, ty)
            return chess.Move(from_sq, to_sq)

    # 56–64: promotions (3 directions × 3 pieces)
    elif 56 <= move_type < 65:
        promo_offset = move_type - 56
        direction = promo_offset // 3
        piece_idx = promo_offset % 3
        dx = [0, -1, 1][direction]  # forward, left, right
        dy = 1 if fy == 6 else -1   # guess color by rank (white promotes from rank 6)
        tx = fx + dx
        ty = fy + dy
        if 0 <= tx < 8 and 0 <= ty < 8:
            to_sq = chess.square(tx, ty)
            promo_piece = PROMOTION_PIECES[piece_idx]
            return chess.Move(from_sq, to_sq, promotion=chess.PIECE_SYMBOLS.index(promo_piece))

    # 65–72: knight moves
    elif 65 <= move_type < 73:
        knight_idx = move_type - 65
        dx, dy = KNIGHT_DIRS[knight_idx]
        tx = fx + dx
        ty = fy + dy
        if 0 <= tx < 8 and 0 <= ty < 8:
            to_sq = chess.square(tx, ty)
            return chess.Move(from_sq, to_sq)

    return None  # unsupported or out of bounds
    
def board_to_tensor_flat(board):
    piece_map = board.piece_map()
    board_tensor = torch.zeros(64 * 12)
    for square, piece in piece_map.items():
        offset = "PNBRQKpnbrqk".index(piece.symbol())
        board_tensor[64 * offset + square] = 1
    turn_tensor = torch.tensor([board.turn], dtype=torch.float32)
    return torch.cat([board_tensor, turn_tensor])

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