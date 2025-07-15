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

def piece_plane_index(piece):
    offset = 0 if piece.color == chess.WHITE else 6
    return offset + {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5
    }[piece.piece_type]

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
    tensor = torch.zeros((13, 8, 8), dtype=torch.float32)

    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        row = 7 - chess.square_rank(square)
        col = chess.square_file(square)
        plane = piece_plane_index(piece)
        tensor[plane, row, col] = 1.0

    tensor[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
    #print(tensor.shape)
    return tensor

def legal_moves_mask(board):
    mask = torch.zeros(4672)
    for move in board.legal_moves:
        idx = move_to_index(move)
        if idx is not None:
            mask[idx] = 1
    #assert mask.sum().item() == len(legal_moves), "Mismatch between mask and legal moves"
    return mask

if __name__ == "__main__":
    idx = move_to_index(chess.Move.from_uci("B1a3"))
    print(idx)
    print(index_to_move(idx))  # should give back Move(g2, h4)