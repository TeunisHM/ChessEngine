import chess
import torch

# Mapping directions for queen/pawn-like movement
DIRECTIONS = [
    (1, 0),    # right
    (-1, 0),   # left
    (0, 1),    # up
    (0, -1),   # down
    (1, 1),    # up-right
    (1, -1),   # down-right
    (-1, 1),   # up-left
    (-1, -1),  # down-left
]

PROMOTION_PIECES = ['n', 'r', 'b', 'q']  # can tweak which ones you include

def square_to_coords(sq):
    return sq % 8, sq // 8

def coords_to_square(x, y):
    if 0 <= x < 8 and 0 <= y < 8:
        return y * 8 + x
    return None

def move_to_index(move: chess.Move) -> int:
    from_sq = move.from_square
    to_sq = move.to_square
    fx, fy = square_to_coords(from_sq)
    tx, ty = square_to_coords(to_sq)

    dx = tx - fx
    dy = ty - fy

    # Handle promotions first
    if move.promotion:
        # Map to 0–15 index space (4 directions × 4 promotion pieces)
        if dx == 0 and dy == 1:
            dir_idx = 0
        elif dx == 1 and dy == 1:
            dir_idx = 1
        elif dx == -1 and dy == 1:
            dir_idx = 2
        elif dx == 0 and dy == -1:  # black promotion
            dir_idx = 3
        else:
            return None  # unsupported
        promo_idx = PROMOTION_PIECES.index(chess.piece_symbol(move.promotion))
        move_type = 56 + dir_idx * 4 + promo_idx
        return from_sq * 73 + move_type

    # Handle standard queen-like moves
    for dir_idx, (dx_step, dy_step) in enumerate(DIRECTIONS):
        for dist in range(1, 8):
            nx, ny = fx + dx_step * dist, fy + dy_step * dist
            if (nx, ny) == (tx, ty):
                move_type = dir_idx * 7 + (dist - 1)  # 0–55
                return from_sq * 73 + move_type
    return None  # unsupported move

def index_to_move(index: int) -> chess.Move:
    from_sq = index // 73
    move_type = index % 73

    if move_type < 56:
        dir_idx = move_type // 7
        dist = (move_type % 7) + 1
        fx, fy = square_to_coords(from_sq)
        dx, dy = DIRECTIONS[dir_idx]
        tx = fx + dx * dist
        ty = fy + dy * dist
        to_sq = coords_to_square(tx, ty)
        if to_sq is None:
            return None
        return chess.Move(from_sq, to_sq)

    else:
        promo_section = move_type - 56
        dir_idx = promo_section // 4
        promo_idx = promo_section % 4
        fx, fy = square_to_coords(from_sq)

        if dir_idx == 0:  # straight
            dx, dy = 0, 1
        elif dir_idx == 1:  # up-right
            dx, dy = 1, 1
        elif dir_idx == 2:  # up-left
            dx, dy = -1, 1
        elif dir_idx == 3:  # black promotion down
            dx, dy = 0, -1
        else:
            return None

        tx = fx + dx
        ty = fy + dy
        to_sq = coords_to_square(tx, ty)
        if to_sq is None:
            return None
        promo_piece = chess.Piece.from_symbol(PROMOTION_PIECES[promo_idx])
        return chess.Move(from_sq, to_sq, promotion=promo_piece.piece_type)
    
def board_to_tensor(board):
    piece_map = board.piece_map()
    board_tensor = torch.zeros(64 * 12)
    for square, piece in piece_map.items():
        offset = "PNBRQKpnbrqk".index(piece.symbol())
        board_tensor[64 * offset + square] = 1
    turn_tensor = torch.tensor([board.turn], dtype=torch.float32)
    return torch.cat([board_tensor, turn_tensor])

def legal_moves_mask(board):
    mask = torch.zeros(4672)
    for move in board.legal_moves:
        idx = move_to_index(move)
        if idx is not None:
            mask[idx] = 1
    return mask