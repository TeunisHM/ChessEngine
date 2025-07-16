import time
import chess

def visualize_game_ascii(game_moves, delay=0.1):
    board = chess.Board()
    for i, move in enumerate(game_moves):
        print(f"\nMove {i+1}: {board.san(move)}")
        board.push(move)
        print(board)
        time.sleep(delay)
