import torch
import torch.nn as nn
import torch.optim as optim
import chess
import chess.pgn
import random
import helper
from train import PolicyNet

def play_vs_random(policy_net, num_games=10):
    policy_net.eval()
    results = []

    for i in range(num_games):
        board = chess.Board()
        is_white = i % 2 == 0  # alternate colors

        while not board.is_game_over():
            if board.turn == is_white:
                # Policy plays
                state = helper.board_to_tensor(board)
                logits, _ = policy_net(state)
                mask = helper.legal_moves_mask(board)
                if mask.sum() == 0:
                    break
                probs = torch.softmax(logits.masked_fill(mask == 0, -1e9), dim=0)
                dist = torch.distributions.Categorical(probs)
                move_idx = dist.sample().item()
                move = helper.index_to_move(move_idx)
                if move is None or move not in board.legal_moves:
                    move = random.choice(list(board.legal_moves))
            else:
                # Random bot
                move = random.choice(list(board.legal_moves))
            board.push(move)

        result = board.result()
        if result == "1-0":
            outcome = 1 if is_white else -1
        elif result == "0-1":
            outcome = -1 if is_white else 1
        else:
            outcome = 0
        results.append(outcome)

    wins = results.count(1)
    draws = results.count(0)
    losses = results.count(-1)
    print(f"Policy vs Random — {wins} Wins / {draws} Draws / {losses} Losses")
    return wins, draws, losses

if __name__ == '__main__':
    policy_net = PolicyNet() 
    #policy = policy_net.load_state_dict(torch.load("policy.pt"))
    #wins, draws, losses = play_vs_random(policy)
    #print(f"wins: {wins}, draws: {draws}, losses: {losses}")