import torch
import torch.nn as nn
import torch.optim as optim
import chess
import chess.pgn
import random
from helper import move_to_index, index_to_move, board_to_tensor, legal_moves_mask, PIECE_VALUES
import json
import os
from datetime import datetime

#Define Policy Network
class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(769, 256),  # board state input size
            nn.ReLU(),
            nn.Linear(256, 4672)  # total legal UCI moves in chess (e.g., e2e4)
        )

    def forward(self, x):
        return self.net(x)

#Play a Self-Game and Store Trajectory ===
def self_play_game(policy_net):
    board = chess.Board()
    white_trajectory = []
    black_trajectory = []
    white_capture_reward = 0
    black_capture_reward = 0

    while not board.is_game_over():
        state = board_to_tensor(board)
        logits = policy_net(state)
        mask = legal_moves_mask(board)
        if mask.sum() == 0:
            print("no legal moves detected, aborting")
            break
        probs = torch.softmax(logits.masked_fill(mask == 0, -1e9), dim=0)
        dist = torch.distributions.Categorical(probs)
        move_idx = dist.sample().item()

        move = index_to_move(move_idx)
        if move is None or move not in board.legal_moves:
            print("Invalid move proposed!")
            print("Move index:", move_idx)
            print("Decoded move:", move)
            print("Legal moves:", [m.uci() for m in board.legal_moves])
            print("Softmax top move UCI:", index_to_move(torch.argmax(probs).item()))
            move = random.choice(list(board.legal_moves))

        captured_piece = board.piece_at(move.to_square)
        if captured_piece:
            capture_reward = PIECE_VALUES.get(captured_piece.piece_type, 0.0) * 0.025
            if board.turn == chess.WHITE:
                white_capture_reward += capture_reward
                #black_reward -= reward
            else:
                black_capture_reward += capture_reward
                #white_reward -= reward
        
        if board.turn == chess.WHITE:
            white_trajectory.append((state, move_idx))
        else:
            black_trajectory.append((state, move_idx))

        board.push(move)

    result = board.result()
    if result == "1-0":
        white_reward = 2.5 + white_capture_reward
        black_reward = -1 -len(black_trajectory)/1000
    elif result == "0-1":
        white_reward = -1 -len(white_trajectory)/1000
        black_reward = 2.5 + black_capture_reward
    else:
        white_reward = -len(white_trajectory)/1000
        black_reward = -len(black_trajectory)/1000
    return white_trajectory, white_reward, black_trajectory, black_reward

def play_vs_random(policy_net, num_games=100):
    entropies = []
    policy_net.eval()
    results = []
    white_wins, black_wins, ties, policy_white_wins, policy_black_wins = 0,0,0,0,0
    game_lengths = []

    for i in range(num_games):
        board = chess.Board()
        #is_white = i % 2 == 0  # alternate colors
        is_white = True
        move_count = 0

        while not board.is_game_over():
            if board.turn == is_white:
                #print(f"turn: {board.turn}, policy plays")
                # Policy plays
                state = board_to_tensor(board)
                logits = policy_net(state)
                mask = legal_moves_mask(board)
                if mask.sum() == 0:
                    print(f"aborting game do to no legal moves available")
                    break
                masked_logits = logits.masked_fill(mask == 0, -1e9)
                probs = torch.softmax(masked_logits, dim=0)
                #probs = torch.softmax(logits.masked_fill(mask == 0, -1e9), dim=0)
                dist = torch.distributions.Categorical(probs)
                entropy = dist.entropy().item()
                entropies.append(entropy)
                move_idx = dist.sample().item()
                move = index_to_move(move_idx)
                if move is None or move not in board.legal_moves:
                    move = random.choice(list(board.legal_moves))
                    print("warning")
            else:
                # Random bot
                move = random.choice(list(board.legal_moves))
            board.push(move)
            move_count += 1

        game_lengths.append(move_count)
        result = board.result()
        if result == "1-0":
            white_wins +=1
    
            if is_white:
                outcome = 1
                policy_white_wins += 1
            else:
                outcome = -1

        elif result == "0-1":
            black_wins += 1

            if is_white:
                outcome = -1
            else:
                outcome = 1
                policy_black_wins += 1
        else:
            outcome = 0
            ties += 1
        results.append(outcome)
        print(f"game finished, result: {result}")

    wins = results.count(1)
    draws = results.count(0)
    losses = results.count(-1)
    print(f"Policy vs Random — {wins} Wins / {draws} Draws / {losses} Losses / pww {policy_white_wins} / pbw {policy_black_wins}")
    return {
        "white_wins": white_wins,
        "black_wins": black_wins,
        "draws": draws,
        "wins": wins,
        "losses": losses,
        "policy_white_wins": policy_white_wins,
        "policy_black_wins": policy_black_wins,
        "avg_game_length": sum(game_lengths) / len(game_lengths),
        "avg_entropy": sum(entropies) / len(entropies)
    }

def train(policy_net, optimizer, num_games=1500, eval_interval=500):
    for game in range(num_games):

        if game % eval_interval == 0 and game != 0:
            eval_stats = play_vs_random(policy_net)
            print(f"[Eval at game {game}] Wins: {eval_stats['wins']}, Draws: {eval_stats['draws']}, Losses: {eval_stats['losses']}")
            log_evaluation(eval_stats)
            #torch.save(policy_net.state_dict(), f"policy_seperate_colors_checkpoint_{game}.pt")

        white_traj, white_reward, black_traj, black_reward = self_play_game(policy_net)
        print(f"Game {game+1}, white_reward: {white_reward}, moves: {len(white_traj)+len(black_traj)}")
        if white_reward == 0:
            continue

        white_loss = 0
        for (state, move_idx) in white_traj:
            logits = policy_net(state)
            log_prob = torch.log_softmax(logits, dim=0)[move_idx]
            white_loss -= log_prob * white_reward

        optimizer.zero_grad()
        white_loss.backward()
        optimizer.step()
"""
        black_loss = 0
        for (state, move_idx) in black_traj:
            logits = policy_net(state)
            log_prob = torch.log_softmax(logits, dim=0)[move_idx]
            black_loss -= log_prob * black_reward
          
        optimizer.zero_grad()
        black_loss.backward()
        optimizer.step()
"""

def log_evaluation(results, out_dir="eval_logs"):
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(out_dir, f"eval_{timestamp}.json")

    with open(filepath, "w") as f:
        json.dump(results, f, indent=4)

    #print(f"Evaluation logged to {filepath}")

if __name__ == "__main__":
    policy_net = PolicyNet()
    #"""
    policy_net.load_state_dict(torch.load("policy_seperate_colors_20250712_only_white.pt"))
    optimizer = optim.Adam(policy_net.parameters(), lr=0.8e-3)
    train(policy_net, optimizer, num_games=5001, eval_interval=1000)
    torch.save(policy_net.state_dict(), "policy_seperate_colors_20250712_only_white.pt")
    #"""
    #policy_net.load_state_dict(torch.load("policy_seperate_colors_20250711_knights.pt"))
    #play_vs_random(policy_net, 100)