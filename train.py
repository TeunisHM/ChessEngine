import torch
import torch.nn as nn
import torch.optim as optim
import chess
import chess.pgn
import random
from helper import move_to_index, index_to_move, board_to_tensor, legal_moves_mask, eval_material, PIECE_VALUES
import json
import os
from datetime import datetime
from Visualize import visualize_game_ascii

#Define Policy Networks
class ConvPolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(13, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 4672)  # output logits for all possible moves
        )

    def forward(self, x):  # x shape: (13, 8, 8)
        x = self.conv(x)
        return self.head(x)

class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(769, 512),  # board state input size
            nn.ReLU(),
            nn.Linear(512, 256),  # --- ADDED HIDDEN LAYER ---
            nn.ReLU(),
            nn.Linear(256, 4672)  # total legal UCI moves in chess (e.g., e2e4)
        )

    def forward(self, x):
        return self.net(x)
    
def compare_weights(initial, trained):
    total_diff = 0.0
    for name in initial:
        diff = torch.norm(initial[name] - trained[name]).item()
        print(f"{name}: L2 diff = {diff:.4f}")
        total_diff += diff
    print(f"Total L2 weight difference: {total_diff:.4f}")

#Play a Self-Game and Store Trajectory ===
def self_play_game(policy_net):
    board = chess.Board()
    white_trajectory = []
    black_trajectory = []
    white_capture_reward = 0
    black_capture_reward = 0

    while not board.is_game_over():
        state = board_to_tensor(board).unsqueeze(0)
        logits = policy_net(state)[0]
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
            capture_reward = PIECE_VALUES.get(captured_piece.piece_type, 0.0) * 0.02
            if board.turn == chess.WHITE:
                white_capture_reward += capture_reward
            else:
                black_capture_reward += capture_reward
        
        if board.turn == chess.WHITE:
            white_trajectory.append((state, move_idx))
        else:
            black_trajectory.append((state, move_idx))

        board.push(move)

    length_penalty = min(0, (40 - len(black_trajectory))/1000)

    result = board.result()
    if result == "1-0":
        white_reward = 100/len(white_trajectory)# + white_capture_reward
        black_reward = -100/len(black_trajectory)# - white_capture_reward #+ length_penalty
    elif result == "0-1":
        white_reward = -100/len(white_trajectory)# - black_capture_reward# + length_penalty
        black_reward = 100/len(black_trajectory)# + black_capture_reward
    else:
        white_reward = 0 # + white_capture_reward - black_capture_reward #+ length_penalty 
        black_reward = 0 #- white_capture_reward + black_capture_reward #+ length_penalty
    
    outcome = board.outcome()
    if outcome.termination == chess.Termination.THREEFOLD_REPETITION:
        black_reward -= 0.1
        white_reward -= -0.1
    elif outcome.termination == chess.Termination.FIFTY_MOVES:
        white_reward -= 0.2
        black_reward -= 0.2
    return white_trajectory, white_reward, black_trajectory, black_reward

def evaluate_vs_random(policy_net, num_games=50):
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
                state = board_to_tensor(board).unsqueeze(0)
                logits = policy_net(state)[0]
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

def train(policy_net, model_name, optimizer, num_games=1500, eval_interval=500):
    white_reward_history = []
    black_reward_history = []
    for game in range(num_games):

        if game % eval_interval == 0 and game != 0:
            eval_stats = evaluate_vs_random(policy_net)
            print(f"[Eval at game {game}] Wins: {eval_stats['wins']}, Draws: {eval_stats['draws']}, Losses: {eval_stats['losses']}")
            log_evaluation(eval_stats, model_name)
            #torch.save(policy_net.state_dict(), f"policy_seperate_colors_checkpoint_{game}.pt")

        white_traj, white_reward, black_traj, black_reward = self_play_game(policy_net)

        # move_list = []
        # for pair in zip(white_traj, black_traj):
        #     for item in pair:
        #         move_list.append(index_to_move(item[1]))
        #         print(move_list)
        # visualize_game_ascii(move_list)
        
        print(f"Game {game+1}, white_reward: {white_reward}, moves: {len(white_traj)+len(black_traj)}")

        # Add rewards to history for baseline calculation
        white_reward_history.append(white_reward)
        black_reward_history.append(black_reward)
        if len(black_reward_history) > 500: 
            black_reward_history.pop(0)
        if len(white_reward_history) > 500: 
            white_reward_history.pop(0)

        white_baseline = sum(white_reward_history) / len(white_reward_history) if white_reward_history else 0
        black_baseline = sum(black_reward_history) / len(black_reward_history) if black_reward_history else 0

        # --- ADJUST REWARDS ---
        adjusted_white_reward = white_reward - white_baseline
        adjusted_black_reward = black_reward - black_baseline

        white_loss = 0
        for (state, move_idx) in white_traj:
            logits = policy_net(state)[0]
            log_prob = torch.log_softmax(logits, dim=0)[move_idx]
            white_loss -= log_prob * adjusted_white_reward

        print(f"white loss: {white_loss.item()}")
        optimizer.zero_grad()
        white_loss.backward()
        optimizer.step()

        black_loss = 0
        for (state, move_idx) in black_traj:
            logits = policy_net(state)
            logits = logits.squeeze(0)
            log_prob = torch.log_softmax(logits, dim=0)[move_idx]
            black_loss -= log_prob * adjusted_black_reward
        
        print(f"black loss: {black_loss.item()}")
        optimizer.zero_grad()
        black_loss.backward()
        optimizer.step()

def train_vs_random(policy_net, model_name, optimizer, color='white', num_games=1000, eval_interval=500):
    #This function trains a policy network against an opponent that makes random moves.
    #It uses a simple reinforcement learning algorithm called REINFORCE.
    reward_history = []
    for game in range(num_games):
        
        move_list = []
        # Periodically evaluate the model against a random opponent to track progress
        if game % eval_interval == 0 and game != 0:
            eval_stats = evaluate_vs_random(policy_net)
            print(f"[Eval at game {game}] Wins: {eval_stats['wins']}, Draws: {eval_stats['draws']}, Losses: {eval_stats['losses']}")
            log_evaluation(eval_stats, model_name)

        trajectory = []
        board = chess.Board()
        # Set the color of the policy network
        is_white = True if color == 'white' else False

        # --- GAME SIMULATION ---
        while not board.is_game_over():
            if board.turn == is_white:
                # --- POLICY'S TURN ---
                state = board_to_tensor(board).unsqueeze(0)
                logits = policy_net(state)[0]
                mask = legal_moves_mask(board)
                if mask.sum() == 0:
                    print(f"aborting game do to no legal moves available")
                    break
                # Apply a mask to the logits to only consider legal moves
                masked_logits = logits.masked_fill(mask == 0, -1e9)
                probs = torch.softmax(masked_logits, dim=0)
                dist = torch.distributions.Categorical(probs)
                entropy = dist.entropy()
                move_idx = dist.sample().item()
                move = index_to_move(move_idx)
                # Store the state, move, logits, and entropy for later use in training
                trajectory.append((state, move_idx, masked_logits, entropy))
                if move is None or move not in board.legal_moves:
                    print(f"warning, illegal or none move selected by policy, move: {move}")
                    move = random.choice(list(board.legal_moves))
            else:
                # --- RANDOM OPPONENT'S TURN ---
                move = random.choice(list(board.legal_moves))
            move_list.append(move)
            board.push(move)

            if len(move_list) > 150:
                break

        # --- REWARD CALCULATION ---
        result = board.result()        

        if result == "1-0":
            reward = 1 if is_white else -1
        elif result == "0-1":
            reward = -1 if is_white else 1
        else:
            reward = 0        
            outcome = board.outcome()
            if outcome is not None:
                # Add small penalties for certain types of draws
                if outcome.termination == chess.Termination.FIVEFOLD_REPETITION:
                    reward -= 0.1
                elif outcome.termination == chess.Termination.SEVENTYFIVE_MOVES:
                    reward -= 0.2
                # In case of a stalemate, use material advantage to assign a small reward
                elif outcome.termination == chess.Termination.STALEMATE:
                    score = eval_material(board)
                    reward += score/40
                    print(f"piece reward in stalemate: {score/40}")

        # --- ADVANTAGE CALCULATION (REWARD - BASELINE) ---
        reward_history.append(reward)
        if len(reward_history) > 500: 
            reward_history.pop(0)
        # Calculate a baseline reward to reduce variance in the training signal
        baseline = sum(reward_history) / len(reward_history) if reward_history else 0
        adjusted_reward = reward - baseline

        # --- LOSS CALCULATION ---
        avg_entropy = sum(e.item() for _, _,_, e in trajectory) / len(trajectory) if trajectory else 0
        policy_loss = 0
        entropy_loss = 0.1 * avg_entropy
        
        for (state, move_idx, masked_logits, entropy) in trajectory:
            log_prob = torch.log_softmax(masked_logits, dim=0)[move_idx]
            # The policy loss is the log probability of the chosen move multiplied by the adjusted reward
            policy_loss -= log_prob * adjusted_reward 
        
        loss = policy_loss - entropy_loss
        #print(f"policy loss: {policy_loss}, entropy loss: {entropy_loss}, loss: {loss}")        
        # --- OPTIMIZATION ---
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Game {game+1}, reward (adjusted): {reward} ({adjusted_reward}), moves: {len(move_list)}, average entropy: {avg_entropy}, loss: {loss.item()}")

def log_evaluation(results, out_dir="evaluation_logs"):
    out_dir = 'eval_logs/' + out_dir
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(out_dir, f"eval_{timestamp}.json")

    with open(filepath, "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    policy_net = ConvPolicyNet() #PolicyNet()
    model_name = 'conv_white_vs_random_entropy_short'
    #"""
    if os.path.exists(model_name + ".pt"):
        policy_net.load_state_dict(torch.load(model_name+".pt"))
    optimizer = optim.Adam(policy_net.parameters(), lr=0.5e-3)
    train_vs_random(policy_net, model_name, optimizer, num_games=2501, eval_interval=500)
    torch.save(policy_net.state_dict(), model_name+".pt")
    #"""
    #policy_net.load_state_dict(torch.load("policy_seperate_colors_20250711_knights.pt"))
    #play_vs_random(policy_net, 100)