import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import chess
import chess.pgn
import random
from helper import move_to_index, index_to_move, board_to_tensor, legal_moves_mask, eval_material, PIECE_VALUES
import json
import os
from datetime import datetime
from Visualize import visualize_game_ascii
import time
from torch.utils.tensorboard import SummaryWriter

#Define Policy Networks
class ActorCriticConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Shared convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(18, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Actor head
        self.policy_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 4672)  # Output logits for all possible moves
        )
        
        # Critic head
        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 1)  # Output a single value for the state
        )

    def forward(self, x):
        # Pass input through the shared convolutional layers
        features = self.conv(x)
        
        # Get policy and value outputs from their respective heads
        policy_logits = self.policy_head(features)
        state_value = self.value_head(features.detach()) # Detach features for the value head to prevent gradients from flowing into the shared layers from the value loss
        
        return policy_logits, state_value

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
            nn.Linear(512, 256),  
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

def train_actor_critic_game(actor_critic_net):
    """
    Plays a single game of self-play using the actor-critic network,
    collecting data for training.

    Args:
        actor_critic_net: The ActorCriticConvNet model.

    Returns:
        A tuple containing:
        - white_trajectory (list): A list of (state, log_prob, value) for white.
        - white_rewards (list): A list of rewards per move for white.
        - black_trajectory (list): A list of (state, log_prob, value) for black.
        - black_rewards (list): A list of rewards per move for black.
    """
    board = chess.Board()
    
    # Trajectories store data needed for loss calculation
    white_trajectory = []  # Stores (state, log_prob, state_value)
    black_trajectory = []
    white_rewards = []
    black_rewards = []

    while not board.is_game_over():
        # Get state, and pass it through the network
        state = board_to_tensor(board).unsqueeze(0)
        policy_logits, state_value = actor_critic_net(state)

        # Select an action (move) based on policy
        mask = legal_moves_mask(board)
        if mask.sum() == 0:
            print("No legal moves detected, aborting game.")
            break
            
        # Apply mask to logits to only consider legal moves
        probs = torch.softmax(policy_logits[0].masked_fill(mask == 0, -1e9), dim=0)
        dist = torch.distributions.Categorical(probs)
        move_idx = dist.sample() # Sample an action
        log_prob = dist.log_prob(move_idx) # Get the log probability of the action

        entropy = dist.entropy()

        move = index_to_move(move_idx.item())

        # Fallback for rare cases of invalid moves from the model
        if move is None or move not in board.legal_moves:
            print(f"Invalid move {move} proposed! Picking a random legal move.")
            move = random.choice(list(board.legal_moves))

        # Calculate immediate reward for the move
        immediate_reward = 0.0
        if board.is_capture(move):
            # Determine the type of the captured piece
            captured_piece_type = None
            if board.is_en_passant(move):
                captured_piece_type = chess.PAWN
            else:
                captured_piece = board.piece_at(move.to_square)
                if captured_piece:
                    captured_piece_type = captured_piece.piece_type
            
            # Calculate reward based on material value
            if captured_piece_type:
                immediate_reward = PIECE_VALUES.get(captured_piece_type, 0.0) / 20.0
            
        immediate_reward -= 0.0025 # small per move penalty
        
        #Store the trajectory data for the current player
        if board.turn == chess.WHITE:
            white_trajectory.append((state, log_prob, state_value, entropy))
            white_rewards.append(immediate_reward)
        else:
            black_trajectory.append((state, log_prob, state_value, entropy))
            black_rewards.append(immediate_reward)

        #Make the move on the board
        board.push(move)

        # if len(black_trajectory) > 120:
        #     break

    #Determine final game outcome and assign terminal rewards
    result = board.result()
    if result == "1-0":
        final_white_reward = 3.0
        final_black_reward = -2.0
    elif result == "0-1":
        final_white_reward = -2.0
        final_black_reward = 3.0
    else:  # Draw
        final_white_reward = -1
        final_black_reward = -1
        outcome = board.outcome()
        if outcome is not None:
            if outcome.termination == chess.Termination.FIVEFOLD_REPETITION:
                final_black_reward -= 0.5
                final_white_reward -= 0.5
            elif outcome.termination == chess.Termination.SEVENTYFIVE_MOVES:
                final_white_reward -= 0.75
                final_black_reward -= 0.75

    #Add the final game outcome reward to the last move's reward
    if white_rewards:
        white_rewards[-1] += final_white_reward
    if black_rewards:
        black_rewards[-1] += final_black_reward
    
    return white_trajectory, white_rewards, black_trajectory, black_rewards

#Play a Self-Game and Store Trajectory
def self_play_game(policy_net):
    board = chess.Board()
    white_trajectory = []
    black_trajectory = []
    white_capture_reward = 0
    black_capture_reward = 0

    while not board.is_game_over():
        state = board_to_tensor(board).unsqueeze(0)
        logits = policy_net(state)[0]
        print("Current board:")
        print(board)
        print("Legal moves:", [m.uci() for m in board.legal_moves])
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
            print(board)
            print(board.turn)
            logits = policy_net(state)  # shape: [1, 4672]
            probs = torch.softmax(logits, dim=-1).squeeze()

            topk = torch.topk(probs, 10)
            for idx, prob in zip(topk.indices.tolist(), topk.values.tolist()):
                move = index_to_move(idx)
                print(f"{move} — index {idx}, prob {prob:.4f}, legal: {move in board.legal_moves}")
            time.sleep(90)
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
        white_reward = 1#100/len(white_trajectory)# + white_capture_reward
        black_reward = -1 #00/len(black_trajectory)# - white_capture_reward #+ length_penalty
    elif result == "0-1":
        white_reward = -1#00/len(white_trajectory)# - black_capture_reward# + length_penalty
        black_reward = 1 #00/len(black_trajectory)# + black_capture_reward
    else:
        white_reward = 0 # + white_capture_reward - black_capture_reward #+ length_penalty 
        black_reward = 0 #- white_capture_reward + black_capture_reward #+ length_penalty
    
        outcome = board.outcome()
        if outcome.termination == chess.Termination.FIVEFOLD_REPETITION:
            black_reward -= 0.1
            white_reward -= 0.1
        elif outcome.termination == chess.Termination.SEVENTYFIVE_MOVES:
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
        is_white = i % 2 == 0  # alternate colors
        #is_white = True
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

        white_baseline = 0# sum(white_reward_history) / len(white_reward_history) if white_reward_history else 0
        black_baseline = 0# sum(black_reward_history) / len(black_reward_history) if black_reward_history else 0

        # --- ADJUST REWARDS ---
        adjusted_white_reward = white_reward - white_baseline
        adjusted_black_reward = black_reward - black_baseline
        print(f"adjusted rewards (white/black): ({adjusted_white_reward:.2f} {adjusted_black_reward:.2f})")

        white_loss = 0
        for (state, move_idx) in white_traj:
            logits = policy_net(state)[0]
            log_prob = torch.log_softmax(logits, dim=0)[move_idx]
            white_loss -= log_prob * adjusted_white_reward

        optimizer.zero_grad()
        white_loss.backward()
        optimizer.step()

        black_loss = 0
        for (state, move_idx) in black_traj:
            logits = policy_net(state)
            logits = logits.squeeze(0)
            log_prob = torch.log_softmax(logits, dim=0)[move_idx]
            black_loss -= log_prob * adjusted_black_reward
        
        optimizer.zero_grad()
        black_loss.backward()
        optimizer.step()

def train_vs_random(policy_net, model_name, optimizer, color='white', num_games=1000, eval_interval=1000):
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
            policy_loss -= log_prob * adjusted_reward 
        
        loss = policy_loss - entropy_loss
        #print(f"policy loss: {policy_loss}, entropy loss: {entropy_loss}, loss: {loss}")        
        # --- OPTIMIZATION ---
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Game {game+1}, reward (adjusted): {reward} ({adjusted_reward:.2f}), moves: {len(move_list)}, average entropy: {avg_entropy:.2f}, loss: {loss.item():.2f}")

def calculate_discounted_returns(rewards, gamma=0.99):
    """
    Calculates the discounted returns for each step in a list of rewards.
    The return for a step is the sum of all future rewards, discounted by gamma.

    Args:
        rewards (list): A list of rewards for an episode.
        gamma (float): The discount factor.

    Returns:
        A tensor of discounted returns.
    """
    returns = []
    discounted_return = 0.0
    # Iterate backwards through the rewards
    for r in reversed(rewards):
        discounted_return = r + gamma * discounted_return
        returns.insert(0, discounted_return)
    
    return torch.tensor(returns, dtype=torch.float32)

### Main Actor-Critic Training Function
def train_actor_critic(actor_critic_net, model_name, optimizer, num_games=1500, eval_interval=500, gamma=0.99, critic_loss_weight=0.5, entropy_weight=0.01, writer=None):
    """
    Main training loop for the Actor-Critic model.

    Args:
        actor_critic_net: The ActorCriticConvNet model.
        model_name (str): The name for saving logs and checkpoints.
        optimizer: The PyTorch optimizer.
        num_games (int): The total number of self-play games to train on.
        eval_interval (int): The interval at which to evaluate the model.
        gamma (float): The discount factor for future rewards.
        critic_loss_weight (float): The weight to apply to the critic's loss.
    """    
    for game in range(num_games):
        # Generate game data using the new actor-critic game function
        white_traj, white_rewards, black_traj, black_rewards = train_actor_critic_game(actor_critic_net)

        # Skip update if a game ends prematurely with no moves
        if not white_traj or not black_traj:
            print(f"Game {game+1}: Skipped due to empty trajectory.")
            continue

        print(f"Game {game+1}, Total Moves: {len(white_traj) + len(black_traj)}, Final White Reward: {sum(white_rewards):.2f}, Final Black Reward: {sum(black_rewards):.2f}")

        # Prepare data for loss calculation
        all_log_probs = []
        all_state_values = []
        all_returns = []
        all_entropies = []

        # Process White's trajectory
        if white_traj:
            white_returns = calculate_discounted_returns(white_rewards, gamma)
            all_returns.extend(white_returns)
            for (_, log_prob, state_value, entropy), R in zip(white_traj, white_returns):
                all_log_probs.append(log_prob)
                all_state_values.append(state_value)
                all_entropies.append(entropy)

        # Process Black's trajectory
        if black_traj:
            black_returns = calculate_discounted_returns(black_rewards, gamma)
            all_returns.extend(black_returns)
            for (_, log_prob, state_value, entropy), R in zip(black_traj, black_returns):
                all_log_probs.append(log_prob)
                all_state_values.append(state_value)
                all_entropies.append(entropy)

        #Convert lists to tensors for batch processing
        log_probs_tensor = torch.stack(all_log_probs)
        state_values_tensor = torch.cat(all_state_values).squeeze()
        returns_tensor = torch.stack(all_returns)
        entropies_tensor = torch.stack(all_entropies)
        
        #Calculate Advantage (A = R - V(s))
        # .detach() is used so that we don't propagate gradients through the value function here.
        advantages = returns_tensor - state_values_tensor.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) #normalize

        #Calculate Actor Loss (Policy Loss)
        actor_loss = -(log_probs_tensor * advantages).mean()

        #Calculate Critic Loss (Value Loss)
        critic_loss = F.mse_loss(state_values_tensor, returns_tensor)

        #Entropy loss over one game
        entropy_loss = -entropies_tensor.mean()

        #Calculate Total Loss and perform backpropagation
        total_loss = actor_loss + critic_loss_weight * critic_loss + entropy_weight * entropy_loss

        writer.add_scalar("Loss/Total", total_loss.item(), game)
        writer.add_scalar("Loss/Actor", actor_loss.item(), game)
        writer.add_scalar("Loss/Critic", critic_loss.item(), game)
        writer.add_scalar("Loss/Entropy", entropy_loss, game)
        writer.add_scalar("Training/FinalReward_White", sum(white_rewards), game)
        writer.add_scalar("Training/FinalReward_Black", sum(black_rewards), game)
        writer.add_scalar("Training/GameLength", len(white_traj) + len(black_traj), game)
        
        optimizer.zero_grad()
        total_loss.backward()
        # Optional: Clip gradients to prevent them from exploding
        torch.nn.utils.clip_grad_norm_(actor_critic_net.parameters(), max_norm=0.5)
        optimizer.step()

        if (game + 1) % 10 == 0:
            print(f"Game {game+1}: Total Loss: {total_loss.item():.4f}, Actor: {actor_loss.item():.4f}, Critic: {critic_loss.item():.4f}, Entropy: {entropy_loss.item():.4f}")
            writer.add_histogram("critic/state_values", state_values_tensor, game)
            writer.add_histogram("critic/returns", returns_tensor, game)
            writer.add_histogram("critic/advantages", advantages, game)
        
        # Evaluate every X games
        if (game + 1) % eval_interval == 0:
            eval_stats = evaluate_vs_random_a2c(actor_critic_net, game, num_games=100, writer=writer)
            print(f"\n[Eval at game {game}] Wins: {eval_stats['wins']}, Draws: {eval_stats['draws']}, Losses: {eval_stats['losses']}\n")
            log_evaluation(eval_stats, model_name)
            torch.save(actor_critic_net.state_dict(), f"actor_critic_{model_name}_checkpoint_{game}.pt")

def log_evaluation(results, out_dir="evaluation_logs"):
    out_dir = 'eval_logs/' + out_dir
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(out_dir, f"eval_{timestamp}.json")

    with open(filepath, "w") as f:
        json.dump(results, f, indent=4)

def evaluate_vs_random_a2c(actor_critic_net, game_num, num_games=50, writer=None):
    """
    Evaluates the actor-critic model's policy against a random opponent.

    Args:
        actor_critic_net: The trained ActorCriticConvNet model.
        num_games (int): The number of games to play for the evaluation.

    Returns:
        A dictionary containing evaluation statistics.
    """
    entropies = []
    # Set the network to evaluation mode
    actor_critic_net.eval()
    
    results = []
    white_wins, black_wins, ties, policy_white_wins, policy_black_wins = 0, 0, 0, 0, 0
    game_lengths = []

    # Disable gradients for performance, as they are not needed for evaluation
    with torch.no_grad():
        for i in range(num_games):
            board = chess.Board()
            # The policy model plays as white in even games, and black in odd games
            is_policy_white = i % 2 == 0
            move_count = 0

            while not board.is_game_over():
                # Check if it's the policy's turn to move
                if (board.turn == chess.WHITE and is_policy_white) or \
                   (board.turn == chess.BLACK and not is_policy_white):
                    
                    # --- MODEL INFERENCE ---
                    state = board_to_tensor(board).unsqueeze(0)
                    policy_logits, _ = actor_critic_net(state)
                    logits = policy_logits[0]
                    
                    mask = legal_moves_mask(board)
                    if mask.sum() == 0:
                        print("Evaluation game aborted: No legal moves for policy.")
                        break
                        
                    masked_logits = logits.masked_fill(mask == 0, -1e9)
                    probs = torch.softmax(masked_logits, dim=0)
                    
                    dist = torch.distributions.Categorical(probs)
                    entropy = dist.entropy().item()
                    entropies.append(entropy)
                    
                    move_idx = torch.argmax(probs).item()
                    move = index_to_move(move_idx)
                    
                    if move is None or move not in board.legal_moves:
                        # Needs more loggins, this should not really happen
                        move = random.choice(list(board.legal_moves))
                else:
                    move = random.choice(list(board.legal_moves))
                
                board.push(move)
                move_count += 1

            # --- RECORD GAME RESULTS ---
            game_lengths.append(move_count)
            result = board.result()
            outcome = 0 # Default to draw
            
            if result == "1-0": # White won
                white_wins += 1
                if is_policy_white:
                    outcome = 1
                    policy_white_wins += 1
                else:
                    outcome = -1
            elif result == "0-1": # Black won
                black_wins += 1
                if not is_policy_white:
                    outcome = 1
                    policy_black_wins += 1
                else:
                    outcome = -1
            else: # Draw
                ties += 1
            
            results.append(outcome)

    # Set the network back to training mode
    actor_critic_net.train()

    wins = results.count(1)
    losses = results.count(-1)
    
    print(f"\n--- Evaluation Results ---")
    print(f"Policy vs Random: {wins} Wins / {ties} Draws / {losses} Losses")
    print(f"Policy as White Wins: {policy_white_wins}, Policy as Black Wins: {policy_black_wins}")
    print(f"--------------------------\n")

    writer.add_scalar("Eval/Wins", wins, game_num)
    writer.add_scalar("Eval/Losses", losses, game_num)
    writer.add_scalar("Eval/Draws", ties, game_num)
    writer.add_scalar("Eval/PolicyWhiteWins", policy_white_wins, game_num)
    writer.add_scalar("Eval/PolicyBlackWins", policy_black_wins, game_num)
    writer.add_scalar("Eval/AvgGameLength", sum(game_lengths) / len(game_lengths), game_num)
    writer.add_scalar("Eval/AvgEntropy", sum(entropies) / len(entropies), game_num)

    return {
        "white_wins": white_wins,
        "black_wins": black_wins,
        "draws": ties,
        "wins": wins,
        "losses": losses,
        "policy_white_wins": policy_white_wins,
        "policy_black_wins": policy_black_wins,
        "avg_game_length": sum(game_lengths) / len(game_lengths) if game_lengths else 0,
        "avg_entropy": sum(entropies) / len(entropies) if entropies else 0
    }

if __name__ == "__main__":
    actor_critic_net = ActorCriticConvNet()    
    model_name = 'actor_critic_chess_symmetry'
    model_filename = model_name + ".pt"

    # Load pre-trained weights if they exist
    if os.path.exists(model_filename):
        print(f"Loading pre-trained model from: {model_filename}")
        actor_critic_net.load_state_dict(torch.load(model_filename))
    else:
        print("Initializing a new model.")
        
    optimizer = optim.Adam(actor_critic_net.parameters(), lr=5e-4)
    writer = SummaryWriter(log_dir=f"runs/{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    train_actor_critic(
        actor_critic_net=actor_critic_net,
        model_name=model_name,
        optimizer=optimizer,
        gamma=0.97,
        num_games=10000,
        eval_interval=1000,
        entropy_weight=0.025,
        writer=writer
    )

    print(f"Training finished. Saving final model to: {model_filename}")
    torch.save(actor_critic_net.state_dict(), model_filename)