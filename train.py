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
from time import perf_counter

#Define Policy Networks
class ResidualBlock(nn.Module):
    """A standard residual block for a ResNet."""
    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels) # Batch Norm is crucial
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        # Store the original input for the skip connection
        residual = x
        
        # First conv layer
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        # Second conv layer
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add the residual (skip connection)
        out += residual
        
        # Apply final activation
        out = F.relu(out)
        
        return out

class ActorCriticResNet(nn.Module):
    def __init__(self, num_input_channels=18, num_residual_blocks=8, num_filters=128):
        super().__init__()
        
        # Initial Convolutional Layer (the "stem")
        self.stem = nn.Sequential(
            nn.Conv2d(num_input_channels, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU()
        )
        
        # This creates a list of 'num_residual_blocks' ResidualBlock modules
        self.residual_tower = nn.Sequential(
            *[ResidualBlock(num_filters) for _ in range(num_residual_blocks)]
        )
        
        # Actor (Policy) Head
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters, 2, kernel_size=1), # Reduce to 2 filters
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, 4672)
        )
        
        # Critic (Value) Head
        self.value_head = nn.Sequential(
            nn.Conv2d(num_filters, 1, kernel_size=1), # Reduce to 1 filter
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        # Pass through the initial stem
        out = self.stem(x)
        
        # Pass through the residual tower
        out = self.residual_tower(out)
        
        # Get policy and value outputs
        policy_logits = self.policy_head(out)
        state_value = self.value_head(out) # No detach here, following AlphaZero's design
        
        return policy_logits, state_value

def generate_self_play_batch(actor_critic_net, batch_size=8):
    """
    Runs multiple self-play games in parallel and collects trajectories.
    Returns:
        - log_probs: list of log probabilities
        - state_values: list of state values
        - returns: list of discounted returns
        - entropies: list of entropies
    """
    all_log_probs = []
    all_state_values = []
    all_returns = []
    all_entropies = []

    for _ in range(batch_size):
        white_traj, white_rewards, black_traj, black_rewards = train_actor_critic_game(actor_critic_net)
        if not white_traj or not black_traj:
            continue  # skip broken games

        # White
        white_returns = calculate_discounted_returns(white_rewards)
        for (_, log_prob, value, entropy), R in zip(white_traj, white_returns):
            all_log_probs.append(log_prob)
            all_state_values.append(value)
            all_returns.append(R)
            all_entropies.append(entropy)

        # Black
        black_returns = calculate_discounted_returns(black_rewards)
        for (_, log_prob, value, entropy), R in zip(black_traj, black_returns):
            all_log_probs.append(log_prob)
            all_state_values.append(value)
            all_returns.append(R)
            all_entropies.append(entropy)

    return all_log_probs, all_state_values, all_returns, all_entropies

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

        move = index_to_move(move_idx.item(), board)

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
                capture_value = PIECE_VALUES.get(captured_piece_type, 0.0) / 20.0
                immediate_reward += capture_value

                """
                if board.turn == chess.WHITE:
                    if black_rewards: 
                        black_rewards[-1] -= 0.5 * capture_value
                else: 
                    if white_rewards: 
                        white_rewards[-1] -= 0.5 * capture_value
                """
            
        immediate_reward -= 0.002 # small per move penalty
        
        #Store the trajectory data for the current player
        if board.turn == chess.WHITE:
            white_trajectory.append((state, log_prob, state_value, entropy))
            white_rewards.append(immediate_reward)
        else:
            black_trajectory.append((state, log_prob, state_value, entropy))
            black_rewards.append(immediate_reward)

        board.push(move)

        # if len(black_trajectory) > 120:
        #     break

    #Determine final game outcome and assign terminal rewards
    result = board.result()
    if result == "1-0":
        final_white_reward = 2.0
        final_black_reward = -1.0
    elif result == "0-1":
        final_white_reward = -1.0
        final_black_reward = 2.0
    else:  # Draw
        final_white_reward = -0.5
        final_black_reward = -0.5
        outcome = board.outcome()
        if outcome is not None:
            if outcome.termination == chess.Termination.FIVEFOLD_REPETITION:
                final_black_reward -= 0.25
                final_white_reward -= 0.25
            elif outcome.termination == chess.Termination.SEVENTYFIVE_MOVES:
                final_white_reward -= 0.5
                final_black_reward -= 0.5

    #Add the final game outcome reward to the last move's reward
    if white_rewards:
        white_rewards[-1] += final_white_reward
    if black_rewards:
        black_rewards[-1] += final_black_reward
    
    return white_trajectory, white_rewards, black_trajectory, black_rewards

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
def train_actor_critic(actor_critic_net, model_name, optimizer, num_batches=1000, eval_interval=500, gamma=0.99, critic_loss_weight=0.5, entropy_weight=0.025, batch_size=4, writer=None):
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
    for batch in range(num_batches):
        t0 = perf_counter()
        log_probs, state_values, returns, entropies = generate_self_play_batch(actor_critic_net, batch_size=batch_size)
        t1 = perf_counter()

        if not log_probs:
            print(f"Game {batch+1}: Skipped due to empty batch.")
            continue

        log_probs_tensor = torch.stack(log_probs)
        state_values_tensor = torch.cat(state_values).squeeze()
        returns_tensor = torch.stack(returns)
        entropies_tensor = torch.stack(entropies)

        # Advantage normalization
        advantages = returns_tensor - state_values_tensor.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_loss = -(log_probs_tensor * advantages).mean()
        critic_loss = F.mse_loss(state_values_tensor, returns_tensor)
        entropy_loss = -entropies_tensor.mean()
        total_loss = actor_loss + critic_loss_weight * critic_loss + entropy_weight * entropy_loss

        writer.add_scalar("Loss/Total", total_loss.item(), batch)
        writer.add_scalar("Loss/Actor", actor_loss.item(), batch)
        writer.add_scalar("Loss/Critic", critic_loss.item(), batch)
        writer.add_scalar("Loss/Entropy", entropy_loss.item(), batch)

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor_critic_net.parameters(), max_norm=0.5)
        optimizer.step()
        t2 = perf_counter()

        print(f"[Batch {batch+1}] DataGen: {t1 - t0:.2f}s | TrainStep: {t2 - t1:.2f}s | Total: {t2 - t0:.2f}s")

        if (batch + 1) % 10 == 0:
            print(f"Batch {batch+1}: Total Loss: {total_loss.item():.4f}, Actor: {actor_loss.item():.4f}, Critic: {critic_loss.item():.4f}, Entropy: {entropy_loss.item():.4f}")
            writer.add_histogram("critic/state_values", state_values_tensor, batch)
            writer.add_histogram("critic/returns", returns_tensor, batch)
            writer.add_histogram("critic/advantages", advantages, batch)
        
        # Evaluate every X games
        if (batch + 1) % eval_interval == 0:
            eval_stats = evaluate_vs_random_a2c(actor_critic_net, batch, num_games=100, writer=writer)
            print(f"\n[Eval at game {batch}] Wins: {eval_stats['wins']}, Draws: {eval_stats['draws']}, Losses: {eval_stats['losses']}\n")
            log_evaluation(eval_stats, model_name)
            torch.save(actor_critic_net.state_dict(), f"actor_critic_{model_name}_checkpoint_{batch}.pt")

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
                    move = index_to_move(move_idx, board)
                    
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
    actor_critic_net = ActorCriticResNet()    
    model_name = 'actor_critic_chess_resnet_v1'
    model_filename = model_name + ".pt"

    # Load pre-trained weights if they exist
    if os.path.exists(model_filename):
        print(f"Loading pre-trained model from: {model_filename}")
        actor_critic_net.load_state_dict(torch.load(model_filename))
    else:
        print("Initializing a new model.")
        # Write the model graph to the TensorBoard log
        #dummy_input = torch.zeros(1, 18, 8, 8)
        #writer.add_graph(actor_critic_net, dummy_input)
        
    optimizer = optim.Adam(actor_critic_net.parameters(), lr=5e-4)
    writer = SummaryWriter(log_dir=f"runs/{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    train_actor_critic(
        actor_critic_net=actor_critic_net,
        model_name=model_name,
        optimizer=optimizer,
        gamma=0.982,
        num_batches=1000,
        eval_interval=100,
        entropy_weight=0.025,
        writer=writer,
        batch_size=8
    )

    print(f"Training finished. Saving final model to: {model_filename}")
    torch.save(actor_critic_net.state_dict(), model_filename)