import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import chess
import random
from helper import move_to_index, index_to_move, board_to_tensor, legal_moves_mask, eval_material, PIECE_VALUES, OPENINGS
import os
from datetime import datetime
from Visualize import visualize_game_ascii
import time
from torch.utils.tensorboard import SummaryWriter
from time import perf_counter
from typing import Optional

# Define Policy Networks
device = "cuda" if torch.cuda.is_available() else "cpu"

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

def generate_self_play_batch(actor_critic_net, batch_size=8, gamma= 0.99, gae_lamb=0.95, device="cpu"):
    """
    Runs multiple self-play games in parallel and collects trajectories.
    Returns:
        - log_probs: list of log probabilities
        - state_values: list of state values
        - returns: list of GAE returns
        - entropies: list of entropies
    """
    all_log_probs = []
    all_state_values = []
    all_returns = []
    all_entropies = []

    for _ in range(batch_size):
        white_traj, white_rewards, black_traj, black_rewards, game_moves = train_actor_critic_game(actor_critic_net, device=device)
        if not white_traj or not black_traj:
            continue  # skip broken games

        # White
        white_values = [v.squeeze().item() for (_, v, _) in white_traj]
        white_returns = calculate_gae_returns(white_rewards, white_values, gamma=gamma, lam=gae_lamb)
        for (log_prob, value, entropy), R in zip(white_traj, white_returns):
            all_log_probs.append(log_prob)
            all_state_values.append(value)
            all_returns.append(R)
            all_entropies.append(entropy)

        # Black
        black_values = [v.squeeze().item() for (_, v, _) in black_traj]
        black_returns = calculate_gae_returns(black_rewards, black_values, gamma=gamma, lam=gae_lamb)
        for (log_prob, value, entropy), R in zip(black_traj, black_returns):
            all_log_probs.append(log_prob)
            all_state_values.append(value)
            all_returns.append(R)
            all_entropies.append(entropy)

    return all_log_probs, all_state_values, all_returns, all_entropies

def _load_random_opponent_model(models_dir: str = "models"):
    """Load a random model checkpoint from `models_dir` into a fresh ActorCriticResNet.
    Returns the loaded network or None if no file is found or loading fails.
    """
    try:
        files = [f for f in os.listdir(models_dir) if f.endswith(".pt")]
        if not files:
            return None
        path = os.path.join(models_dir, random.choice(files))
        print(f"Opponent selected: {path}")
        net = ActorCriticResNet()
        net.load_state_dict(torch.load(path, map_location="cpu"))
        net.eval()
        return net
    except Exception as e:
        print(f"Failed to load opponent model from {models_dir}: {e}")
        return None

def _play_policy_vs_opponent_game(actor_critic_net,
                                  opponent_mode: str = "random",
                                  opponent_net=None,
                                  models_dir: str = "models",
                                  device="cpu",
                                  opponent_temperature: Optional[float] = None):
    """
    Play one game where the given policy plays against an opponent.
    Collect only the policy's (log_prob, value, entropy) trajectory and rewards.

    Returns:
        policy_traj: list[(log_prob, value, entropy)] for policy turns only.
        policy_rewards: list[float] immediate rewards for policy turns.
        game_moves: list[chess.Move] moves played in the game.
    """
    board = chess.Board()

    # Randomize policy color for variety
    policy_is_white = bool(random.getrandbits(1))

    policy_traj = []
    policy_rewards = []
    game_moves = []

    while not board.is_game_over():
        policy_to_move = ((board.turn == chess.WHITE and policy_is_white) or
                          (board.turn == chess.BLACK and not policy_is_white))

        if policy_to_move:
            # POLICY selects a move (sampled as in training)
            state = board_to_tensor(board).unsqueeze(0).to(device)
            policy_logits, state_value = actor_critic_net(state)

            mask = legal_moves_mask(board)
            if mask.sum() == 0:
                print("No legal moves detected for policy; aborting game.")
                break

            probs = torch.softmax(policy_logits[0].masked_fill(mask == 0, -1e9), dim=0)
            dist = torch.distributions.Categorical(probs)
            move_idx = dist.sample()
            log_prob = dist.log_prob(move_idx)
            entropy = dist.entropy()

            move = index_to_move(move_idx.item(), board)
            if move is None or move not in board.legal_moves:
                move = random.choice(list(board.legal_moves))

            # Immediate reward shaping (same as self-play)
            immediate_reward = -0.01
            if board.is_capture(move):
                if board.is_en_passant(move):
                    captured_sq = chess.square(
                        chess.square_file(move.to_square),
                        chess.square_rank(move.from_square)
                    )
                else:
                    captured_sq = move.to_square
                captured_piece = board.piece_at(captured_sq)
                if captured_piece is not None:
                    immediate_reward += PIECE_VALUES.get(captured_piece.piece_type, 0.0) / 20

            policy_traj.append((log_prob, state_value, entropy))
            policy_rewards.append(immediate_reward)

            board.push(move)
            game_moves.append(move)
        else:
            # OPPONENT move: either random move bot or a greedy move from a random model
            move = None
            if opponent_mode.lower() in ("model", "models", "random_model"):
                opp = opponent_net if opponent_net is not None else _load_random_opponent_model(models_dir)
                if opp is not None:
                    with torch.no_grad():
                        state = board_to_tensor(board).unsqueeze(0).to(device)
                        logits, _ = opp(state)
                        move = pick_move_topk_value(
                            board,
                            opp,
                            logits[0],
                            k=20,
                            device=device,
                            temperature=opponent_temperature,
                        )

            # Fallback to random-move bot
            if move is None or move not in board.legal_moves:
                move = random.choice(list(board.legal_moves))

            board.push(move)
            game_moves.append(move)

    # Final outcome-based reward added to the last policy step
    result = board.result()
    if result == "1-0":
        final_policy_reward = 2.5 if policy_is_white else -1
    elif result == "0-1":
        final_policy_reward = 2.5 if not policy_is_white else -1
    else:  # Draw
        final_policy_reward = -0.5
        outcome = board.outcome()
        if outcome is not None:
            if outcome.termination == chess.Termination.FIVEFOLD_REPETITION:
                final_policy_reward -= 0.25
            elif outcome.termination == chess.Termination.SEVENTYFIVE_MOVES:
                final_policy_reward -= 0.5

    if policy_rewards:
        policy_rewards[-1] += final_policy_reward

    return policy_traj, policy_rewards, game_moves

def generate_opponent_play_batch(actor_critic_net,
                                 batch_size: int = 8,
                                 gamma: float = 0.99,
                                 gae_lamb: float = 0.95,
                                 opponent: str = "random",
                                 models_dir: str = "models",
                                 opponent_temperature: Optional[float] = 1.0):
    """
    Run multiple games where the policy plays an external opponent and collect trajectories
    for the POLICY ONLY. Output matches generate_self_play_batch: lists of log_probs, values,
    returns (GAE), entropies — but only for policy turns.
    If opponent == 'model', one random model from models_dir is loaded and reused for the batch.
    opponent_temperature controls stochasticity of the model opponent's move selection.
    Pass None or a non-positive value to keep the previous deterministic top-k choice.
    """
    all_log_probs = []
    all_state_values = []
    all_returns = []
    all_entropies = []

    opponent_net = None
    if opponent.lower() in ("model", "models", "random_model"):
        opponent_net = _load_random_opponent_model(models_dir)

    for _ in range(batch_size):
        policy_traj, policy_rewards, _ = _play_policy_vs_opponent_game(
            actor_critic_net,
            opponent_mode=opponent,
            opponent_net=opponent_net,
            models_dir=models_dir,
            device=device,
            opponent_temperature=opponent_temperature,
        )

        if not policy_traj:
            continue

        values = [v.squeeze().item() for (_, v, _) in policy_traj]
        gae_returns = calculate_gae_returns(policy_rewards, values, gamma=gamma, lam=gae_lamb)

        for (log_prob, value, entropy), R in zip(policy_traj, gae_returns):
            all_log_probs.append(log_prob)
            all_state_values.append(value)
            all_returns.append(R)
            all_entropies.append(entropy)

    return all_log_probs, all_state_values, all_returns, all_entropies

def train_actor_critic_game(actor_critic_net, opening_prob=0.66, device="cpu"):
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

    game_moves = []
    if  random.random() < opening_prob:
        name, san_line = random.choice(list(OPENINGS.items()))
        n = random.randint(2, len(san_line))  # random truncation
        for san in san_line[:n]:
            board.push_san(san)
            game_moves.append(board.peek().uci())
    
    # Trajectories store data needed for loss calculation
    white_trajectory = []  # Stores (state, log_prob, state_value)
    black_trajectory = []
    white_rewards = []
    black_rewards = []
    game_moves = []

    while not board.is_game_over():
        # Get state, and pass it through the network
        state = board_to_tensor(board).unsqueeze(0).to(device)
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

        """MATERIAL ADVANTAGE ADDED TO REWARD, DOESNT SEEM TO WORK VERY WELL
        # Calculate immediate reward for the move by looking ahead one step
        board.push(move)
        material_advantage = eval_material(board)
        board.pop() # Revert the board to its original state before storing trajectory

        # The reward is from the perspective of the player who made the move
        if board.turn == chess.WHITE:
            immediate_reward = material_advantage * 0.025
        else:
            immediate_reward = -material_advantage * 0.025
        """

        # Base per-move penalty
        immediate_reward = -0.025

        # Add piece value to reward if this move captures a piece (incl. en passant)
        if board.is_capture(move):
            if board.is_en_passant(move):
                # Captured pawn sits on the from-rank and to-file
                captured_sq = chess.square(
                    chess.square_file(move.to_square),
                    chess.square_rank(move.from_square)
                )
            else:
                captured_sq = move.to_square

            captured_piece = board.piece_at(captured_sq)
            if captured_piece is not None:
                immediate_reward += PIECE_VALUES.get(captured_piece.piece_type, 0.0)/20
        
        #Store the trajectory data for the current player
        if board.turn == chess.WHITE:
            white_trajectory.append((log_prob, state_value, entropy))
            white_rewards.append(immediate_reward)
        else:
            black_trajectory.append((log_prob, state_value, entropy))
            black_rewards.append(immediate_reward)

        board.push(move)
        game_moves.append(move)

    #Determine final game outcome and assign terminal rewards
    result = board.result()
    if result == "1-0":
        final_white_reward = 2.5
        final_black_reward = -1
    elif result == "0-1":
        final_white_reward = -1
        final_black_reward = 2.5
    else:  # Draw
        final_white_reward = -.5
        final_black_reward = -.5
        outcome = board.outcome()
        if outcome is not None:
            if outcome.termination == chess.Termination.FIVEFOLD_REPETITION:
                final_black_reward -= .25
                final_white_reward -= .25
            elif outcome.termination == chess.Termination.SEVENTYFIVE_MOVES:
                final_white_reward -= .5
                final_black_reward -= .5

    #Add the final game outcome reward to the last move's reward
    if white_rewards:
        white_rewards[-1] += final_white_reward
    if black_rewards:
        black_rewards[-1] += final_black_reward
    
    return white_trajectory, white_rewards, black_trajectory, black_rewards, game_moves

@torch.no_grad()
def pick_move_topk_value(board, actor_critic_net, logits, k=5, device="cpu", temperature: Optional[float] = None):
    # Mask illegal moves
    mask = legal_moves_mask(board)
    if mask.sum().item() == 0:
        return None

    masked_logits = logits.masked_fill(mask == 0, -1e9)
    k_eff = min(k, int(mask.sum().item()))
    topk = torch.topk(masked_logits, k=k_eff)

    candidate_scores = []
    candidate_moves = []

    for idx in topk.indices.tolist():
        move = index_to_move(idx, board)
        if move is None or move not in board.legal_moves:
            continue

        board.push(move)
        # Terminal handling (aligns with your reward scale)
        if board.is_game_over():
            result = board.result()
            mover_is_white = not board.turn  # after push, it flipped
            if result == "1-0":
                score = 2.5 if mover_is_white else -1.0
            elif result == "0-1":
                score = 2.5 if not mover_is_white else -1.0
            else:
                score = 0
        else:
            # Value for opponent → negate for current player
            state_next = board_to_tensor(board).unsqueeze(0).to(device)
            _, v_next = actor_critic_net(state_next)
            score = -v_next.item()
        board.pop()

        candidate_moves.append(move)
        candidate_scores.append(score)

    if candidate_moves:
        if temperature is not None and temperature > 1e-6 and len(candidate_moves) > 1:
            scores_tensor = torch.tensor(candidate_scores, dtype=torch.float32)
            probs = torch.softmax(scores_tensor / float(temperature), dim=0)
            choice = torch.distributions.Categorical(probs=probs).sample().item()
            best_move = candidate_moves[choice]
        else:
            best_idx = max(range(len(candidate_scores)), key=candidate_scores.__getitem__)
            best_move = candidate_moves[best_idx]
    else:
        best_move = None

    # Fallback if needed
    if best_move is None:
        probs = torch.softmax(masked_logits, dim=0)
        idx = torch.argmax(probs).item()
        best_move = index_to_move(idx, board)
        if best_move is None or best_move not in board.legal_moves:
            best_move = next(iter(board.legal_moves))
    return best_move

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

def calculate_gae_returns(rewards, values, gamma=0.99, lam=0.95):
    """
    Calculates GAE(lambda) returns (a.k.a. lambda-returns) for a single trajectory.
    Args:
        rewards (list[float]): per-step rewards for a trajectory (same player turns only).
        values (list[float]): value estimates V(s_t) for each step in the trajectory.
        gamma (float): discount factor.
        lam (float): GAE lambda parameter.
    Returns:
        list[torch.Tensor]: lambda-returns for each step, to be used as value targets.
    """
    T = len(rewards)
    assert len(values) == T #rewards and values must have same length
    gae = 0.0
    returns = [None] * T
    for t in reversed(range(T)):
        v_t = values[t]
        next_v = values[t + 1] if t + 1 < T else 0.0  # terminal bootstrap
        delta = rewards[t] + gamma * next_v - v_t
        gae = delta + gamma * lam * gae
        returns[t] = torch.tensor(v_t + gae, dtype=torch.float32)
    return returns

### Main Actor-Critic Training Function
def train_actor_critic(actor_critic_net, model_name, optimizer, device="cpu", num_batches=1000, eval_interval=500, gamma=0.99, gae_lamb=0.95, critic_loss_weight=0.5, entropy_weight=0.025, batch_size=8, writer=None, opponent_ratio=0.0, opponent_temperature: Optional[float] = 1.0):
    """
    Main training loop for the Actor-Critic model.

    Args:
        actor_critic_net: The ActorCriticConvNet model.
        model_name (str): The name for saving logs and checkpoints.
        optimizer: The PyTorch optimizer.
        num_batches (int): Number of self-play batches to train on.
        eval_interval (int): Interval (in batches) for evaluation/checkpointing.
        gamma (float): The discount factor for future rewards.
        critic_loss_weight (float): The weight to apply to the critic's loss.
        batch_size (int): Number of games to simulate per batch.
        opponent_ratio (float): Probability [0,1] that a given batch is played
            against a random/old-policy opponent instead of pure self-play.
            If any checkpoints exist in `models/`, a random one is used;
            otherwise a random-move bot is used. 0.0 = only self-play.
        opponent_temperature (float or None): Temperature used when sampling the
            model opponent's moves. None or <= 0 keeps deterministic selection.
    """
    actor_critic_net.to(device)
    for batch in range(num_batches):
        t0 = perf_counter()
        # Decide batch type: self-play or opponent-play.
        use_opponent = random.random() < max(0.0, min(1.0, float(opponent_ratio)))

        if use_opponent:
            opponent_mode = "random"
            try:
                # Prefer model opponents if checkpoints present
                if any(f.endswith(".pt") for f in os.listdir("models")):
                    opponent_mode = "model"
            except Exception:
                opponent_mode = "random"
            # One-line batch mode trace
            print(f"[Batch {batch+1}] Opponent batch: {opponent_mode}")
            log_probs, state_values, returns, entropies = generate_opponent_play_batch(
                actor_critic_net,
                batch_size=batch_size,
                gamma=gamma,
                gae_lamb=gae_lamb,
                opponent=opponent_mode,
                models_dir="models",
                opponent_temperature=opponent_temperature,
            )
        else:
            # One-line batch mode trace
            print(f"[Batch {batch+1}] Self-play batch")
            log_probs, state_values, returns, entropies = generate_self_play_batch(
                actor_critic_net,
                batch_size=batch_size,
                gamma=gamma,
                gae_lamb=gae_lamb,
                device=device,
            )
        t1 = perf_counter()

        if not log_probs:
            print(f"Game {batch+1}: Skipped due to empty batch.")
            continue

        log_probs_tensor = torch.stack(log_probs).to(device)
        # Ensure 1D even when there's a single sample
        state_values_tensor = torch.cat(state_values).view(-1).to(device)
        returns_tensor = torch.stack(returns).to(device)

        # v_norm  = (state_values_tensor - returns_tensor.mean()) / returns_tensor.std().clamp_min(1e-6)
        # rt_norm = (returns_tensor - returns_tensor.mean()) / returns_tensor.std().clamp_min(1e-6)

        entropies_tensor = torch.stack(entropies).to(device)

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
            eval_stats = evaluate_vs_random(actor_critic_net, batch, num_games=100, writer=writer, device=device)
            print(f"\n[Eval at game {batch}] Wins: {eval_stats['wins']}, Draws: {eval_stats['draws']}, Losses: {eval_stats['losses']}\n")
            torch.save(actor_critic_net.state_dict(), f"{model_name}_checkpoint_{batch}.pt")

def evaluate_vs_random(actor_critic_net, game_num, num_games=100, writer=None, show_progress=True, device="cpu"):
    """
    Evaluates the actor-critic model's policy against a random opponent.

    Args:
        actor_critic_net: The trained ActorCriticConvNet model.
        num_games (int): The number of games to play for the evaluation.

    Returns:
        A dictionary containing evaluation statistics.
    """
    entropies = []
    # Set the network to evaluation mode, # This is important!
    actor_critic_net.eval() 
    
    results = []
    white_wins, black_wins, ties, policy_white_wins, policy_black_wins = 0, 0, 0, 0, 0
    game_lengths = []

    # Disable gradients for performance, as they are not needed for evaluation
    with torch.no_grad():
        start_t = perf_counter()
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
                    state = board_to_tensor(board).unsqueeze(0).to(device)
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
                    
                    move = pick_move_topk_value(board, actor_critic_net, logits, k=5, device=device)
                    
                    if move is None or move not in board.legal_moves:
                        # Needs more loggins, this should not really happen
                        print("wrong move found, look into this")
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

            # --- Progress bar update ---
            if show_progress:
                done = i + 1
                frac = done / max(1, num_games)
                bar_len = 30
                filled = int(frac * bar_len)
                bar = "=" * filled + "-" * (bar_len - filled)
                elapsed = perf_counter() - start_t
                rate = done / elapsed if elapsed > 0 else 0.0
                eta = (num_games - done) / rate if rate > 0 else 0.0
                print(f"\rEvaluating [{bar}] {done}/{num_games} | {elapsed:5.1f}s elapsed | ETA {eta:5.1f}s", end="", flush=True)

    # Set the network back to training mode
    actor_critic_net.train()

    if show_progress:
        # End the progress line cleanly
        print()

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

def evaluateVsStockfish():
    pass

if __name__ == "__main__":
    actor_critic_net = ActorCriticResNet().to(device)    
    model_name = 'actor_critic_chess_resnet_v3'
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
        
    # Split parameters into policy (actor) and value (critic) heads
    actor_params = list(actor_critic_net.policy_head.parameters())
    critic_params = list(actor_critic_net.value_head.parameters())
    shared_params = list(actor_critic_net.stem.parameters()) + list(actor_critic_net.residual_tower.parameters())

    # Typically: critic LR about 2× actor LR
    lr_actor = 6e-4
    lr_critic = 3e-4

    optimizer = torch.optim.AdamW([
        {"params": shared_params, "lr": lr_actor},   # shared trunk → follow actor pace
        {"params": actor_params,  "lr": lr_actor},
        {"params": critic_params, "lr": lr_critic},
    ], betas=(0.9, 0.999), weight_decay=0.0)

    writer = SummaryWriter(log_dir=f"runs/{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    train_actor_critic(
        actor_critic_net=actor_critic_net,
        model_name=model_name,
        optimizer=optimizer,
        device=device,
        gamma=0.992,
        gae_lamb=0.9,
        num_batches=2000,
        eval_interval=400,
        entropy_weight=0.05,
        critic_loss_weight=0.5,
        writer=writer,
        batch_size=8,
        opponent_ratio=0.4,
        opponent_temperature=5,
    )

    print(f"Training finished. Saving final model to: {model_filename}")
    torch.save(actor_critic_net.state_dict(), model_filename)