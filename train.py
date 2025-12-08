import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import chess
import chess.engine
import random
from helper import move_to_index, index_to_move, board_to_tensor, legal_moves_mask, eval_material, PIECE_VALUES, OPENINGS
import os
from datetime import datetime
from Visualize import visualize_game_ascii
from torch.utils.tensorboard import SummaryWriter
from time import perf_counter
from typing import Optional
from models import ActorCriticResNet
from search import search_select_move, DEFAULT_SEARCH_DEPTH
from evaluate_vs_random import evaluate_vs_random

MODELS_DIR = "search_models"

# Define Policy Networks
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    inference_mode = torch.inference_mode
except AttributeError:
    inference_mode = torch.no_grad

def generate_self_play_batch(actor_critic_net,
                             batch_size=8,
                             gamma=0.99,
                             gae_lamb=0.95,
                             device="cpu",
                             search_temperature: float = 1.0,
                             search_k: int = 3,
                             search_depth: int = DEFAULT_SEARCH_DEPTH):
    """
    Runs multiple self-play games in parallel and collects trajectories.
    Args:
        temperature (float): Softmax temperature applied to policy logits before sampling.
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
        white_traj, white_rewards, black_traj, black_rewards, game_moves = train_actor_critic_game(
            actor_critic_net,
            device=device,
            search_temperature=search_temperature,
            search_k=search_k,
            search_depth=search_depth,
        )
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

def train_actor_critic_game(actor_critic_net,
                            opening_prob=0.7,
                            device="cpu",
                            search_temperature: float = 1.0,
                            search_k: int = 3,
                            search_depth: int = DEFAULT_SEARCH_DEPTH):
    """
    Plays a single game of self-play using the actor-critic network,
    collecting data for training.
    """
    board = chess.Board()

    game_moves = []
    if random.random() < opening_prob:
        name, san_line = random.choice(list(OPENINGS.items()))
        n = random.randint(2, len(san_line))
        for san in san_line[:n]:
            board.push_san(san)
            game_moves.append(board.peek().uci())

    white_trajectory = []
    black_trajectory = []
    white_rewards = []
    black_rewards = []
    game_moves = []

    while not board.is_game_over():
        state = board_to_tensor(board).unsqueeze(0).to(device)
        policy_logits, state_value = actor_critic_net(state)

        move, log_prob, entropy = search_select_move(
            board=board,
            actor_critic_net=actor_critic_net,
            logits=policy_logits[0],
            device=device,
            k=search_k,
            temperature=search_temperature,
            depth=search_depth,
        )

        if move is None or log_prob is None or entropy is None:
            print("No legal moves detected, aborting game.")
            break

        immediate_reward = 0.0

        if board.turn == chess.WHITE:
            white_trajectory.append((log_prob, state_value, entropy))
            white_rewards.append(immediate_reward)
        else:
            black_trajectory.append((log_prob, state_value, entropy))
            black_rewards.append(immediate_reward)

        board.push(move)
        game_moves.append(move)

    result = board.result()
    if result == "1-0":
        final_white_reward = 1
        final_black_reward = -1
    elif result == "0-1":
        final_white_reward = -1
        final_black_reward = 1
    else:
        final_white_reward = 0
        final_black_reward = 0
        outcome = board.outcome()
        if outcome is not None:
            if outcome.termination == chess.Termination.FIVEFOLD_REPETITION:
                final_black_reward -= 0.1
                final_white_reward -= 0.1
            elif outcome.termination == chess.Termination.SEVENTYFIVE_MOVES:
                final_white_reward -= 0.2
                final_black_reward -= 0.2

    if white_rewards:
        white_rewards[-1] += final_white_reward
    if black_rewards:
        black_rewards[-1] += final_black_reward

    return white_trajectory, white_rewards, black_trajectory, black_rewards, game_moves

def _load_random_opponent_weights(device: str = "cpu"):
    """Return (state_dict, path) for a random opponent checkpoint, or (None, None) on failure."""
    try:
        files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pt")]
        if not files:
            return None, None
        path = os.path.join(MODELS_DIR, random.choice(files))
        state = torch.load(path, map_location=device)
        return state, path
    except Exception as e:
        print(f"Failed to load opponent model from {MODELS_DIR}: {e}")
        return None, None

def _play_policy_vs_opponent_game(actor_critic_net,
                                  opponent_net=None,
                                  device="cpu",
                                  opponent_temperature: Optional[float] = None,
                                  search_temperature: float = 1.0,
                                  search_k: int = 3,
                                  search_depth: int = DEFAULT_SEARCH_DEPTH):
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

            move, log_prob, entropy = search_select_move(
                board=board,
                actor_critic_net=actor_critic_net,
                logits=policy_logits[0],
                device=device,
                k=search_k,
                temperature=search_temperature,
                depth=search_depth,
            )
            if move is None or log_prob is None or entropy is None:
                print("No legal moves detected for policy; aborting game.")
                break

            # No shaping: only terminal outcome rewards are used
            immediate_reward = 0.0

            policy_traj.append((log_prob, state_value, entropy))
            policy_rewards.append(immediate_reward)

            board.push(move)
            game_moves.append(move)
        else:
            # OPPONENT move: model if available, otherwise random
            move = None
            if opponent_net is not None:
                with inference_mode():
                    state = board_to_tensor(board).unsqueeze(0).to(device)
                    logits, _ = opponent_net(state)
                    move, _, _ = search_select_move(
                        board=board,
                        actor_critic_net=opponent_net,
                        logits=logits[0],
                        device=device,
                        k=5,
                        temperature=opponent_temperature if opponent_temperature is not None else 1.0,
                        depth=1,
                    )

            if move is None or move not in board.legal_moves:
                move = random.choice(list(board.legal_moves))

            board.push(move)
            game_moves.append(move)

    # Final outcome-based reward added to the last policy step
    result = board.result()
    if result == "1-0":
        final_policy_reward = 1 if policy_is_white else -1
    elif result == "0-1":
        final_policy_reward = 1 if not policy_is_white else -1
    else:  # Draw
        final_policy_reward = 0
        outcome = board.outcome()
        if outcome is not None:
            if outcome.termination == chess.Termination.FIVEFOLD_REPETITION:
                final_policy_reward -= 0.1
            elif outcome.termination == chess.Termination.SEVENTYFIVE_MOVES:
                final_policy_reward -= 0.2

    if policy_rewards:
        policy_rewards[-1] += final_policy_reward

    return policy_traj, policy_rewards, game_moves

def _play_policy_vs_uci_engine_game(actor_critic_net,
                                    engine: chess.engine.SimpleEngine,
                                    device="cpu",
                                    engine_move_time: float = 0.05,
                                    engine_skill_level: Optional[int] = None,
                                    search_temperature: float = 1.0,
                                    search_k: int = 3,
                                    search_depth: int = DEFAULT_SEARCH_DEPTH):
    """
    Play one game where the policy plays a UCI engine (e.g., Stockfish).
    Collect only the policy's (log_prob, value, entropy) trajectory and rewards.
    """
    board = chess.Board()
    policy_is_white = bool(random.getrandbits(1))

    policy_traj = []
    policy_rewards = []
    game_moves = []

    if engine_skill_level is not None:
        try:
            engine.configure({"Skill Level": int(engine_skill_level)})
        except Exception as exc:
            print(f"[WARN] Could not set engine skill level: {exc}")

    while not board.is_game_over():
        policy_to_move = ((board.turn == chess.WHITE and policy_is_white) or
                          (board.turn == chess.BLACK and not policy_is_white))

        if policy_to_move:
            state = board_to_tensor(board).unsqueeze(0).to(device)
            policy_logits, state_value = actor_critic_net(state)

            move, log_prob, entropy = search_select_move(
                board=board,
                actor_critic_net=actor_critic_net,
                logits=policy_logits[0],
                device=device,
                k=search_k,
                temperature=search_temperature,
                depth=search_depth,
            )
            if move is None or log_prob is None or entropy is None:
                print("No legal moves detected for policy; aborting game.")
                break

            immediate_reward = 0.0
            policy_traj.append((log_prob, state_value, entropy))
            policy_rewards.append(immediate_reward)

            board.push(move)
            game_moves.append(move)
        else:
            try:
                limit = chess.engine.Limit(time=max(engine_move_time, 0.01))
                result = engine.play(board, limit=limit)
                move = result.move
            except Exception as exc:
                print(f"[WARN] Engine failed to move: {exc}")
                move = random.choice(list(board.legal_moves))

            if move not in board.legal_moves:
                move = random.choice(list(board.legal_moves))

            board.push(move)
            game_moves.append(move)

    result = board.result()
    if result == "1-0":
        final_policy_reward = 1 if policy_is_white else -1
    elif result == "0-1":
        final_policy_reward = 1 if not policy_is_white else -1
    else:
        final_policy_reward = 0
        outcome = board.outcome()
        if outcome is not None:
            if outcome.termination == chess.Termination.FIVEFOLD_REPETITION:
                final_policy_reward -= 0.1
            elif outcome.termination == chess.Termination.SEVENTYFIVE_MOVES:
                final_policy_reward -= 0.2

    if policy_rewards:
        policy_rewards[-1] += final_policy_reward

    return policy_traj, policy_rewards, game_moves

def generate_opponent_play_batch(actor_critic_net,
                                 batch_size: int = 8,
                                 gamma: float = 0.99,
                                 gae_lamb: float = 0.95,
                                 opponent_temperature: Optional[float] = 1.0,
                                 search_temperature: float = 1.0,
                                 search_k: int = 3,
                                 search_depth: int = DEFAULT_SEARCH_DEPTH):
    """
    Run multiple games where the policy plays an external opponent and collect trajectories
    for the POLICY ONLY. Output matches generate_self_play_batch: lists of log_probs, values,
    returns (GAE), entropies — but only for policy turns.
    A single network is instantiated once per batch and its weights are loaded
    from a randomly chosen checkpoint for the whole batch. No new models are created per game.
    opponent_temperature controls stochasticity of the model opponent's move selection.
    search_temperature controls softmax over candidate scores during search selection.
    Pass None or a non-positive value to keep the previous deterministic top-k choice.
    """
    all_log_probs = []
    all_state_values = []
    all_returns = []
    all_entropies = []

    opponent_net = None
    state, path = _load_random_opponent_weights(device=device)
    if state is not None:
        opponent_net = ActorCriticResNet().to(device)
        opponent_net.load_state_dict(state)
        opponent_net.eval()
        print(f"Opponent selected: {path}")
    else:
        print("[WARN] No model opponent available; falling back to random moves.")

    for _ in range(batch_size):
        policy_traj, policy_rewards, _ = _play_policy_vs_opponent_game(
            actor_critic_net,
            opponent_net=opponent_net,
            device=device,
            opponent_temperature=opponent_temperature,
            search_temperature=search_temperature,
            search_k=search_k,
            search_depth=search_depth,
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

def generate_uci_engine_batch(actor_critic_net,
                              engine: chess.engine.SimpleEngine,
                              batch_size: int = 4,
                              gamma: float = 0.99,
                              gae_lamb: float = 0.95,
                              engine_move_time: float = 0.05,
                              engine_skill_level: Optional[int] = None,
                              search_temperature: float = 1.0,
                              search_k: int = 3,
                              search_depth: int = DEFAULT_SEARCH_DEPTH,
                              device="cpu"):
    """
    Run multiple games where the policy plays a UCI engine opponent and collect trajectories
    for the POLICY ONLY. The engine instance is provided by the caller to avoid per-batch launches.
    Output matches generate_self_play_batch: lists of log_probs, values, returns (GAE), entropies.
    """
    all_log_probs = []
    all_state_values = []
    all_returns = []
    all_entropies = []

    for _ in range(batch_size):
        policy_traj, policy_rewards, _ = _play_policy_vs_uci_engine_game(
            actor_critic_net,
            engine=engine,
            device=device,
            engine_move_time=engine_move_time,
            engine_skill_level=engine_skill_level,
            search_temperature=search_temperature,
            search_k=search_k,
            search_depth=search_depth,
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
def train_actor_critic(actor_critic_net,
                       model_name,
                       optimizer,
                       device="cpu",
                       num_batches=5000,
                       eval_interval=500,
                       gamma=0.99,
                       gae_lamb=0.95,
                       critic_loss_weight=0.5,
                       entropy_weight=0.025,
                       batch_size=128,
                       writer=None,
                       opponent_ratio=0.0,
                       opponent_temperature: Optional[float] = 1.0,
                       sampling_temperature: float = 1.2,
                       sampling_temperature_end: Optional[float] = None,
                       search_k: int = 3,
                       search_depth: int = DEFAULT_SEARCH_DEPTH,
                       engine_path: Optional[str] = None,
                       engine_ratio: float = 0.0,
                       engine_move_time: float = 0.05,
                       engine_skill_level: Optional[int] = None,
                       entropy_weight_end: Optional[float] = None):
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
        sampling_temperature (float): Temperature applied to policy sampling during training (start).
        sampling_temperature_end (float or None): Optional end temperature for linear annealing.
        search_k (int): Number of top policy moves to explore in the lightweight search selector.
        search_depth (int): Fixed search depth (plies) for the lightweight selector (>=1).
        engine_path (str or None): Path to a UCI engine (e.g., Stockfish) for occasional opponent batches.
        engine_ratio (float): Probability [0,1] that a batch uses the UCI engine opponent instead of other modes.
        engine_move_time (float): Seconds allowed per engine move during training batches.
        engine_skill_level (int or None): Optional engine skill level; passed directly to engine.configure.
        entropy_weight (float): Entropy regularization weight (start).
        entropy_weight_end (float or None): Optional end weight for linear annealing.
    """
    actor_critic_net.to(device)

    engine = None
    if engine_path is not None:
        try:
            engine = chess.engine.SimpleEngine.popen_uci(engine_path)
            print(f"[INFO] Loaded UCI engine from {engine_path}")
        except Exception as exc:
            print(f"[WARN] Could not launch engine at '{engine_path}': {exc}")
            engine = None

    try:
        for batch in range(num_batches):
            t0 = perf_counter()
            progress = batch / max(1, num_batches - 1)
            entropy_weight_curr = (
                entropy_weight if entropy_weight_end is None
                else entropy_weight + (entropy_weight_end - entropy_weight) * progress
            )
            sampling_temperature_curr = (
                sampling_temperature if sampling_temperature_end is None
                else sampling_temperature + (sampling_temperature_end - sampling_temperature) * progress
            )
            # Decide batch type: self-play, opponent-play, or UCI engine-play.
            use_engine = (
                engine is not None
                and random.random() < max(0.0, min(1.0, float(engine_ratio)))
            )
            use_opponent = (
                not use_engine
                and random.random() < max(0.0, min(1.0, float(opponent_ratio)))
            )

            if use_engine:
                print(f"[Batch {batch+1}] UCI engine batch (Stockfish)")
                log_probs, state_values, returns, entropies = generate_uci_engine_batch(
                    actor_critic_net,
                    engine=engine,
                    batch_size=batch_size,
                    gamma=gamma,
                    gae_lamb=gae_lamb,
                    engine_move_time=engine_move_time,
                    engine_skill_level=engine_skill_level,
                    search_temperature=sampling_temperature_curr,
                    search_k=search_k,
                    search_depth=search_depth,
                    device=device,
                )
            elif use_opponent:
                print(f"[Batch {batch+1}] Opponent batch: model")
                log_probs, state_values, returns, entropies = generate_opponent_play_batch(
                    actor_critic_net,
                    batch_size=batch_size,
                    gamma=gamma,
                    gae_lamb=gae_lamb,
                    opponent_temperature=opponent_temperature,
                    search_temperature=sampling_temperature_curr,
                    search_k=search_k,
                    search_depth=search_depth,
                )
            else:
                print(f"[Batch {batch+1}] Self-play batch")
                log_probs, state_values, returns, entropies = generate_self_play_batch(
                    actor_critic_net,
                    batch_size=batch_size,
                    gamma=gamma,
                    gae_lamb=gae_lamb,
                    device=device,
                    search_temperature=sampling_temperature_curr,
                    search_k=search_k,
                    search_depth=search_depth,
                )
            t1 = perf_counter()

            if not log_probs:
                print(f"Game {batch+1}: Skipped due to empty batch.")
                continue

            log_probs_tensor = torch.stack(log_probs).to(device)

            state_values_tensor = torch.cat(state_values).view(-1).to(device) # Ensure 1D even when there's a single sample
            returns_tensor = torch.stack(returns).to(device)
            entropies_tensor = torch.stack(entropies).to(device)

            # Advantage normalization
            advantages = returns_tensor - state_values_tensor.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            actor_loss = -(log_probs_tensor * advantages).mean()
            critic_loss = F.mse_loss(state_values_tensor, returns_tensor)
            entropy_loss = -entropies_tensor.mean()
            total_loss = actor_loss + critic_loss_weight * critic_loss + entropy_weight_curr * entropy_loss

            writer.add_scalar("Loss/Total", total_loss.item(), batch)
            writer.add_scalar("Loss/Actor", actor_loss.item(), batch)
            writer.add_scalar("Loss/Critic", critic_loss.item(), batch)
            writer.add_scalar("Loss/Entropy", entropy_loss.item(), batch)
            writer.add_scalar("Schedule/EntropyWeight", entropy_weight_curr, batch)
            writer.add_scalar("Schedule/SamplingTemperature", sampling_temperature_curr, batch)

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
                eval_stats = evaluate_vs_random(actor_critic_net, batch, num_games=100, writer=writer, device=device, search_k=search_k, search_depth=search_depth)
                print(f"\n[Eval at game {batch}] Wins: {eval_stats['wins']}, Draws: {eval_stats['draws']}, Losses: {eval_stats['losses']}\n")
                torch.save(actor_critic_net.state_dict(), f"{model_name}_checkpoint_{batch}.pt")
    finally:
        if engine is not None:
            try:
                engine.quit()
            except Exception:
                pass

if __name__ == "__main__":
    actor_critic_net = ActorCriticResNet().to(device)    
    model_name = 'search_net'
    model_filename = model_name + ".pt"

    # Load pre-trained weights if they exist
    if os.path.exists(model_filename):
        print(f"Loading pre-trained model from: {model_filename}")
        actor_critic_net.load_state_dict(torch.load(model_filename))
    else:
        print("Initializing a new model.")
        
    # Split parameters into policy (actor) and value (critic) heads
    actor_params = list(actor_critic_net.policy_head.parameters())
    critic_params = list(actor_critic_net.value_head.parameters())
    shared_params = list(actor_critic_net.stem.parameters()) + list(actor_critic_net.residual_tower.parameters())

    # Typically: critic LR about 2× actor LR
    lr_actor = 1e-4
    lr_critic = 2e-4

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
        gamma=0.99,
        gae_lamb=0.9,
        num_batches=1000,
        eval_interval=100,
        entropy_weight=0.025,
        critic_loss_weight=0.5,
        writer=writer,
        batch_size=64,
        opponent_ratio=0.4,
        opponent_temperature=1.3,
        sampling_temperature=1.5,
        engine_path= "./stockfish/stockfish",
        engine_ratio=0.4,
        engine_move_time=0.01,
        engine_skill_level=0,
        sampling_temperature_end=0.8,
        entropy_weight_end=0.001,
        search_k = 5,
        search_depth = DEFAULT_SEARCH_DEPTH
    )

    print(f"Training finished. Saving final model to: {model_filename}")
    torch.save(actor_critic_net.state_dict(), model_filename)