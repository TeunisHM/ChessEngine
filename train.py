import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import chess
import chess.engine
import math
import random
import os
from contextlib import contextmanager
from datetime import datetime
from typing import Optional, List
from time import perf_counter

from helper import (
    move_to_index, index_to_move, board_to_tensor, legal_moves_mask,
    eval_material, PIECE_VALUES, OPENINGS,
    mirror_board_tensor_batch, MIRROR_ACTION_PERM,
    random_endgame_board,
)
from Visualize import visualize_game_ascii
from torch.utils.tensorboard import SummaryWriter
from models import ActorCriticResNet, load_actor_critic_state_dict
from search import search_select_move, DEFAULT_SEARCH_DEPTH
from evaluate_vs_random import evaluate_vs_random
from mcts import MCTS

MODELS_DIR = "models"
device = "cuda" if torch.cuda.is_available() else "cpu"
LR_SCHEDULE_MIN_RATIO = 0.1

try:
    inference_mode = torch.inference_mode
except AttributeError:
    inference_mode = torch.no_grad

# ---- Potential-based reward shaping -------------------------------------

def _potential(board: chess.Board, player: chess.Color) -> float:
    """Phi(s) = tanh(material_diff / 10) from the given player's perspective."""
    m = eval_material(board)
    if player == chess.BLACK:
        m = -m
    return math.tanh(m / 10.0)

def _draw_reward(board: chess.Board, player: chess.Color) -> float:
    """Material-aware tiebreaker: penalise draws from winning positions,
    reward draws from losing positions. Small magnitude so final win remains
    the dominant signal.
    """
    m = eval_material(board)
    if player == chess.BLACK:
        m = -m
    # ahead by >=3 -> ~-0.3; behind by >=3 -> ~+0.1 (capped)
    return max(-0.3, min(0.1, -m * 0.05))

def _draw_penalty(board: chess.Board) -> float:
    """Extra penalties for low-signal draw terminations."""
    outcome = board.outcome()
    if outcome is None:
        return 0.0
    if outcome.termination == chess.Termination.FIVEFOLD_REPETITION:
        return -0.1
    if outcome.termination == chess.Termination.SEVENTYFIVE_MOVES:
        return -0.2
    return 0.0

def _final_self_play_rewards(board: chess.Board):
    """Return terminal rewards for both colors using the shared definition."""
    result = board.result() if board.is_game_over() else "*"
    if result == "1-0":
        return 1.0, -1.0
    if result == "0-1":
        return -1.0, 1.0

    penalty = _draw_penalty(board)
    return (
        _draw_reward(board, chess.WHITE) + penalty,
        _draw_reward(board, chess.BLACK) + penalty,
    )

def _final_policy_reward(board: chess.Board,
                         player: chess.Color) -> float:
    """Return terminal reward for a single tracked player."""
    result = board.result() if board.is_game_over() else "*"
    if result == "1-0":
        return 1.0 if player == chess.WHITE else -1.0
    if result == "0-1":
        return 1.0 if player == chess.BLACK else -1.0

    return _draw_reward(board, player) + _draw_penalty(board)


def _warn_skipped_game(kind: str, reason: str):
    print(f"[WARN] Skipping {kind}: {reason}")

# ---- Rollout helpers -----------------------------------------------------

def _make_rollout_step(board, state_tensor, legal_mask, move, old_log_prob, state_value):
    """Rollout step storage; tensors are kept on the current device to avoid
    per-ply H<->D copies."""
    return (
        state_tensor.detach(),
        legal_mask.detach(),
        int(move_to_index(move, board)),
        old_log_prob.detach(),
        state_value.detach().view(-1),
    )

@contextmanager
def _module_eval(module):
    was_training = module.training
    module.eval()
    try:
        yield
    finally:
        if was_training:
            module.train()

def _select_policy_move(board, logits, legal_mask, actor_critic_net, device,
                        temperature: float, search_k: int, search_depth: int,
                        use_search: bool):
    mask = legal_mask.to(device)
    if mask.sum() == 0:
        return None, None, None

    if use_search:
        return search_select_move(
            board=board, actor_critic_net=actor_critic_net, logits=logits,
            device=device, k=search_k, temperature=temperature,
            depth=search_depth, legal_mask=legal_mask,
        )

    masked_logits = logits.masked_fill(~mask, -1e9)
    base_dist = torch.distributions.Categorical(logits=masked_logits)

    if temperature is None or temperature <= 1e-6:
        move_idx = torch.argmax(masked_logits)
    else:
        sample_logits = masked_logits / float(temperature)
        sample_dist = torch.distributions.Categorical(logits=sample_logits)
        move_idx = sample_dist.sample()

    move = index_to_move(int(move_idx.item()), board)
    if move is None or move not in board.legal_moves:
        return None, None, None

    log_prob = base_dist.log_prob(move_idx)
    entropy = base_dist.entropy()
    return move, log_prob, entropy

# ---- Stockfish pool & helpers -------------------------------------------

class StockfishPool:
    """Simple pool of Stockfish engine handles for parallel/round-robin queries.

    True concurrent play requires threading on top of this; for now this at
    least lets distillation queries and opponent play use separate handles.
    """

    def __init__(self, engine_path: str, size: int = 1):
        self.engines: List[chess.engine.SimpleEngine] = []
        for _ in range(max(1, size)):
            self.engines.append(chess.engine.SimpleEngine.popen_uci(engine_path))
        self._rr = 0

    def get(self) -> chess.engine.SimpleEngine:
        eng = self.engines[self._rr]
        self._rr = (self._rr + 1) % len(self.engines)
        return eng

    def configure_all(self, options: dict):
        for e in self.engines:
            try:
                e.configure(options)
            except Exception as exc:
                print(f"[WARN] Could not configure engine: {exc}")

    def close(self):
        for e in self.engines:
            try:
                e.quit()
            except Exception:
                pass

def _stockfish_score_tanh(engine: chess.engine.SimpleEngine, board: chess.Board,
                          move_time: float) -> Optional[float]:
    """Query Stockfish for centipawn eval, return tanh(cp/600) from side-to-move perspective."""
    try:
        info = engine.analyse(board, chess.engine.Limit(time=move_time))
        score = info["score"].pov(board.turn)
        cp = score.score(mate_score=10000)
        if cp is None:
            return None
        return math.tanh(cp / 600.0)
    except Exception as exc:
        if not getattr(_stockfish_score_tanh, "_warned", False):
            print(f"[WARN] Stockfish distillation query failed: {exc}")
            _stockfish_score_tanh._warned = True
        return None


def _eval_policy_log_probs_and_values(actor_critic_net,
                                      states: torch.Tensor,
                                      legal_masks: torch.Tensor,
                                      actions: torch.Tensor,
                                      minibatch_size: int,
                                      amp_enabled: bool = False):
    """Snapshot old-policy log-probs/values in eval mode.

    Rollouts are collected with the network in eval mode. Recomputing the PPO
    baseline here keeps BatchNorm behavior identical and gives mirrored samples
    their own old log-prob/value instead of reusing the original position's.
    """
    was_training = actor_critic_net.training
    actor_critic_net.eval()
    old_log_probs = []
    old_values = []
    try:
        with torch.no_grad():
            n = states.shape[0]
            mb_size = max(1, min(int(minibatch_size), n))
            for start in range(0, n, mb_size):
                end = min(start + mb_size, n)
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    logits, values = actor_critic_net(states[start:end])
                    masked_logits = logits.masked_fill(~legal_masks[start:end], -1e9)
                    dist = torch.distributions.Categorical(logits=masked_logits)
                    old_log_probs.append(dist.log_prob(actions[start:end]).float())
                    old_values.append(values.view(-1).float())
    finally:
        if was_training:
            actor_critic_net.train()

    return torch.cat(old_log_probs, dim=0), torch.cat(old_values, dim=0)

# ---- Start position (openings / tablebase) ------------------------------

def _start_position(opening_prob: float, tablebase_prob: float):
    """Return (board, game_moves) using optional endgame or opening start."""
    tablebase_prob = max(0.0, min(1.0, float(tablebase_prob)))
    opening_prob = max(0.0, min(1.0 - tablebase_prob, float(opening_prob)))
    r = random.random()
    if r < tablebase_prob:
        return random_endgame_board(), []
    board = chess.Board()
    game_moves: List[chess.Move] = []
    if r < tablebase_prob + opening_prob:
        _, san_line = random.choice(list(OPENINGS.items()))
        n = random.randint(2, len(san_line))
        for san in san_line[:n]:
            try:
                board.push_san(san)
                game_moves.append(board.peek())
            except Exception:
                break
    return board, game_moves

# ---- Self-play game ------------------------------------------------------

def train_actor_critic_game(actor_critic_net,
                            opening_prob: float = 0.7,
                            tablebase_prob: float = 0.0,
                            device="cpu",
                            search_temperature: float = 1.0,
                            use_search: bool = True,
                            search_k: int = 3,
                            search_depth: int = DEFAULT_SEARCH_DEPTH,
                            shaping: bool = True,
                            shaping_gamma: float = 0.99,
                            distill_engine: Optional[chess.engine.SimpleEngine] = None,
                            distill_move_time: float = 0.02,
                            distill_prob: float = 0.25):
    """Self-play game with potential-based shaping and optional Stockfish
    value-distillation targets."""
    board, game_moves = _start_position(opening_prob, tablebase_prob)

    white_trajectory, black_trajectory = [], []
    white_rewards, black_rewards = [], []
    white_distill, black_distill = [], []

    phi_prev_white = _potential(board, chess.WHITE)
    phi_prev_black = _potential(board, chess.BLACK)

    while not board.is_game_over():
        state_tensor = board_to_tensor(board).to(device)
        legal_mask = legal_moves_mask(board).to(device)
        with inference_mode():
            state = state_tensor.unsqueeze(0)
            policy_logits, state_value = actor_critic_net(state)
            move, log_prob, _ = _select_policy_move(
                board=board, logits=policy_logits[0], legal_mask=legal_mask,
                actor_critic_net=actor_critic_net, device=device,
                temperature=search_temperature, search_k=search_k,
                search_depth=search_depth, use_search=use_search,
            )

        if move is None or log_prob is None:
            _warn_skipped_game("self-play game", "policy produced no legal move")
            return [], [], [], [], [], [], []

        current_player = board.turn
        rollout_step = _make_rollout_step(board, state_tensor, legal_mask, move, log_prob, state_value)

        distill_target = None
        if distill_engine is not None and random.random() < distill_prob:
            distill_target = _stockfish_score_tanh(distill_engine, board, distill_move_time)

        if current_player == chess.WHITE:
            white_trajectory.append(rollout_step)
            white_rewards.append(0.0)
            white_distill.append(distill_target)
        else:
            black_trajectory.append(rollout_step)
            black_rewards.append(0.0)
            black_distill.append(distill_target)

        board.push(move)
        game_moves.append(move)

        # F(s,s') = gamma * Phi(s') - Phi(s) for the player who just moved.
        if shaping:
            if current_player == chess.WHITE:
                phi_now = _potential(board, chess.WHITE)
                white_rewards[-1] += shaping_gamma * phi_now - phi_prev_white
                phi_prev_white = phi_now
            else:
                phi_now = _potential(board, chess.BLACK)
                black_rewards[-1] += shaping_gamma * phi_now - phi_prev_black
                phi_prev_black = phi_now

    if not board.is_game_over():
        _warn_skipped_game("self-play game", "game ended without a terminal board state")
        return [], [], [], [], [], [], []

    final_white_reward, final_black_reward = _final_self_play_rewards(board)

    if white_rewards:
        white_rewards[-1] += final_white_reward
    if black_rewards:
        black_rewards[-1] += final_black_reward

    return (white_trajectory, white_rewards, white_distill,
            black_trajectory, black_rewards, black_distill,
            game_moves)


# ---- Self-play batch -----------------------------------------------------

def generate_self_play_batch(actor_critic_net,
                             batch_size: int = 8,
                             gamma: float = 0.99,
                             gae_lamb: float = 0.95,
                             device: str = "cpu",
                             opening_prob: float = 0.7,
                             search_temperature: float = 1.0,
                             search_k: int = 3,
                             search_depth: int = DEFAULT_SEARCH_DEPTH,
                             shaping: bool = True,
                             tablebase_prob: float = 0.0,
                             distill_engine: Optional[chess.engine.SimpleEngine] = None,
                             distill_move_time: float = 0.02,
                             distill_prob: float = 0.25):
    """Runs multiple self-play games and collects PPO rollout data."""
    all_states = []
    all_legal_masks = []
    all_actions = []
    all_old_log_probs = []
    all_old_state_values = []
    all_returns = []
    all_distill_targets: List[Optional[float]] = []

    for _ in range(batch_size):
        (white_traj, white_rewards, white_distill,
         black_traj, black_rewards, black_distill, _) = train_actor_critic_game(
            actor_critic_net,
            device=device,
            opening_prob=opening_prob,
            tablebase_prob=tablebase_prob,
            search_temperature=search_temperature,
            use_search=True,
            search_k=search_k,
            search_depth=search_depth,
            shaping=shaping,
            distill_engine=distill_engine,
            distill_move_time=distill_move_time,
            distill_prob=distill_prob,
        )
        if not white_traj and not black_traj:
            continue

        for traj, rewards, distill in ((white_traj, white_rewards, white_distill),
                                       (black_traj, black_rewards, black_distill)):
            if not traj:
                continue
            vals = [float(value.item()) for (_, _, _, _, value) in traj]
            returns = calculate_gae_returns(rewards, vals, gamma=gamma, lam=gae_lamb)
            for (state_tensor, legal_mask, action_idx, old_log_prob, old_value), ret, dt in zip(traj, returns, distill):
                all_states.append(state_tensor)
                all_legal_masks.append(legal_mask)
                all_actions.append(action_idx)
                all_old_log_probs.append(old_log_prob)
                all_old_state_values.append(old_value)
                all_returns.append(ret)
                all_distill_targets.append(dt)

    return (all_states, all_legal_masks, all_actions, all_old_log_probs,
            all_old_state_values, all_returns, all_distill_targets)


# ---- Opponent (previous checkpoint) game --------------------------------

def _load_random_opponent_weights(device: str = "cpu"):
    try:
        files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pt")]
        if not files:
            return None, None
        random.shuffle(files)
        for filename in files:
            path = os.path.join(MODELS_DIR, filename)
            try:
                state = torch.load(path, map_location=device)
                probe_net = ActorCriticResNet().to(device)
                load_actor_critic_state_dict(probe_net, state)
                return state, path
            except Exception as exc:
                print(f"[WARN] Skipping incompatible opponent checkpoint {path}: {exc}")
        return None, None
    except Exception as e:
        print(f"Failed to load opponent model from {MODELS_DIR}: {e}")
        return None, None


def _play_policy_vs_opponent_game(actor_critic_net,
                                  opponent_net,
                                  device="cpu",
                                  opponent_temperature: Optional[float] = None,
                                  opening_prob: float = 0.7,
                                  search_temperature: float = 1.0,
                                  use_search: bool = True,
                                  search_k: int = 3,
                                  search_depth: int = DEFAULT_SEARCH_DEPTH,
                                  shaping: bool = True,
                                  shaping_gamma: float = 0.99,
                                  distill_engine: Optional[chess.engine.SimpleEngine] = None,
                                  distill_move_time: float = 0.02,
                                  distill_prob: float = 0.25,
                                  tablebase_prob: float = 0.0):
    """Policy vs a previous checkpoint opponent. Returns None if the opponent
    net produced an illegal move (we drop the game rather than fall back to
    random)."""
    board, game_moves = _start_position(opening_prob, tablebase_prob)

    policy_is_white = bool(random.getrandbits(1))
    policy_traj: list = []
    policy_rewards: list = []
    policy_distill: list = []
    phi_prev = _potential(board, chess.WHITE if policy_is_white else chess.BLACK)

    while not board.is_game_over():
        policy_to_move = ((board.turn == chess.WHITE and policy_is_white) or
                          (board.turn == chess.BLACK and not policy_is_white))

        if policy_to_move:
            state_tensor = board_to_tensor(board).to(device)
            legal_mask = legal_moves_mask(board).to(device)
            with inference_mode():
                state = state_tensor.unsqueeze(0)
                policy_logits, state_value = actor_critic_net(state)
                move, log_prob, _ = _select_policy_move(
                    board=board, logits=policy_logits[0], legal_mask=legal_mask,
                    actor_critic_net=actor_critic_net, device=device,
                    temperature=search_temperature, search_k=search_k,
                    search_depth=search_depth, use_search=use_search,
                )

            if move is None or log_prob is None:
                _warn_skipped_game("policy-vs-opponent game", "policy produced no legal move")
                return [], [], [], []

            distill_target = None
            if distill_engine is not None and random.random() < distill_prob:
                distill_target = _stockfish_score_tanh(distill_engine, board, distill_move_time)

            policy_traj.append(_make_rollout_step(board, state_tensor, legal_mask, move, log_prob, state_value))
            policy_rewards.append(0.0)
            policy_distill.append(distill_target)

            board.push(move)
            game_moves.append(move)

            if shaping:
                phi_now = _potential(board, chess.WHITE if policy_is_white else chess.BLACK)
                policy_rewards[-1] += shaping_gamma * phi_now - phi_prev
                phi_prev = phi_now
        else:
            with inference_mode():
                state = board_to_tensor(board).unsqueeze(0).to(device)
                logits, _ = opponent_net(state)
                move, _, _ = _select_policy_move(
                    board=board, logits=logits[0],
                    legal_mask=legal_moves_mask(board),
                    actor_critic_net=opponent_net, device=device,
                    temperature=opponent_temperature if opponent_temperature is not None else 1.0,
                    search_k=5, search_depth=1, use_search=False,
                )

            if move is None or move not in board.legal_moves:
                # Opponent produced an illegal move — drop this game rather
                # than train against a random mover (biases toward piece-grabbing).
                _warn_skipped_game("policy-vs-opponent game", "opponent produced an illegal move")
                return [], [], [], []

            board.push(move)
            game_moves.append(move)
            if shaping:
                phi_prev = _potential(board, chess.WHITE if policy_is_white else chess.BLACK)

    if not board.is_game_over():
        _warn_skipped_game("policy-vs-opponent game", "game ended without a terminal board state")
        return [], [], [], []

    final_reward = _final_policy_reward(
        board,
        chess.WHITE if policy_is_white else chess.BLACK,
    )

    if policy_rewards:
        policy_rewards[-1] += final_reward

    return policy_traj, policy_rewards, policy_distill, game_moves


def _play_policy_vs_uci_engine_game(actor_critic_net,
                                    engine: chess.engine.SimpleEngine,
                                    device="cpu",
                                    engine_move_time: float = 0.05,
                                    engine_skill_level: Optional[int] = None,
                                    opening_prob: float = 0.7,
                                    search_temperature: float = 1.0,
                                    use_search: bool = True,
                                    search_k: int = 3,
                                    search_depth: int = DEFAULT_SEARCH_DEPTH,
                                    shaping: bool = True,
                                    shaping_gamma: float = 0.99,
                                    distill_engine: Optional[chess.engine.SimpleEngine] = None,
                                    distill_move_time: float = 0.02,
                                    distill_prob: float = 0.25,
                                    tablebase_prob: float = 0.0):
    board, game_moves = _start_position(opening_prob, tablebase_prob)
    policy_is_white = bool(random.getrandbits(1))

    policy_traj: list = []
    policy_rewards: list = []
    policy_distill: list = []
    phi_prev = _potential(board, chess.WHITE if policy_is_white else chess.BLACK)

    if engine_skill_level is not None:
        try:
            engine.configure({"Skill Level": int(engine_skill_level)})
        except Exception as exc:
            print(f"[WARN] Could not set engine skill level: {exc}")

    while not board.is_game_over():
        policy_to_move = ((board.turn == chess.WHITE and policy_is_white) or
                          (board.turn == chess.BLACK and not policy_is_white))

        if policy_to_move:
            state_tensor = board_to_tensor(board).to(device)
            legal_mask = legal_moves_mask(board).to(device)
            with inference_mode():
                state = state_tensor.unsqueeze(0)
                policy_logits, state_value = actor_critic_net(state)
                move, log_prob, _ = _select_policy_move(
                    board=board, logits=policy_logits[0], legal_mask=legal_mask,
                    actor_critic_net=actor_critic_net, device=device,
                    temperature=search_temperature, search_k=search_k,
                    search_depth=search_depth, use_search=use_search,
                )

            if move is None or log_prob is None:
                _warn_skipped_game("policy-vs-engine game", "policy produced no legal move")
                return [], [], [], []

            distill_target = None
            if distill_engine is not None and random.random() < distill_prob:
                distill_target = _stockfish_score_tanh(distill_engine, board, distill_move_time)

            policy_traj.append(_make_rollout_step(board, state_tensor, legal_mask, move, log_prob, state_value))
            policy_rewards.append(0.0)
            policy_distill.append(distill_target)

            board.push(move)
            game_moves.append(move)

            if shaping:
                phi_now = _potential(board, chess.WHITE if policy_is_white else chess.BLACK)
                policy_rewards[-1] += shaping_gamma * phi_now - phi_prev
                phi_prev = phi_now
        else:
            try:
                limit = chess.engine.Limit(time=max(engine_move_time, 0.01))
                result = engine.play(board, limit=limit)
                move = result.move
            except Exception as exc:
                _warn_skipped_game("policy-vs-engine game", f"engine failed to move: {exc}")
                return [], [], [], []

            if move is None or move not in board.legal_moves:
                _warn_skipped_game("policy-vs-engine game", "engine produced an illegal move")
                return [], [], [], []

            board.push(move)
            game_moves.append(move)
            if shaping:
                phi_prev = _potential(board, chess.WHITE if policy_is_white else chess.BLACK)

    if not board.is_game_over():
        _warn_skipped_game("policy-vs-engine game", "game ended without a terminal board state")
        return [], [], [], []

    final_reward = _final_policy_reward(
        board,
        chess.WHITE if policy_is_white else chess.BLACK,
    )

    if policy_rewards:
        policy_rewards[-1] += final_reward

    return policy_traj, policy_rewards, policy_distill, game_moves


def generate_opponent_play_batch(actor_critic_net,
                                 batch_size: int = 8,
                                 gamma: float = 0.99,
                                 gae_lamb: float = 0.95,
                                 opponent_temperature: Optional[float] = 1.0,
                                 opening_prob: float = 0.7,
                                 search_temperature: float = 1.0,
                                 search_k: int = 3,
                                 search_depth: int = DEFAULT_SEARCH_DEPTH,
                                 shaping: bool = True,
                                 tablebase_prob: float = 0.0,
                                 distill_engine: Optional[chess.engine.SimpleEngine] = None,
                                 distill_move_time: float = 0.02,
                                 distill_prob: float = 0.25,
                                 device="cpu"):
    all_states, all_legal_masks, all_actions = [], [], []
    all_old_log_probs, all_old_state_values, all_returns = [], [], []
    all_distill_targets: List[Optional[float]] = []

    state, path = _load_random_opponent_weights(device=device)
    if state is None:
        # Do NOT fall back to random play — biases toward aggressive trades.
        print("[WARN] No model opponent available; skipping opponent batch.")
        return (all_states, all_legal_masks, all_actions, all_old_log_probs,
                all_old_state_values, all_returns, all_distill_targets)

    opponent_net = ActorCriticResNet().to(device)
    load_actor_critic_state_dict(opponent_net, state)
    opponent_net.eval()
    print(f"Opponent selected: {path}")

    for _ in range(batch_size):
        policy_traj, policy_rewards, policy_distill, _ = _play_policy_vs_opponent_game(
            actor_critic_net, opponent_net=opponent_net, device=device,
            opponent_temperature=opponent_temperature,
            opening_prob=opening_prob,
            search_temperature=search_temperature,
            use_search=True, search_k=search_k, search_depth=search_depth,
            shaping=shaping,
            distill_engine=distill_engine, distill_move_time=distill_move_time,
            distill_prob=distill_prob, tablebase_prob=tablebase_prob,
        )

        if not policy_traj:
            continue

        vals = [float(v.item()) for (_, _, _, _, v) in policy_traj]
        returns = calculate_gae_returns(policy_rewards, vals, gamma=gamma, lam=gae_lamb)
        for (state_tensor, legal_mask, action_idx, old_log_prob, old_value), ret, dt in zip(policy_traj, returns, policy_distill):
            all_states.append(state_tensor)
            all_legal_masks.append(legal_mask)
            all_actions.append(action_idx)
            all_old_log_probs.append(old_log_prob)
            all_old_state_values.append(old_value)
            all_returns.append(ret)
            all_distill_targets.append(dt)

    return (all_states, all_legal_masks, all_actions, all_old_log_probs,
            all_old_state_values, all_returns, all_distill_targets)


def generate_uci_engine_batch(actor_critic_net,
                              engine_pool: StockfishPool,
                              batch_size: int = 4,
                              gamma: float = 0.99,
                              gae_lamb: float = 0.95,
                              engine_move_time: float = 0.05,
                              engine_skill_level: Optional[int] = None,
                              opening_prob: float = 0.7,
                              search_temperature: float = 1.0,
                              search_k: int = 3,
                              search_depth: int = DEFAULT_SEARCH_DEPTH,
                              shaping: bool = True,
                              tablebase_prob: float = 0.0,
                              distill_engine: Optional[chess.engine.SimpleEngine] = None,
                              distill_move_time: float = 0.02,
                              distill_prob: float = 0.25,
                              device="cpu"):
    all_states, all_legal_masks, all_actions = [], [], []
    all_old_log_probs, all_old_state_values, all_returns = [], [], []
    all_distill_targets: List[Optional[float]] = []

    for _ in range(batch_size):
        engine = engine_pool.get()
        policy_traj, policy_rewards, policy_distill, _ = _play_policy_vs_uci_engine_game(
            actor_critic_net, engine=engine, device=device,
            engine_move_time=engine_move_time,
            engine_skill_level=engine_skill_level,
            opening_prob=opening_prob,
            search_temperature=search_temperature,
            use_search=True, search_k=search_k, search_depth=search_depth,
            shaping=shaping,
            distill_engine=distill_engine, distill_move_time=distill_move_time,
            distill_prob=distill_prob, tablebase_prob=tablebase_prob,
        )

        if not policy_traj:
            continue

        vals = [float(v.item()) for (_, _, _, _, v) in policy_traj]
        returns = calculate_gae_returns(policy_rewards, vals, gamma=gamma, lam=gae_lamb)
        for (state_tensor, legal_mask, action_idx, old_log_prob, old_value), ret, dt in zip(policy_traj, returns, policy_distill):
            all_states.append(state_tensor)
            all_legal_masks.append(legal_mask)
            all_actions.append(action_idx)
            all_old_log_probs.append(old_log_prob)
            all_old_state_values.append(old_value)
            all_returns.append(ret)
            all_distill_targets.append(dt)

    return (all_states, all_legal_masks, all_actions, all_old_log_probs,
            all_old_state_values, all_returns, all_distill_targets)


# ---- GAE ----------------------------------------------------------------

def calculate_discounted_returns(rewards, gamma=0.99):
    returns = []
    discounted_return = 0.0
    for r in reversed(rewards):
        discounted_return = r + gamma * discounted_return
        returns.insert(0, discounted_return)
    return torch.tensor(returns, dtype=torch.float32)


def calculate_gae_returns(rewards, values, gamma=0.99, lam=0.95):
    """GAE(lambda) on a single player's own-turn trajectory.

    Note: values[t+1] is used as the bootstrap for values[t]. Both are
    evaluated at the same player's turns, with the opponent's ply between
    them. This is the standard per-player GAE formulation used in most
    self-play PPO implementations.
    """
    T = len(rewards)
    assert len(values) == T
    gae = 0.0
    returns = [None] * T
    for t in reversed(range(T)):
        v_t = values[t]
        next_v = values[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + gamma * next_v - v_t
        gae = delta + gamma * lam * gae
        returns[t] = torch.tensor(v_t + gae, dtype=torch.float32)
    return returns


# ---- Batched parallel self-play rollouts --------------------------------

def generate_self_play_batch_vectorized(actor_critic_net,
                                        batch_size: int = 16,
                                        gamma: float = 0.99,
                                        gae_lamb: float = 0.95,
                                        device: str = "cpu",
                                        search_temperature: float = 1.0,
                                        shaping: bool = True,
                                        shaping_gamma: float = 0.99,
                                        opening_prob: float = 0.7,
                                        tablebase_prob: float = 0.0,
                                        distill_engine: Optional[chess.engine.SimpleEngine] = None,
                                        distill_move_time: float = 0.02,
                                        distill_prob: float = 0.25,
                                        max_plies: int = 600):
    """Play `batch_size` games in lockstep: one forward pass per ply across
    all live games. Massively reduces per-ply CUDA launch overhead.

    This path does not use fixed-depth search (search_select_move adds extra
    per-game NN calls which break the batching win). Use the scalar path for
    search.
    """
    # Initialize all games.
    boards: List[chess.Board] = []
    for _ in range(batch_size):
        b, _ = _start_position(opening_prob, tablebase_prob)
        boards.append(b)

    # Per-game trajectories keyed by player color.
    white_traj = [[] for _ in range(batch_size)]
    black_traj = [[] for _ in range(batch_size)]
    white_rewards = [[] for _ in range(batch_size)]
    black_rewards = [[] for _ in range(batch_size)]
    white_distill = [[] for _ in range(batch_size)]
    black_distill = [[] for _ in range(batch_size)]
    phi_prev_white = [_potential(boards[i], chess.WHITE) for i in range(batch_size)]
    phi_prev_black = [_potential(boards[i], chess.BLACK) for i in range(batch_size)]
    done = [False] * batch_size
    invalid_reasons: List[Optional[str]] = [None] * batch_size

    for ply in range(max_plies):
        live_ids = [i for i in range(batch_size) if not done[i] and not boards[i].is_game_over()]
        if not live_ids:
            break

        states = torch.stack([board_to_tensor(boards[i]) for i in live_ids]).to(device)
        masks = torch.stack([legal_moves_mask(boards[i]) for i in live_ids]).to(device)

        with inference_mode():
            logits, values = actor_critic_net(states)
            masked_logits = logits.masked_fill(~masks, -1e9)
            log_probs_base = F.log_softmax(masked_logits, dim=1)

            if search_temperature is None or search_temperature <= 1e-6:
                move_idxs = masked_logits.argmax(dim=1)
            else:
                sample_logits = masked_logits / float(search_temperature)
                move_idxs = torch.distributions.Categorical(logits=sample_logits).sample()

        move_idxs_cpu = move_idxs.cpu().tolist()

        for k, gid in enumerate(live_ids):
            action_idx = move_idxs_cpu[k]
            board = boards[gid]
            move = index_to_move(action_idx, board)
            if move is None or move not in board.legal_moves:
                # Try fallback: greedy argmax over legal mask.
                fallback_idx = int(masked_logits[k].argmax().item())
                move = index_to_move(fallback_idx, board)
                action_idx = fallback_idx
                if move is None or move not in board.legal_moves:
                    invalid_reasons[gid] = "policy produced no legal move"
                    done[gid] = True
                    continue

            log_prob = log_probs_base[k, action_idx]
            current_player = board.turn
            state_tensor = states[k].detach()
            legal_mask = masks[k].detach()
            state_value = values[k].detach().view(-1)

            step = (state_tensor, legal_mask, int(action_idx), log_prob.detach(), state_value)
            distill_target = None
            if distill_engine is not None and random.random() < distill_prob:
                distill_target = _stockfish_score_tanh(distill_engine, board, distill_move_time)

            if current_player == chess.WHITE:
                white_traj[gid].append(step)
                white_rewards[gid].append(0.0)
                white_distill[gid].append(distill_target)
            else:
                black_traj[gid].append(step)
                black_rewards[gid].append(0.0)
                black_distill[gid].append(distill_target)

            board.push(move)

            if shaping:
                if current_player == chess.WHITE:
                    phi_now = _potential(board, chess.WHITE)
                    white_rewards[gid][-1] += shaping_gamma * phi_now - phi_prev_white[gid]
                    phi_prev_white[gid] = phi_now
                else:
                    phi_now = _potential(board, chess.BLACK)
                    black_rewards[gid][-1] += shaping_gamma * phi_now - phi_prev_black[gid]
                    phi_prev_black[gid] = phi_now

    # Apply final rewards per game.
    all_states, all_legal_masks, all_actions = [], [], []
    all_old_log_probs, all_old_state_values, all_returns = [], [], []
    all_distill_targets: List[Optional[float]] = []

    for gid in range(batch_size):
        board = boards[gid]
        if invalid_reasons[gid] is not None:
            _warn_skipped_game(
                f"vectorized self-play game {gid}",
                invalid_reasons[gid],
            )
            continue
        if not board.is_game_over():
            _warn_skipped_game(
                f"vectorized self-play game {gid}",
                f"game did not finish within max_plies={max_plies}",
            )
            continue
        final_w, final_b = _final_self_play_rewards(board)

        if white_rewards[gid]:
            white_rewards[gid][-1] += final_w
        if black_rewards[gid]:
            black_rewards[gid][-1] += final_b

        for traj, rewards, distill in ((white_traj[gid], white_rewards[gid], white_distill[gid]),
                                       (black_traj[gid], black_rewards[gid], black_distill[gid])):
            if not traj:
                continue
            vals = [float(v.item()) for (_, _, _, _, v) in traj]
            returns = calculate_gae_returns(rewards, vals, gamma=gamma, lam=gae_lamb)
            for (state_tensor, legal_mask, action_idx, old_log_prob, old_value), ret, dt in zip(traj, returns, distill):
                all_states.append(state_tensor)
                all_legal_masks.append(legal_mask)
                all_actions.append(action_idx)
                all_old_log_probs.append(old_log_prob)
                all_old_state_values.append(old_value)
                all_returns.append(ret)
                all_distill_targets.append(dt)

    return (all_states, all_legal_masks, all_actions, all_old_log_probs,
            all_old_state_values, all_returns, all_distill_targets)


# ---- MCTS self-play -----------------------------------------------------

def mcts_self_play_game(actor_critic_net,
                        device: str = "cpu",
                        num_simulations: int = 100,
                        temperature: float = 1.0,
                        temperature_threshold: int = 15,
                        c_puct: float = 1.5,
                        dirichlet_alpha: float = 0.3,
                        dirichlet_frac: float = 0.25,
                        opening_prob: float = 0.0,
                        tablebase_prob: float = 0.0,
                        max_plies: int = 300):
    """Play a single MCTS self-play game. Returns a list of training samples:
    (state_tensor, pi_target, player_turn) and the final outcome (reward from
    white's perspective: +1 white wins, -1 black wins, 0 draw).
    """
    board, _ = _start_position(opening_prob, tablebase_prob)
    tree = MCTS(actor_critic_net, device=device, c_puct=c_puct,
                dirichlet_alpha=dirichlet_alpha, dirichlet_frac=dirichlet_frac)

    samples = []  # list of (state_tensor, pi_target, player)
    ply = 0
    while not board.is_game_over() and ply < max_plies:
        t = temperature if ply < temperature_threshold else 1e-6
        move, pi, _, action = tree.run(
            board, num_simulations=num_simulations, temperature=t,
            add_root_noise=True,
        )
        if move is None or move not in board.legal_moves:
            _warn_skipped_game("MCTS self-play game", "tree produced no legal move")
            return [], "*"
        state_tensor = board_to_tensor(board)
        samples.append((state_tensor, pi, board.turn))
        board.push(move)
        ply += 1

    if not board.is_game_over():
        _warn_skipped_game(
            "MCTS self-play game",
            f"game did not finish within max_plies={max_plies}",
        )
        return [], "*"

    result = board.result() if board.is_game_over() else "*"
    if result == "1-0":
        white_outcome = 1.0
    elif result == "0-1":
        white_outcome = -1.0
    else:
        white_outcome = 0.0

    # Value target per sample: +outcome if this was the side that won from
    # that state, -outcome otherwise.
    enriched = []
    for state_tensor, pi, player in samples:
        v_target = white_outcome if player == chess.WHITE else -white_outcome
        enriched.append((state_tensor, pi, v_target))

    return enriched, result


def generate_mcts_batch(actor_critic_net,
                        batch_games: int = 8,
                        num_simulations: int = 100,
                        temperature: float = 1.0,
                        temperature_threshold: int = 15,
                        c_puct: float = 1.5,
                        dirichlet_alpha: float = 0.3,
                        dirichlet_frac: float = 0.25,
                        opening_prob: float = 0.0,
                        tablebase_prob: float = 0.0,
                        device: str = "cpu"):
    """Generate a batch of MCTS training samples from self-play games."""
    all_samples = []
    for _ in range(batch_games):
        samples, _ = mcts_self_play_game(
            actor_critic_net, device=device,
            num_simulations=num_simulations, temperature=temperature,
            temperature_threshold=temperature_threshold, c_puct=c_puct,
            dirichlet_alpha=dirichlet_alpha, dirichlet_frac=dirichlet_frac,
            opening_prob=opening_prob, tablebase_prob=tablebase_prob,
        )
        all_samples.extend(samples)
    return all_samples


def train_mcts(actor_critic_net,
               model_name: str,
               optimizer: torch.optim.Optimizer,
               device: str = "cpu",
               num_batches: int = 1000,
               batch_games: int = 8,
               num_simulations: int = 100,
               temperature: float = 1.0,
               temperature_threshold: int = 15,
               c_puct: float = 1.5,
               dirichlet_alpha: float = 0.3,
               dirichlet_frac: float = 0.25,
               opening_prob: float = 0.0,
               tablebase_prob_start: float = 0.0,
               tablebase_prob_end: float = 0.0,
               ppo_minibatch_size: int = 256,
               train_epochs_per_batch: int = 2,
               critic_loss_weight: float = 1.0,
               illegal_logit_weight: float = 1e-4,
               eval_interval: int = 100,
               eval_games: int = 50,
               writer: Optional[SummaryWriter] = None):
    """AlphaZero-style training loop: self-play with MCTS, supervised fit on
    visit-count policy targets and game-outcome value targets."""
    actor_critic_net.to(device)
    scheduler = LambdaLR(optimizer, lr_lambda=_cosine_lr_lambda(num_batches, LR_SCHEDULE_MIN_RATIO))
    amp_enabled = bool(device == "cuda" and torch.cuda.is_available())
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    mirror_perm = MIRROR_ACTION_PERM.to(device)

    for batch in range(num_batches):
        t0 = perf_counter()
        progress = batch / max(1, num_batches - 1)
        tablebase_prob_curr = tablebase_prob_start + (tablebase_prob_end - tablebase_prob_start) * progress

        with _module_eval(actor_critic_net):
            samples = generate_mcts_batch(
                actor_critic_net, batch_games=batch_games,
                num_simulations=num_simulations, temperature=temperature,
                temperature_threshold=temperature_threshold, c_puct=c_puct,
                dirichlet_alpha=dirichlet_alpha, dirichlet_frac=dirichlet_frac,
                opening_prob=opening_prob, tablebase_prob=tablebase_prob_curr,
                device=device,
            )
        t1 = perf_counter()

        if not samples:
            print(f"[MCTS Batch {batch+1}] Empty batch, skipping.")
            scheduler.step()
            continue

        states = torch.stack([s[0] for s in samples]).to(device)
        pis = torch.stack([s[1] for s in samples]).to(device)
        values = torch.tensor([s[2] for s in samples], dtype=torch.float32, device=device)

        m_states = mirror_board_tensor_batch(states)
        m_pis = pis[:, mirror_perm]
        states = torch.cat([states, m_states], dim=0)
        pis = torch.cat([pis, m_pis], dim=0)
        values = torch.cat([values, values], dim=0)

        N = states.shape[0]
        mb_size = max(1, min(int(ppo_minibatch_size), N))

        total_loss_sum = policy_loss_sum = value_loss_sum = illegal_loss_sum = 0.0
        update_count = 0
        actor_critic_net.train()

        for _ in range(max(1, train_epochs_per_batch)):
            perm_idx = torch.randperm(N, device=device)
            for start in range(0, N, mb_size):
                end = min(start + mb_size, N)
                mb = perm_idx[start:end]
                mb_states = states[mb]
                mb_pis = pis[mb]
                mb_values = values[mb]
                mb_legal_mask = mb_pis > 0

                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    logits, v_pred = actor_critic_net(mb_states)
                    masked_logits = logits.masked_fill(~mb_legal_mask, -1e9)
                    log_probs = F.log_softmax(masked_logits, dim=1)
                    # Cross-entropy against pi target (mb_pis rows sum to 1).
                    policy_loss = -(mb_pis * log_probs).sum(dim=1).mean()
                    value_loss = F.mse_loss(v_pred.view(-1), mb_values)
                    illegal_loss = (logits.pow(2) * (~mb_legal_mask).float()).sum(dim=1).mean()
                    total_loss = policy_loss + critic_loss_weight * value_loss + illegal_logit_weight * illegal_loss

                optimizer.zero_grad()
                if amp_enabled:
                    scaler.scale(total_loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(actor_critic_net.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor_critic_net.parameters(), max_norm=1.0)
                    optimizer.step()

                total_loss_sum += total_loss.item()
                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
                illegal_loss_sum += float(illegal_loss.item())
                update_count += 1

        scheduler.step()
        t2 = perf_counter()
        denom = max(1, update_count)
        avg_total = total_loss_sum / denom
        avg_policy = policy_loss_sum / denom
        avg_value = value_loss_sum / denom
        avg_illegal = illegal_loss_sum / denom

        if writer is not None:
            writer.add_scalar("MCTS/Total", avg_total, batch)
            writer.add_scalar("MCTS/Policy", avg_policy, batch)
            writer.add_scalar("MCTS/Value", avg_value, batch)
            writer.add_scalar("MCTS/IllegalLogit", avg_illegal, batch)
            writer.add_scalar("MCTS/SamplesPerBatch", N, batch)
            writer.add_scalar("Schedule/LR", optimizer.param_groups[0]["lr"], batch)
            writer.add_scalar("Schedule/TablebaseProb", tablebase_prob_curr, batch)

        print(f"[MCTS Batch {batch+1}] DataGen: {t1 - t0:.2f}s | Train: {t2 - t1:.2f}s | "
              f"Policy: {avg_policy:.4f} Value: {avg_value:.4f} Total: {avg_total:.4f}")

        if (batch + 1) % eval_interval == 0:
            eval_stats = evaluate_vs_random(
                actor_critic_net, batch, num_games=eval_games,
                writer=writer, device=device,
                search_k=3, search_depth=DEFAULT_SEARCH_DEPTH,
            )
            print(f"\n[MCTS Eval at batch {batch}] Wins: {eval_stats['wins']}, "
                  f"Draws: {eval_stats['draws']}, Losses: {eval_stats['losses']}\n")
            torch.save(actor_critic_net.state_dict(), f"{model_name}_mcts_checkpoint_{batch}.pt")


# ---- LR schedule --------------------------------------------------------

def _cosine_lr_lambda(total_steps: int, min_ratio: float = 0.1):
    total = max(1, total_steps)

    def lr_lambda(step: int) -> float:
        frac = min(1.0, step / total)
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * frac))

    return lr_lambda


# ---- Main PPO training loop ---------------------------------------------

def train_actor_critic(actor_critic_net,
                       model_name,
                       optimizer,
                       device="cpu",
                       num_batches=5000,
                       eval_interval=500,
                       eval_games: int = 50,
                       gamma=0.99,
                       gae_lamb=0.95,
                       critic_loss_weight=0.5,
                       entropy_weight=0.025,
                       batch_size=128,
                       writer=None,
                       opponent_ratio=0.0,
                       opponent_temperature: Optional[float] = 1.0,
                       opening_prob: float = 0.7,
                       sampling_temperature: float = 1.2,
                       sampling_temperature_end: Optional[float] = None,
                       search_k: int = 5,
                       search_depth: int = DEFAULT_SEARCH_DEPTH,
                       engine_path: Optional[str] = None,
                       engine_pool_size: int = 1,
                       engine_ratio: float = 0.0,
                       engine_move_time: float = 0.05,
                       engine_skill_level: Optional[int] = None,
                       engine_skill_start: Optional[int] = None,
                       engine_skill_end: Optional[int] = None,
                       entropy_weight_end: Optional[float] = None,
                       ppo_clip_ratio: float = 0.2,
                       ppo_epochs: int = 4,
                       ppo_minibatch_size: int = 256,
                       target_kl: float = 0.015,
                       illegal_logit_weight: float = 1e-4,
                       distill_weight: float = 0.5,
                       distill_prob: float = 0.25,
                       distill_move_time: float = 0.02,
                       tablebase_prob_start: float = 0.0,
                       tablebase_prob_end: float = 0.0,
                       shaping: bool = True,
                       vectorized_rollouts: bool = False):
    """PPO training loop with: mini-batches, KL early stop, AMP, LR schedule,
    mirror augmentation, illegal-logit auxiliary loss, Stockfish value
    distillation, curriculum Stockfish skill, tablebase curriculum, and
    optional vectorized (lock-step) self-play rollouts."""
    actor_critic_net.to(device)

    engine_pool: Optional[StockfishPool] = None
    distill_engine: Optional[chess.engine.SimpleEngine] = None
    if engine_path is not None:
        try:
            # +1 for a dedicated distillation handle.
            pool = StockfishPool(engine_path, size=max(1, engine_pool_size) + (1 if distill_weight > 0 else 0))
            engine_pool = pool
            if distill_weight > 0:
                distill_engine = pool.engines[-1]
                print(f"[INFO] Loaded UCI engine pool from {engine_path} (size={len(pool.engines)}, distill=yes)")
            else:
                print(f"[INFO] Loaded UCI engine pool from {engine_path} (size={len(pool.engines)})")
        except Exception as exc:
            print(f"[WARN] Could not launch engine at '{engine_path}': {exc}")
            engine_pool = None

    scheduler = LambdaLR(optimizer, lr_lambda=_cosine_lr_lambda(num_batches, LR_SCHEDULE_MIN_RATIO))

    amp_enabled = bool(device == "cuda" and torch.cuda.is_available())
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    mirror_perm = MIRROR_ACTION_PERM.to(device)

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

            # Stockfish skill curriculum
            if engine_skill_start is not None and engine_skill_end is not None:
                skill_curr = int(round(engine_skill_start + (engine_skill_end - engine_skill_start) * progress))
            else:
                skill_curr = engine_skill_level

            tablebase_prob_curr = tablebase_prob_start + (tablebase_prob_end - tablebase_prob_start) * progress

            use_engine = (
                engine_pool is not None
                and random.random() < max(0.0, min(1.0, float(engine_ratio)))
            )
            use_opponent = (
                not use_engine
                and random.random() < max(0.0, min(1.0, float(opponent_ratio)))
            )

            if use_engine:
                print(f"[Batch {batch+1}] UCI engine batch (Stockfish, skill={skill_curr})")
                with _module_eval(actor_critic_net):
                    rollout = generate_uci_engine_batch(
                        actor_critic_net, engine_pool=engine_pool,
                        batch_size=batch_size, gamma=gamma, gae_lamb=gae_lamb,
                        engine_move_time=engine_move_time,
                        engine_skill_level=skill_curr,
                        opening_prob=opening_prob,
                        search_temperature=sampling_temperature_curr,
                        search_k=search_k, search_depth=search_depth,
                        shaping=shaping,
                        tablebase_prob=tablebase_prob_curr,
                        distill_engine=distill_engine,
                        distill_move_time=distill_move_time,
                        distill_prob=distill_prob, device=device,
                    )
            elif use_opponent:
                print(f"[Batch {batch+1}] Opponent batch: model")
                with _module_eval(actor_critic_net):
                    rollout = generate_opponent_play_batch(
                        actor_critic_net, batch_size=batch_size,
                        gamma=gamma, gae_lamb=gae_lamb,
                        opponent_temperature=opponent_temperature,
                        opening_prob=opening_prob,
                        search_temperature=sampling_temperature_curr,
                        search_k=search_k, search_depth=search_depth,
                        shaping=shaping,
                        tablebase_prob=tablebase_prob_curr,
                        distill_engine=distill_engine,
                        distill_move_time=distill_move_time,
                        distill_prob=distill_prob, device=device,
                    )
            else:
                if vectorized_rollouts:
                    print(f"[Batch {batch+1}] Self-play batch (vectorized)")
                    with _module_eval(actor_critic_net):
                        rollout = generate_self_play_batch_vectorized(
                            actor_critic_net, batch_size=batch_size,
                            gamma=gamma, gae_lamb=gae_lamb, device=device,
                            search_temperature=sampling_temperature_curr,
                            shaping=shaping,
                            opening_prob=opening_prob,
                            tablebase_prob=tablebase_prob_curr,
                            distill_engine=distill_engine,
                            distill_move_time=distill_move_time,
                            distill_prob=distill_prob,
                        )
                else:
                    print(f"[Batch {batch+1}] Self-play batch")
                    with _module_eval(actor_critic_net):
                        rollout = generate_self_play_batch(
                            actor_critic_net, batch_size=batch_size,
                            gamma=gamma, gae_lamb=gae_lamb, device=device,
                            opening_prob=opening_prob,
                            search_temperature=sampling_temperature_curr,
                            search_k=search_k, search_depth=search_depth,
                            shaping=shaping,
                            tablebase_prob=tablebase_prob_curr,
                            distill_engine=distill_engine,
                            distill_move_time=distill_move_time,
                            distill_prob=distill_prob,
                        )

            (states, legal_masks, actions, old_log_probs, old_state_values,
             returns, distill_targets) = rollout
            t1 = perf_counter()

            if not actions:
                print(f"Batch {batch+1}: Skipped due to empty batch.")
                scheduler.step()
                continue

            actor_critic_net.train()
            states_tensor = torch.stack(states).to(device)
            legal_masks_tensor = torch.stack(legal_masks).to(device)
            actions_tensor = torch.tensor(actions, dtype=torch.long, device=device)
            returns_tensor = torch.stack(returns).view(-1).to(device)

            # Distillation targets: tensor of target values; mask for where we have one.
            distill_vals = torch.zeros(len(distill_targets), dtype=torch.float32, device=device)
            distill_mask = torch.zeros(len(distill_targets), dtype=torch.bool, device=device)
            for i, dt in enumerate(distill_targets):
                if dt is not None:
                    distill_vals[i] = float(dt)
                    distill_mask[i] = True
            raw_distill_target_count = int(distill_mask.sum().item())

            # Mirror augmentation is always on to bake board symmetry into training.
            m_states = mirror_board_tensor_batch(states_tensor)
            m_masks = legal_masks_tensor[:, mirror_perm]
            m_actions = mirror_perm[actions_tensor]
            states_tensor = torch.cat([states_tensor, m_states], dim=0)
            legal_masks_tensor = torch.cat([legal_masks_tensor, m_masks], dim=0)
            actions_tensor = torch.cat([actions_tensor, m_actions], dim=0)
            returns_tensor = torch.cat([returns_tensor, returns_tensor], dim=0)
            distill_vals = torch.cat([distill_vals, distill_vals], dim=0)
            distill_mask = torch.cat([distill_mask, distill_mask], dim=0)

            old_log_probs_tensor, old_state_values_tensor = _eval_policy_log_probs_and_values(
                actor_critic_net,
                states_tensor,
                legal_masks_tensor,
                actions_tensor,
                minibatch_size=ppo_minibatch_size,
                amp_enabled=amp_enabled,
            )

            advantages = returns_tensor - old_state_values_tensor
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

            N = states_tensor.shape[0]
            mb_size = max(1, min(int(ppo_minibatch_size), N))

            total_loss_sum = actor_loss_sum = critic_loss_sum = 0.0
            entropy_loss_sum = distill_loss_sum = illegal_loss_sum = 0.0
            approx_kl_sum = clip_frac_sum = 0.0
            update_count = 0
            early_stopped_at: Optional[int] = None

            actor_critic_net.eval()
            for epoch in range(max(1, int(ppo_epochs))):
                perm_idx = torch.randperm(N, device=device)
                epoch_kl_sum = 0.0
                epoch_steps = 0
                for start in range(0, N, mb_size):
                    end = min(start + mb_size, N)
                    mb = perm_idx[start:end]

                    mb_states = states_tensor[mb]
                    mb_masks = legal_masks_tensor[mb]
                    mb_actions = actions_tensor[mb]
                    mb_old_lp = old_log_probs_tensor[mb]
                    mb_advantages = advantages[mb]
                    mb_returns = returns_tensor[mb]
                    mb_distill_vals = distill_vals[mb]
                    mb_distill_mask = distill_mask[mb]

                    with torch.amp.autocast("cuda", enabled=amp_enabled):
                        policy_logits_batch, state_values_batch = actor_critic_net(mb_states)
                        masked_logits = policy_logits_batch.masked_fill(~mb_masks, -1e9)
                        dist = torch.distributions.Categorical(logits=masked_logits)
                        new_log_probs = dist.log_prob(mb_actions)
                        entropies = dist.entropy()

                        ratios = torch.exp(new_log_probs - mb_old_lp)
                        clipped = torch.clamp(ratios, 1.0 - ppo_clip_ratio, 1.0 + ppo_clip_ratio)
                        actor_loss = -torch.min(ratios * mb_advantages, clipped * mb_advantages).mean()
                        critic_loss = F.mse_loss(state_values_batch.view(-1), mb_returns)
                        entropy_loss = -entropies.mean()

                        # Illegal-logit aux loss — penalise magnitude at illegal positions.
                        illegal_loss = (policy_logits_batch.pow(2) * (~mb_masks).float()).sum(dim=1).mean()

                        # Stockfish distillation loss on subset where we have targets.
                        if mb_distill_mask.any():
                            pred = state_values_batch.view(-1)[mb_distill_mask]
                            tgt = mb_distill_vals[mb_distill_mask]
                            distill_loss = F.mse_loss(pred, tgt)
                        else:
                            distill_loss = torch.zeros((), device=device)

                        total_loss = (
                            actor_loss
                            + critic_loss_weight * critic_loss
                            + entropy_weight_curr * entropy_loss
                            + illegal_logit_weight * illegal_loss
                            + distill_weight * distill_loss
                        )

                    optimizer.zero_grad()
                    if amp_enabled:
                        scaler.scale(total_loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(actor_critic_net.parameters(), max_norm=0.5)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        total_loss.backward()
                        torch.nn.utils.clip_grad_norm_(actor_critic_net.parameters(), max_norm=0.5)
                        optimizer.step()

                    with torch.no_grad():
                        approx_kl_mb = (mb_old_lp - new_log_probs).mean().item()
                        clip_frac_mb = ((ratios - 1.0).abs() > ppo_clip_ratio).float().mean().item()

                    total_loss_sum += total_loss.item()
                    actor_loss_sum += actor_loss.item()
                    critic_loss_sum += critic_loss.item()
                    entropy_loss_sum += entropy_loss.item()
                    distill_loss_sum += float(distill_loss.item())
                    illegal_loss_sum += float(illegal_loss.item())
                    approx_kl_sum += approx_kl_mb
                    clip_frac_sum += clip_frac_mb
                    update_count += 1
                    epoch_kl_sum += approx_kl_mb
                    epoch_steps += 1

                # KL early stop between epochs.
                avg_epoch_kl = epoch_kl_sum / max(1, epoch_steps)
                if avg_epoch_kl > 1.5 * target_kl:
                    early_stopped_at = epoch + 1
                    break

            scheduler.step()
            actor_critic_net.train()
            denom = max(1, update_count)
            avg_total_loss = total_loss_sum / denom
            avg_actor_loss = actor_loss_sum / denom
            avg_critic_loss = critic_loss_sum / denom
            avg_entropy_loss = entropy_loss_sum / denom
            avg_distill_loss = distill_loss_sum / denom
            avg_illegal_loss = illegal_loss_sum / denom
            avg_approx_kl = approx_kl_sum / denom
            avg_clip_frac = clip_frac_sum / denom
            t2 = perf_counter()

            if writer is not None:
                writer.add_scalar("Loss/Total", avg_total_loss, batch)
                writer.add_scalar("Loss/Actor", avg_actor_loss, batch)
                writer.add_scalar("Loss/Critic", avg_critic_loss, batch)
                writer.add_scalar("Loss/Entropy", avg_entropy_loss, batch)
                writer.add_scalar("Loss/Distill", avg_distill_loss, batch)
                writer.add_scalar("Distill/Targets", raw_distill_target_count, batch)
                writer.add_scalar("Loss/IllegalLogit", avg_illegal_loss, batch)
                writer.add_scalar("Schedule/EntropyWeight", entropy_weight_curr, batch)
                writer.add_scalar("Schedule/SamplingTemperature", sampling_temperature_curr, batch)
                writer.add_scalar("Schedule/LR", optimizer.param_groups[0]["lr"], batch)
                writer.add_scalar("Schedule/TablebaseProb", tablebase_prob_curr, batch)
                if skill_curr is not None:
                    writer.add_scalar("Schedule/EngineSkill", float(skill_curr), batch)
                writer.add_scalar("PPO/ApproxKL", avg_approx_kl, batch)
                writer.add_scalar("PPO/ClipFraction", avg_clip_frac, batch)
                writer.add_scalar("PPO/Updates", update_count, batch)
                if early_stopped_at is not None:
                    writer.add_scalar("PPO/EarlyStopEpoch", early_stopped_at, batch)

            suffix = f" | EarlyStop@epoch={early_stopped_at}" if early_stopped_at else ""
            print(f"[Batch {batch+1}] DataGen: {t1 - t0:.2f}s | Train: {t2 - t1:.2f}s | Total: {t2 - t0:.2f}s{suffix}")

            if (batch + 1) % 10 == 0:
                print(
                    f"Batch {batch+1}: Total: {avg_total_loss:.4f}, Actor: {avg_actor_loss:.4f}, "
                    f"Critic: {avg_critic_loss:.4f}, Entropy: {avg_entropy_loss:.4f}, "
                    f"Distill: {avg_distill_loss:.4f}, KL: {avg_approx_kl:.4f}, "
                    f"ClipFrac: {avg_clip_frac:.4f}, DistillTargets: {raw_distill_target_count}"
                )
                if writer is not None:
                    writer.add_histogram("critic/state_values", old_state_values_tensor, batch)
                    writer.add_histogram("critic/returns", returns_tensor, batch)
                    writer.add_histogram("critic/advantages", advantages, batch)

            if (batch + 1) % eval_interval == 0:
                eval_stats = evaluate_vs_random(
                    actor_critic_net, batch, num_games=eval_games,
                    writer=writer, device=device,
                    search_k=search_k, search_depth=search_depth,
                )
                print(f"\n[Eval at game {batch}] Wins: {eval_stats['wins']}, "
                      f"Draws: {eval_stats['draws']}, Losses: {eval_stats['losses']}\n")
                torch.save(actor_critic_net.state_dict(), f"{MODELS_DIR}/{model_name}_checkpoint_{batch}.pt")
    finally:
        if engine_pool is not None:
            engine_pool.close()

train_ppo = train_actor_critic

if __name__ == "__main__":
    actor_critic_net = ActorCriticResNet().to(device)
    model_name = 'net4'
    model_filename = model_name + ".pt"

    if os.path.exists(model_filename):
        print(f"Loading pre-trained model from: {model_filename}")
        try:
            load_actor_critic_state_dict(actor_critic_net, torch.load(model_filename, map_location=device))
        except Exception as e:
            print(f"[WARN] Old checkpoint incompatible with new architecture ({e}); initializing fresh weights.")
    else:
        print("Initializing a new model.")

    actor_params = list(actor_critic_net.policy_head.parameters())
    critic_params = list(actor_critic_net.value_head.parameters())
    shared_params = list(actor_critic_net.stem.parameters()) + list(actor_critic_net.residual_tower.parameters())

    lr_actor = 1.5e-6
    lr_critic = 2e-6

    optimizer = torch.optim.AdamW([
        {"params": shared_params, "lr": lr_actor},
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
        num_batches=995,
        eval_interval=99,
        eval_games=100,
        entropy_weight=0.02,
        critic_loss_weight=0.5,
        writer=writer,
        batch_size=32,
        opponent_ratio=0.2,
        opponent_temperature=1.3,
        opening_prob=0.7,
        sampling_temperature=1.2,
        engine_path="./stockfish/stockfish",
        engine_pool_size=1,
        engine_ratio=0.2,
        engine_move_time=0.01,
        engine_skill_start=0,
        engine_skill_end=0,
        sampling_temperature_end=0.8,
        entropy_weight_end=0.001,
        ppo_clip_ratio=0.1,
        ppo_epochs=4,
        ppo_minibatch_size=600,
        target_kl=0.015,
        illegal_logit_weight=1e-4,
        distill_weight=0.5,
        distill_prob=0.25,
        distill_move_time=0.02,
        tablebase_prob_start=0.05,
        tablebase_prob_end=0.1,
        shaping=True,
        search_k=3,
        search_depth=DEFAULT_SEARCH_DEPTH,
        vectorized_rollouts = True,
    )

    print(f"Training finished. Saving final model to: {model_filename}")
    torch.save(actor_critic_net.state_dict(), model_filename)
