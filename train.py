import argparse
import csv
import math
import os
import random
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from queue import Queue
from time import perf_counter
from typing import Callable, List, Optional

import chess
import chess.engine
import chess.syzygy
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from helper import (
    OPENINGS,
    MIRROR_ACTION_PERM,
    index_to_move,
    mirror_board_tensor_batch,
)
from lookahead import select_moves_from_policy, select_moves_with_lookahead
from models import (
    ActorCriticResNet,
    DEFAULT_NUM_FILTERS,
    DEFAULT_NUM_RESIDUAL_BLOCKS,
    load_actor_critic_state_dict,
    net_from_state_dict,
)
from evaluate_vs_random import evaluate_vs_random

MODELS_DIR = "models"

PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
}


def _material_balance(board: chess.Board, color: chess.Color) -> float:
    """Own material minus opponent's, in pawn-equivalent units."""
    own = opp = 0
    for piece in board.piece_map().values():
        v = PIECE_VALUES[piece.piece_type]
        if piece.color == color:
            own += v
        else:
            opp += v
    return float(own - opp)


def _dtz_progress_potential(
    tablebase: Optional[chess.syzygy.Tablebase], board: chess.Board, color: chess.Color
) -> Optional[float]:
    """Bounded progress potential inside a clean <=5-man tablebase win for `color`.

    None outside the tablebase domain, on a draw/loss, or on a cursed win (DTZ
    magnitude > 100, a 50-move-rule edge case) — shaping only applies while
    color has an unconditional, table-confirmed win. WDL/DTZ are reported from
    the mover's perspective, so both are negated when color is not to move.
    Returns -|dtz|/100 in [-1, 0], rising toward 0 as the forced conversion
    (zeroing move) gets closer.
    """
    if tablebase is None or chess.popcount(board.occupied) > 5:
        return None
    try:
        wdl = tablebase.probe_wdl(board)
    except (chess.syzygy.MissingTableError, KeyError):
        return None
    wdl_color = wdl if board.turn == color else -wdl
    if wdl_color != 2:
        return None
    try:
        dtz = tablebase.probe_dtz(board)
    except (chess.syzygy.MissingTableError, KeyError):
        return None
    dtz_color = dtz if board.turn == color else -dtz
    if abs(dtz_color) > 100:
        return None
    return -abs(dtz_color) / 100.0


def _tb_value_target(
    tablebase: Optional[chess.syzygy.Tablebase], board: chess.Board, color: chess.Color
) -> Optional[float]:
    """Ground-truth value target for `color` from the <=5-man tablebase, or
    None outside its domain / on a probe miss.

    Unlike _dtz_progress_potential this is defined on every in-domain
    position (win, draw, or loss), for use as a value-head-only auxiliary
    supervision target — kept out of the reward stream so it corrects the
    critic's baseline without directly rewarding the actor. Cursed win /
    blessed loss (50-move-rule edge cases) collapse to 0, same conservative
    mapping as the WDL bootstrap termination.
    """
    if tablebase is None or chess.popcount(board.occupied) > 5:
        return None
    try:
        wdl = tablebase.probe_wdl(board)
    except (chess.syzygy.MissingTableError, KeyError):
        return None
    wdl_color = wdl if board.turn == color else -wdl
    if wdl_color >= 2:
        return 1.0
    if wdl_color <= -2:
        return -1.0
    return 0.0


def _start_position(opening_prob: float) -> chess.Board:
    """Return a fresh board, optionally seeded with a random opening line."""
    board = chess.Board()
    if random.random() < max(0.0, min(1.0, opening_prob)):
        _, san_line = random.choice(list(OPENINGS.items()))
        n = random.randint(2, len(san_line))
        for san in san_line[:n]:
            try:
                board.push_san(san)
            except Exception:
                break
    return board


def _terminal_rewards(board: chess.Board, draw_penalty: float = 0.1):
    """Win/loss/draw signal from each color's perspective.

    Draw penalty makes draws strictly worse than ongoing-but-winnable play, so the
    sparse terminal signal pushes toward decisive games rather than shuffling.
    """
    result = board.result() if board.is_game_over() else "*"
    if result == "1-0":
        return 1.25, -1.0
    if result == "0-1":
        return -1.0, 1.25
    return -draw_penalty, -draw_penalty


def calculate_gae_returns(rewards, values, gamma=0.99, lam=0.95):
    """GAE(lambda) over a single player's own-turn trajectory."""
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


# ---- Opponent move providers --------------------------------------------

OpponentMoveFn = Callable[[List[chess.Board]], List[Optional[chess.Move]]]


def _checkpoint_opponent_fn(opponent_net, device: str, temperature: float,
                            top_k: int, alpha: float,
                            value_weight: float = 1.0) -> OpponentMoveFn:
    """Vectorized opponent that picks moves with a frozen ActorCriticResNet using
    the same top-k value lookahead as the policy player."""

    def move_fn(boards: List[chess.Board]) -> List[Optional[chess.Move]]:
        idxs, *_ = select_moves_with_lookahead(
            opponent_net, boards, device,
            top_k=top_k, alpha=alpha, temperature=temperature,
            value_weight=value_weight,
        )
        idxs_cpu = idxs.cpu().tolist()
        out: List[Optional[chess.Move]] = []
        for k, board in enumerate(boards):
            move = index_to_move(idxs_cpu[k], board)
            if move is None or move not in board.legal_moves:
                out.append(None)
            else:
                out.append(move)
        return out

    return move_fn


def _load_random_opponent(device: str):
    """Pick a random checkpoint from MODELS_DIR and return it as an eval-mode net."""
    if not os.path.isdir(MODELS_DIR):
        return None, None
    files = [
        filename for filename in os.listdir(MODELS_DIR) if filename.endswith(".pt")
    ]
    if not files:
        return None, None
    random.shuffle(files)
    for filename in files:
        path = os.path.join(MODELS_DIR, filename)
        try:
            state = torch.load(path, map_location=device)
            # Match the checkpoint's head architecture so legacy dense-head
            # opponents keep their trained policy instead of a fresh head.
            net = net_from_state_dict(state, device)
            net.eval()
            return net, path
        except Exception as exc:
            print(f"[WARN] skipping incompatible checkpoint {path}: {exc}")
    return None, None


class EnginePool:
    """Pool of UCI engine processes; play_many dispatches in parallel via threads."""

    def __init__(self, engine_path: str, size: int = 4):
        self.engines: List[chess.engine.SimpleEngine] = [
            chess.engine.SimpleEngine.popen_uci(engine_path) for _ in range(max(1, size))
        ]
        self._free: Queue = Queue()
        for e in self.engines:
            self._free.put(e)
        self._executor = ThreadPoolExecutor(max_workers=len(self.engines))

    def configure_all(self, options: dict):
        for e in self.engines:
            try:
                e.configure(options)
            except Exception as exc:
                print(f"[WARN] engine configure failed: {exc}")

    def _play_one(self, board: chess.Board, move_time: float) -> Optional[chess.Move]:
        e = self._free.get()
        try:
            result = e.play(board, chess.engine.Limit(time=max(move_time, 0.01)))
            return result.move
        except Exception as exc:
            print(f"[WARN] engine.play failed: {exc}")
            return None
        finally:
            self._free.put(e)

    def play_many(self, boards: List[chess.Board], move_time: float) -> List[Optional[chess.Move]]:
        return list(self._executor.map(lambda b: self._play_one(b, move_time), boards))

    def close(self):
        self._executor.shutdown(wait=True)
        for e in self.engines:
            try:
                e.quit()
            except Exception:
                pass


def _engine_opponent_fn(pool: EnginePool, move_time: float) -> OpponentMoveFn:
    def move_fn(boards: List[chess.Board]) -> List[Optional[chess.Move]]:
        return pool.play_many(boards, move_time)

    return move_fn


# ---- Vectorized batch generation ----------------------------------------

def generate_batch(actor_critic_net,
                   batch_size: int = 16,
                   gamma: float = 0.99,
                   gae_lamb: float = 0.95,
                   device: str = "cpu",
                   temperature: float = 1.0,
                   opening_prob: float = 0.7,
                   opponent_move_fn: Optional[OpponentMoveFn] = None,
                   max_plies: int = 600,
                   step_penalty: float = 0.0,
                   draw_penalty: float = 0.1,
                   material_shaping_per_pawn: float = 0.0,
                   dtz_shaping_weight: float = 0.0,
                   tb_value_aux_weight: float = 0.0,
                   tablebase: Optional[chess.syzygy.Tablebase] = None,
                   tablebase_terminate_prob: float = 1.0,
                   trainee_search: bool = False,
                   trainee_top_k: int = 6,
                   trainee_alpha: float = 0.33,
                   trainee_value_weight: float = 1.0,
                   progress_label: str = ""):
    """Lockstep batch of games. One forward pass per ply across all live games.

    If opponent_move_fn is None: pure self-play (both colors are the policy).
    Otherwise: each game's policy color is randomized; opponent_move_fn supplies
    moves for boards on the opponent's turn (called once per ply with the live
    set, so engine/checkpoint moves can be parallelized internally).

    If trainee_search is True the trainee samples moves via the value-lookahead
    search (top-k by π, scored with quiescence). The recorded old_log_prob is
    log b(a|s) — the search behavior policy — so PPO's IS ratio is well-formed.
    """
    boards = [_start_position(opening_prob) for _ in range(batch_size)]
    self_play = opponent_move_fn is None
    policy_is_white = (
        [True] * batch_size if self_play
        else [random.random() < 0.5 for _ in range(batch_size)]
    )

    white_traj = [[] for _ in range(batch_size)]
    black_traj = [[] for _ in range(batch_size)]
    white_rewards = [[] for _ in range(batch_size)]
    black_rewards = [[] for _ in range(batch_size)]
    last_phi_white: List[Optional[float]] = [None] * batch_size
    last_phi_black: List[Optional[float]] = [None] * batch_size
    last_dtz_phi_white: List[Optional[float]] = [None] * batch_size
    last_dtz_phi_black: List[Optional[float]] = [None] * batch_size
    done = [False] * batch_size
    tb_terminate = [random.random() < tablebase_terminate_prob for _ in range(batch_size)]
    shape_on = material_shaping_per_pawn > 0.0
    dtz_shape_on = dtz_shaping_weight > 0.0 and tablebase is not None
    tb_aux_on = tb_value_aux_weight > 0.0 and tablebase is not None

    # π-vs-search rollout diagnostics (only meaningful when trainee_search).
    search_kl_sum = 0.0
    search_disagree_sum = 0
    search_pick_rank_sum = 0.0
    search_delta_logp_sum = 0.0
    search_states_n = 0

    show_progress = bool(progress_label)
    progress_t0 = perf_counter() if show_progress else 0.0

    for ply in range(max_plies):
        live_ids = [i for i in range(batch_size) if not done[i] and not boards[i].is_game_over()]
        if not live_ids:
            break

        if tablebase is not None:
            for gid in list(live_ids):
                if not tb_terminate[gid]:
                    continue
                board = boards[gid]
                if chess.popcount(board.occupied) > 5:
                    continue
                try:
                    wdl = tablebase.probe_wdl(board)
                except (chess.syzygy.MissingTableError, KeyError):
                    continue
                wdl_w = wdl if board.turn == chess.WHITE else -wdl
                if wdl_w >= 2:
                    white_term, black_term = 1.0, -1.0
                elif wdl_w <= -2:
                    white_term, black_term = -1.0, 1.0
                else:
                    white_term, black_term = 0.0, 0.0
                if white_rewards[gid]:
                    white_rewards[gid][-1] += white_term
                if black_rewards[gid]:
                    black_rewards[gid][-1] += black_term
                done[gid] = True
            live_ids = [i for i in live_ids if not done[i]]
            if not live_ids:
                break

        if self_play:
            policy_ids = live_ids
            opponent_ids: List[int] = []
        else:
            policy_ids = [
                i for i in live_ids
                if (boards[i].turn == chess.WHITE) == policy_is_white[i]
            ]
            opp_set = set(live_ids) - set(policy_ids)
            opponent_ids = [i for i in live_ids if i in opp_set]

        if policy_ids:
            pol_boards = [boards[i] for i in policy_ids]
            if trainee_search:
                (move_idxs, log_pi, masks, values, states,
                 log_b_chosen, kl_b_pi_step, pick_rank_step,
                 topk_idx_step, log_b_topk_step) = select_moves_with_lookahead(
                    actor_critic_net, pol_boards, device,
                    top_k=trainee_top_k, alpha=trainee_alpha, temperature=temperature,
                    value_weight=trainee_value_weight,
                )
                # Diagnostics: how much does search disagree with raw π?
                search_kl_sum += float(kl_b_pi_step.sum().item())
                search_disagree_sum += int((pick_rank_step != 0).sum().item())
                search_pick_rank_sum += float(pick_rank_step.float().sum().item())
                log_pi_at_chosen = log_pi.gather(1, move_idxs.view(-1, 1)).view(-1)
                search_delta_logp_sum += float((log_b_chosen - log_pi_at_chosen).sum().item())
                search_states_n += len(pol_boards)
            else:
                move_idxs, log_pi, masks, values, states = select_moves_from_policy(
                    actor_critic_net, pol_boards, device, temperature=temperature,
                )
                log_b_chosen = None
            move_idxs_cpu = move_idxs.cpu().tolist()
            for k, gid in enumerate(policy_ids):
                action_idx = move_idxs_cpu[k]
                board = boards[gid]
                move = index_to_move(action_idx, board)
                if move is None or move not in board.legal_moves:
                    done[gid] = True
                    continue

                if trainee_search:
                    log_prob = log_b_chosen[k].detach()
                else:
                    log_prob = log_pi[k, action_idx].detach()
                current_player = board.turn

                # Cycle-based potential-based material shaping. Φ(s) is computed
                # at the start of color's decision. Shaping for color's previous
                # decision = γ·Φ_now − Φ_at_previous_decision; this is added to
                # the reward of that previous step (closing one full cycle).
                if shape_on:
                    phi_now = _material_balance(board, current_player)
                    if current_player == chess.WHITE:
                        last = last_phi_white[gid]
                        if last is not None and white_rewards[gid]:
                            white_rewards[gid][-1] += material_shaping_per_pawn * (gamma * phi_now - last)
                        last_phi_white[gid] = phi_now
                    else:
                        last = last_phi_black[gid]
                        if last is not None and black_rewards[gid]:
                            black_rewards[gid][-1] += material_shaping_per_pawn * (gamma * phi_now - last)
                        last_phi_black[gid] = phi_now

                # Same cycle pattern, but Φ is only defined inside a clean TB
                # win (see _dtz_progress_potential); undefined on either side
                # of a cycle means skip (don't shape entering/leaving the
                # tablebase domain, only progress *within* it).
                if dtz_shape_on:
                    dtz_phi_now = _dtz_progress_potential(tablebase, board, current_player)
                    if current_player == chess.WHITE:
                        dtz_last = last_dtz_phi_white[gid]
                        if dtz_phi_now is not None and dtz_last is not None and white_rewards[gid]:
                            white_rewards[gid][-1] += dtz_shaping_weight * (gamma * dtz_phi_now - dtz_last)
                        last_dtz_phi_white[gid] = dtz_phi_now
                    else:
                        dtz_last = last_dtz_phi_black[gid]
                        if dtz_phi_now is not None and dtz_last is not None and black_rewards[gid]:
                            black_rewards[gid][-1] += dtz_shaping_weight * (gamma * dtz_phi_now - dtz_last)
                        last_dtz_phi_black[gid] = dtz_phi_now

                tb_target = _tb_value_target(tablebase, board, current_player) if tb_aux_on else None

                step = (
                    states[k].detach(),
                    masks[k].detach(),
                    int(action_idx),
                    log_prob,
                    values[k].detach().view(-1),
                    topk_idx_step[k].detach() if trainee_search else None,
                    log_b_topk_step[k].detach() if trainee_search else None,
                    tb_target,
                )
                if current_player == chess.WHITE:
                    white_traj[gid].append(step)
                    white_rewards[gid].append(-step_penalty)
                else:
                    black_traj[gid].append(step)
                    black_rewards[gid].append(-step_penalty)

                board.push(move)

        if opponent_ids:
            opp_boards = [boards[i] for i in opponent_ids]
            opp_moves = opponent_move_fn(opp_boards)
            for gid, move in zip(opponent_ids, opp_moves):
                board = boards[gid]
                if move is None or move not in board.legal_moves:
                    done[gid] = True
                    continue
                board.push(move)

        if show_progress and (ply + 1) % 10 == 0:
            elapsed = perf_counter() - progress_t0
            print(
                f"\r{progress_label} | rollout ply {ply+1} | "
                f"live {len(live_ids)}/{batch_size} | {elapsed:5.1f}s",
                end="", flush=True,
            )

    if show_progress:
        print("\r" + " " * 100 + "\r", end="", flush=True)

    all_states, all_masks, all_actions = [], [], []
    all_old_log_probs, all_old_values, all_returns = [], [], []
    all_topk_idx, all_log_b_topk = [], []
    all_tb_targets = []

    for gid in range(batch_size):
        board = boards[gid]

        # Close out the cycle for both colors using current-board material so the
        # last decision before terminal also gets shaping credit/debit.
        if shape_on:
            if last_phi_white[gid] is not None and white_rewards[gid]:
                phi_w = _material_balance(board, chess.WHITE)
                white_rewards[gid][-1] += material_shaping_per_pawn * (gamma * phi_w - last_phi_white[gid])
            if last_phi_black[gid] is not None and black_rewards[gid]:
                phi_b = _material_balance(board, chess.BLACK)
                black_rewards[gid][-1] += material_shaping_per_pawn * (gamma * phi_b - last_phi_black[gid])

        if dtz_shape_on:
            if last_dtz_phi_white[gid] is not None and white_rewards[gid]:
                dtz_phi_w = _dtz_progress_potential(tablebase, board, chess.WHITE)
                if dtz_phi_w is not None:
                    white_rewards[gid][-1] += dtz_shaping_weight * (gamma * dtz_phi_w - last_dtz_phi_white[gid])
            if last_dtz_phi_black[gid] is not None and black_rewards[gid]:
                dtz_phi_b = _dtz_progress_potential(tablebase, board, chess.BLACK)
                if dtz_phi_b is not None:
                    black_rewards[gid][-1] += dtz_shaping_weight * (gamma * dtz_phi_b - last_dtz_phi_black[gid])

        if board.is_game_over():
            final_w, final_b = _terminal_rewards(board, draw_penalty=draw_penalty)
            if white_rewards[gid]:
                white_rewards[gid][-1] += final_w
            if black_rewards[gid]:
                black_rewards[gid][-1] += final_b
        # else: max-plies timeout. Keep the per-ply step-penalty rewards and let
        # GAE bootstrap from 0 — discarding the trajectory wasted training data.

        for traj, rewards in ((white_traj[gid], white_rewards[gid]),
                              (black_traj[gid], black_rewards[gid])):
            if not traj:
                continue
            vals = [float(v.item()) for (_, _, _, _, v, _, _, _) in traj]
            returns = calculate_gae_returns(rewards, vals, gamma=gamma, lam=gae_lamb)
            for (state_tensor, mask, action_idx, old_lp, old_v,
                 topk_idx, log_b_topk, tb_target), ret in zip(traj, returns):
                all_states.append(state_tensor)
                all_masks.append(mask)
                all_actions.append(action_idx)
                all_old_log_probs.append(old_lp)
                all_old_values.append(old_v)
                all_returns.append(ret)
                all_topk_idx.append(topk_idx)
                all_log_b_topk.append(log_b_topk)
                all_tb_targets.append(tb_target)

    stats: dict = {}
    if trainee_search and search_states_n > 0:
        n = search_states_n
        stats = {
            "n_search_states": n,
            "mean_kl_b_pi": search_kl_sum / n,
            "argmax_disagree_rate": search_disagree_sum / n,
            "mean_pick_rank": search_pick_rank_sum / n,
            "mean_delta_logp_at_chosen": search_delta_logp_sum / n,
        }

    # Trainee outcomes (W/D/L/T) for opponent batches. In self-play the trainee
    # plays both colors so per-game outcomes are ambiguous from its perspective;
    # we skip recording in that case.
    if not self_play:
        outcomes = {"W": 0, "D": 0, "L": 0, "T": 0}
        for gid in range(batch_size):
            board = boards[gid]
            if not board.is_game_over():
                outcomes["T"] += 1
                continue
            r = board.result()
            if r == "1/2-1/2":
                outcomes["D"] += 1
            elif (r == "1-0") == policy_is_white[gid]:
                outcomes["W"] += 1
            else:
                outcomes["L"] += 1
        stats["trainee_outcomes"] = outcomes

    return (all_states, all_masks, all_actions, all_old_log_probs,
            all_old_values, all_returns, all_topk_idx, all_log_b_topk,
            all_tb_targets, stats)


# ---- Training loop ------------------------------------------------------

def _cosine_lr_lambda(total_steps: int, min_ratio: float = 0.1):
    total = max(1, total_steps)

    def lr_lambda(step: int) -> float:
        frac = min(1.0, step / total)
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * frac))

    return lr_lambda


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _eval_old_policy(actor_critic_net, states, masks, actions, mb_size):
    """Snapshot old-policy log-probs and values in eval mode for PPO baseline."""
    was_training = actor_critic_net.training
    actor_critic_net.eval()
    old_lps, old_vs = [], []
    try:
        with torch.no_grad():
            n = states.shape[0]
            mb = max(1, min(int(mb_size), n))
            for start in range(0, n, mb):
                end = min(start + mb, n)
                logits, values = actor_critic_net(states[start:end])
                masked = logits.masked_fill(~masks[start:end], -1e9)
                dist = torch.distributions.Categorical(logits=masked)
                old_lps.append(dist.log_prob(actions[start:end]).float())
                old_vs.append(values.view(-1).float())
    finally:
        if was_training:
            actor_critic_net.train()
    return torch.cat(old_lps, 0), torch.cat(old_vs, 0)


def train_actor_critic(actor_critic_net,
                       model_name: str,
                       optimizer: torch.optim.Optimizer,
                       device: str = "cpu",
                       num_batches: int = 1500,
                       eval_interval: int = 100,
                       eval_games: int = 100,
                       gamma: float = 0.99,
                       gae_lamb: float = 0.95,
                       critic_loss_weight: float = 0.5,
                       entropy_weight: float = 0.02,
                       batch_size: int = 64,
                       opening_prob: float = 0.7,
                       temperature: float = 1.0,
                       ppo_clip_ratio: float = 0.2,
                       ppo_epochs: int = 4,
                       ppo_minibatch_size: int = 256,
                       target_kl: float = 0.015,
                       opponent_ratio: float = 0.0,
                       opponent_temperature: float = 1.0,
                       engine_ratio: float = 0.0,
                       engine_path: Optional[str] = None,
                       engine_pool_size: int = 4,
                       engine_move_time: float = 0.05,
                       engine_skill_level: Optional[int] = None,
                       step_penalty: float = 0.001,
                       draw_penalty: float = 0.1,
                       material_shaping_per_pawn: float = 0.0,
                       dtz_shaping_weight: float = 0.0,
                       tb_value_aux_weight: float = 0.0,
                       lookahead_k: int = 5,
                       lookahead_alpha: float = 0.5,
                       lookahead_value_weight: float = 1.0,
                       trainee_search: bool = False,
                       distill_weight: float = 0.0,
                       tablebase_path: Optional[str] = "syzygy",
                       tablebase_terminate_prob: float = 1.0,
                       seed: Optional[int] = None):
    """PPO training loop with self-play, optional checkpoint and engine opponents."""
    if seed is not None:
        _seed_everything(seed)
    actor_critic_net.to(device)
    scheduler = LambdaLR(optimizer, lr_lambda=_cosine_lr_lambda(num_batches, 0.1))
    mirror_perm = MIRROR_ACTION_PERM.to(device)

    engine_pool: Optional[EnginePool] = None
    if engine_path is not None and engine_ratio > 0.0:
        try:
            engine_pool = EnginePool(engine_path, size=engine_pool_size)
            if engine_skill_level is not None:
                engine_pool.configure_all({"Skill Level": int(engine_skill_level)})
            print(f"[INFO] engine pool: {engine_path} x{len(engine_pool.engines)}")
        except Exception as exc:
            print(f"[WARN] could not start engine at {engine_path}: {exc}")
            engine_pool = None

    tablebase: Optional[chess.syzygy.Tablebase] = None
    if tablebase_path and os.path.isdir(tablebase_path):
        try:
            tablebase = chess.syzygy.open_tablebase(tablebase_path)
            print(f"[INFO] tablebase loaded from {tablebase_path}")
        except Exception as exc:
            print(f"[WARN] could not open tablebase at {tablebase_path}: {exc}")
            tablebase = None
    elif tablebase_path:
        print(f"[INFO] tablebase dir not found: {tablebase_path}; running without TB")

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(
        log_dir, f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    log_file = open(log_path, "w", newline="", buffering=1)
    log_writer = csv.writer(log_file)
    _hparams = {
        "num_batches": num_batches, "eval_interval": eval_interval, "eval_games": eval_games,
        "gamma": gamma, "gae_lamb": gae_lamb,
        "critic_loss_weight": critic_loss_weight, "entropy_weight": entropy_weight,
        "batch_size": batch_size, "opening_prob": opening_prob, "temperature": temperature,
        "ppo_clip_ratio": ppo_clip_ratio, "ppo_epochs": ppo_epochs,
        "ppo_minibatch_size": ppo_minibatch_size, "target_kl": target_kl,
        "opponent_ratio": opponent_ratio, "opponent_temperature": opponent_temperature,
        "engine_ratio": engine_ratio, "engine_path": engine_path,
        "engine_pool_size": engine_pool_size, "engine_move_time": engine_move_time,
        "engine_skill_level": engine_skill_level,
        "step_penalty": step_penalty, "draw_penalty": draw_penalty,
        "material_shaping_per_pawn": material_shaping_per_pawn,
        "dtz_shaping_weight": dtz_shaping_weight,
        "tb_value_aux_weight": tb_value_aux_weight,
        "lookahead_k": lookahead_k, "lookahead_alpha": lookahead_alpha,
        "lookahead_value_weight": lookahead_value_weight,
        "trainee_search": trainee_search, "distill_weight": distill_weight,
        "tablebase_path": tablebase_path, "tablebase_terminate_prob": tablebase_terminate_prob,
        "seed": seed,
        "optimizer": type(optimizer).__name__,
        "lr": [g["lr"] for g in optimizer.param_groups],
    }
    log_writer.writerow(["# " + " ".join(f"{k}={v}" for k, v in _hparams.items())])
    log_writer.writerow(["batch", "wins", "draws", "losses"])
    print(f"[INFO] eval log: {log_path}")

    # Cumulative trainee-vs-Stockfish-engine outcomes across the run.
    engine_cum = {"W": 0, "D": 0, "L": 0, "T": 0}

    try:
        baseline = evaluate_vs_random(
            actor_critic_net, num_games=eval_games, device=device,
        )
        print(
            f"[Eval at batch 0] Wins: {baseline['wins']}, "
            f"Draws: {baseline['draws']}, Losses: {baseline['losses']}"
        )
        log_writer.writerow([0, baseline["wins"], baseline["draws"], baseline["losses"]])

        for batch in range(num_batches):
            t0 = perf_counter()

            r = random.random()
            trainee_search_now = False
            if engine_pool is not None and r < engine_ratio:
                opponent_fn = _engine_opponent_fn(engine_pool, engine_move_time)
                source = "engine"
                trainee_search_now = trainee_search
            elif r < engine_ratio + opponent_ratio:
                opp_net, opp_path = _load_random_opponent(device)
                if opp_net is None:
                    opponent_fn = None
                    source = "self-play (no checkpoints)"
                else:
                    opponent_fn = _checkpoint_opponent_fn(
                        opp_net, device, opponent_temperature,
                        top_k=lookahead_k, alpha=lookahead_alpha,
                        value_weight=lookahead_value_weight,
                    )
                    source = f"checkpoint ({os.path.basename(opp_path)})"
                    # Trainee search applies in opponent batches (checkpoint and
                    # engine); pure π in self-play.
                    trainee_search_now = trainee_search
            else:
                opponent_fn = None
                source = "self-play"

            actor_critic_net.eval()
            (states, masks, actions, old_lps, old_vs, returns,
             topk_idxs, log_b_topks, tb_targets, rollout_stats) = generate_batch(
                actor_critic_net, batch_size=batch_size, gamma=gamma, gae_lamb=gae_lamb,
                device=device, temperature=temperature, opening_prob=opening_prob,
                opponent_move_fn=opponent_fn,
                step_penalty=step_penalty, draw_penalty=draw_penalty,
                material_shaping_per_pawn=material_shaping_per_pawn,
                dtz_shaping_weight=dtz_shaping_weight,
                tb_value_aux_weight=tb_value_aux_weight,
                tablebase=tablebase,
                tablebase_terminate_prob=tablebase_terminate_prob,
                trainee_search=trainee_search_now,
                trainee_top_k=lookahead_k,
                trainee_alpha=lookahead_alpha,
                trainee_value_weight=lookahead_value_weight,
                progress_label=f"[Batch {batch+1}/{num_batches}] {source}",
            )
            t1 = perf_counter()

            if not actions:
                print(f"[Batch {batch+1}] {source} | empty, skipping.")
                scheduler.step()
                continue

            states_t = torch.stack(states).to(device)
            masks_t = torch.stack(masks).to(device)
            actions_t = torch.tensor(actions, dtype=torch.long, device=device)
            returns_t = torch.stack(returns).view(-1).to(device)
            tb_aux_on = tb_value_aux_weight > 0.0 and any(t is not None for t in tb_targets)
            if tb_aux_on:
                # NaN marks "no tablebase target" (outside the domain); masked
                # out of the auxiliary loss, never treated as a real value.
                tb_target_t = torch.tensor(
                    [float("nan") if t is None else t for t in tb_targets],
                    dtype=torch.float32, device=device,
                )
            distill_on = (
                distill_weight > 0.0 and trainee_search
                and topk_idxs and topk_idxs[0] is not None
            )
            if distill_on:
                # Per-ply candidate sets have variable length (widened lookahead
                # unions top-k(π) with captures/checks, which differs per board).
                # Pad to the rollout-global max so torch.stack works; padding
                # slots get b ≈ 0 via -1e9, so they contribute nothing to the
                # distill cross-entropy.
                max_k_global = max(t.shape[0] for t in topk_idxs)
                padded_idxs, padded_logbs = [], []
                for idx, lb in zip(topk_idxs, log_b_topks):
                    cur_k = idx.shape[0]
                    if cur_k < max_k_global:
                        pad_idx = torch.zeros(
                            max_k_global - cur_k, dtype=idx.dtype, device=idx.device,
                        )
                        pad_lb = torch.full(
                            (max_k_global - cur_k,), -1e9,
                            dtype=lb.dtype, device=lb.device,
                        )
                        padded_idxs.append(torch.cat([idx, pad_idx]))
                        padded_logbs.append(torch.cat([lb, pad_lb]))
                    else:
                        padded_idxs.append(idx)
                        padded_logbs.append(lb)
                topk_idx_t = torch.stack(padded_idxs).to(device).long()
                log_b_topk_t = torch.stack(padded_logbs).to(device).float()
            else:
                topk_idx_t = None
                log_b_topk_t = None

            # Mirror augmentation: file-flip the board and remap actions.
            m_states = mirror_board_tensor_batch(states_t)
            m_masks = masks_t[:, mirror_perm]
            m_actions = mirror_perm[actions_t]
            states_t = torch.cat([states_t, m_states], 0)
            masks_t = torch.cat([masks_t, m_masks], 0)
            actions_t = torch.cat([actions_t, m_actions], 0)
            returns_t = torch.cat([returns_t, returns_t], 0)
            if distill_on:
                m_topk_idx = mirror_perm[topk_idx_t]
                topk_idx_t = torch.cat([topk_idx_t, m_topk_idx], 0)
                log_b_topk_t = torch.cat([log_b_topk_t, log_b_topk_t], 0)
            if tb_aux_on:
                # Tablebase value target is invariant to the file-flip mirror.
                tb_target_t = torch.cat([tb_target_t, tb_target_t], 0)

            # pi_old_lp_t: log π at rollout time (== current weights, since no PPO
            # update has happened yet this batch). Used for the approx_kl /
            # early-stop diagnostic so it always measures *policy drift*.
            pi_old_lp_t, old_v_t = _eval_old_policy(
                actor_critic_net, states_t, masks_t, actions_t,
                ppo_minibatch_size,
            )
            if trainee_search_now and old_lps:
                # IS ratio uses log b (the search behavior policy) as denominator,
                # so the actor loss is π_new(a|s) / b(a|s) — correct under search
                # rollouts. Mirror states inherit b by symmetry of the search.
                behavior_lp = torch.stack(old_lps).to(device).float()
                old_lp_t = torch.cat([behavior_lp, behavior_lp], 0)
            else:
                old_lp_t = pi_old_lp_t
            advantages = returns_t - old_v_t
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

            N = states_t.shape[0]
            mb_size = max(1, min(int(ppo_minibatch_size), N))

            actor_loss_sum = critic_loss_sum = entropy_loss_sum = 0.0
            distill_loss_sum = 0.0
            tb_aux_loss_sum = 0.0
            approx_kl_sum = clip_frac_sum = 0.0
            update_count = 0
            early_stopped_at: Optional[int] = None

            actor_critic_net.train()
            for epoch in range(max(1, ppo_epochs)):
                perm_idx = torch.randperm(N, device=device)
                epoch_kl_sum = 0.0
                epoch_steps = 0
                for start in range(0, N, mb_size):
                    end = min(start + mb_size, N)
                    mb = perm_idx[start:end]
                    mb_states = states_t[mb]
                    mb_masks = masks_t[mb]
                    mb_actions = actions_t[mb]
                    mb_old_lp = old_lp_t[mb]            # IS denominator (log b or log π_old)
                    mb_pi_old_lp = pi_old_lp_t[mb]      # always log π_old, for KL diagnostic
                    mb_adv = advantages[mb]
                    mb_ret = returns_t[mb]

                    logits, values = actor_critic_net(mb_states)
                    masked = logits.masked_fill(~mb_masks, -1e9)
                    dist = torch.distributions.Categorical(logits=masked)
                    new_lp = dist.log_prob(mb_actions)
                    entropies = dist.entropy()
                    ratios = torch.exp(new_lp - mb_old_lp)
                    clipped = torch.clamp(ratios, 1.0 - ppo_clip_ratio, 1.0 + ppo_clip_ratio)
                    actor_loss = -torch.min(ratios * mb_adv, clipped * mb_adv).mean()
                    critic_loss = F.mse_loss(values.view(-1), mb_ret)
                    entropy_loss = -entropies.mean()
                    if distill_on:
                        # Soft cross-entropy toward the search distribution b:
                        #   per-state loss = -E_b[log π_new] = KL(b ‖ π_new) − H(b).
                        # Outcome-filtered: only states where the search-chosen action
                        # had positive advantage contribute. This aligns the distill
                        # gradient with the PPO actor signal (also advantage-weighted)
                        # — search picks that the rollout vindicated reinforce π;
                        # search picks that lost don't get copied. Without this filter,
                        # the asymmetric clip protects the actor but distill flows
                        # unrestricted, dragging π toward b's biased choices.
                        new_log_pi = F.log_softmax(masked, dim=1)
                        new_log_pi_topk = new_log_pi.gather(1, topk_idx_t[mb])
                        mb_b_topk = log_b_topk_t[mb].exp()
                        distill_per_state = -(mb_b_topk * new_log_pi_topk).sum(dim=1)
                        pos_mask = (mb_adv > 0).float()
                        n_pos = pos_mask.sum().clamp(min=1.0)
                        distill_loss = (distill_per_state * pos_mask).sum() / n_pos
                    else:
                        distill_loss = torch.zeros((), device=device)
                    if tb_aux_on:
                        # Value-head-only supervision toward tablebase ground
                        # truth — kept off the reward stream so it corrects the
                        # critic's baseline without directly rewarding the actor;
                        # a better-calibrated baseline still sharpens the actor's
                        # advantage (and thus its training signal) for failed
                        # conversions, but only via the normal PPO mechanism.
                        mb_tb_target = tb_target_t[mb]
                        tb_valid = ~torch.isnan(mb_tb_target)
                        if tb_valid.any():
                            tb_aux_loss = F.mse_loss(values.view(-1)[tb_valid], mb_tb_target[tb_valid])
                        else:
                            tb_aux_loss = torch.zeros((), device=device)
                    else:
                        tb_aux_loss = torch.zeros((), device=device)
                    total_loss = (
                        actor_loss
                        + critic_loss_weight * critic_loss
                        + entropy_weight * entropy_loss
                        + distill_weight * distill_loss
                        + tb_value_aux_weight * tb_aux_loss
                    )

                    optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor_critic_net.parameters(), max_norm=0.5)
                    optimizer.step()

                    with torch.no_grad():
                        # Trust-region check: drift of π_new from π_old, NOT from b.
                        # Schulman k3 estimator: unbiased, non-negative.
                        log_r_pi = new_lp - mb_pi_old_lp
                        approx_kl = (torch.exp(log_r_pi) - 1 - log_r_pi).mean().item()
                        clip_frac = ((ratios - 1.0).abs() > ppo_clip_ratio).float().mean().item()

                    actor_loss_sum += actor_loss.item()
                    critic_loss_sum += critic_loss.item()
                    entropy_loss_sum += entropy_loss.item()
                    distill_loss_sum += distill_loss.item()
                    tb_aux_loss_sum += tb_aux_loss.item()
                    approx_kl_sum += approx_kl
                    clip_frac_sum += clip_frac
                    update_count += 1
                    epoch_kl_sum += approx_kl
                    epoch_steps += 1

                if epoch_kl_sum / max(1, epoch_steps) > 1.5 * target_kl:
                    early_stopped_at = epoch + 1
                    break

            scheduler.step()
            denom = max(1, update_count)
            avg_actor = actor_loss_sum / denom
            avg_critic = critic_loss_sum / denom
            avg_entropy = entropy_loss_sum / denom
            avg_distill = distill_loss_sum / denom
            avg_tb_aux = tb_aux_loss_sum / denom
            avg_kl = approx_kl_sum / denom
            avg_clip = clip_frac_sum / denom
            t2 = perf_counter()

            suffix = f" | EarlyStop@epoch={early_stopped_at}" if early_stopped_at else ""
            search_info = ""
            if rollout_stats.get("n_search_states", 0) > 0:
                s = rollout_stats
                search_info = (
                    f" | π↔search KL: {s['mean_kl_b_pi']:.3f} "
                    f"disagree: {s['argmax_disagree_rate']*100:.0f}% "
                    f"rank: {s['mean_pick_rank']:.2f} "
                    f"Δlogp@a: {s['mean_delta_logp_at_chosen']:+.3f}"
                )
            distill_info = f" Distill: {avg_distill:.4f}" if distill_weight > 0 else ""
            tb_aux_info = f" TBaux: {avg_tb_aux:.4f}" if tb_value_aux_weight > 0 else ""
            outcome_info = ""
            outcomes = rollout_stats.get("trainee_outcomes")
            if outcomes is not None:
                if source == "engine":
                    engine_cum["W"] += outcomes["W"]
                    engine_cum["D"] += outcomes["D"]
                    engine_cum["L"] += outcomes["L"]
                    engine_cum["T"] += outcomes["T"]
                    outcome_info = (
                        f" | vs SF: {outcomes['W']}W/{outcomes['D']}D/{outcomes['L']}L"
                        f" (cum {engine_cum['W']}/{engine_cum['D']}/{engine_cum['L']})"
                    )
                else:
                    outcome_info = (
                        f" | trainee: {outcomes['W']}W/{outcomes['D']}D/{outcomes['L']}L"
                    )
            print(
                f"[Batch {batch+1}] {source} | DataGen: {t1 - t0:.2f}s "
                f"Train: {t2 - t1:.2f}s{suffix} | "
                f"Actor: {avg_actor:.4f} Critic: {avg_critic:.4f} "
                f"Entropy: {avg_entropy:.4f}{distill_info}{tb_aux_info} "
                f"KL: {avg_kl:.4f} Clip: {avg_clip:.3f}"
                f"{search_info}"
                f"{outcome_info}"
            )

            if (batch + 1) % eval_interval == 0:
                eval_stats = evaluate_vs_random(
                    actor_critic_net, num_games=eval_games, device=device,
                )
                print(
                    f"[Eval at batch {batch+1}] Wins: {eval_stats['wins']}, "
                    f"Draws: {eval_stats['draws']}, Losses: {eval_stats['losses']}"
                )
                log_writer.writerow([
                    batch + 1, eval_stats["wins"], eval_stats["draws"], eval_stats["losses"],
                ])
                os.makedirs(MODELS_DIR, exist_ok=True)
                torch.save(actor_critic_net.state_dict(),
                           f"{MODELS_DIR}/{model_name}_checkpoint_{batch}.pt")
    finally:
        log_file.close()
        if engine_pool is not None:
            engine_pool.close()
        if tablebase is not None:
            tablebase.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the chess actor-critic model.")
    parser.add_argument(
        "--init-from", required=True,
        help="Model checkpoint used as the warm-start weights.",
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--num-batches", type=int, default=400)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1401)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--trainee-search", action="store_true",
        help="Trainee samples from the lookahead search (policy-scale formula) "
             "in checkpoint and engine batches; pure pi in self-play.",
    )
    parser.add_argument(
        "--lookahead-alpha", type=float, default=1.0,
        help="Weight on log pi in the search score (1.0 = policy-scale formula).",
    )
    parser.add_argument(
        "--value-weight", type=float, default=1.0,
        help="Weight beta on net-derived quiescence values in the search score.",
    )
    parser.add_argument(
        "--dtz-shaping-weight", type=float, default=0.0,
        help="Weight on the DTZ conversion-progress potential inside <=5-man "
             "tablebase wins (0 = off). Dense reward for shrinking distance "
             "to the forced zeroing move; silent outside a confirmed win.",
    )
    parser.add_argument(
        "--tb-value-aux-weight", type=float, default=0.0,
        help="Weight on a value-head-only auxiliary loss toward tablebase "
             "ground truth (0 = off). Kept out of the reward stream so it "
             "corrects the critic's baseline without directly rewarding the "
             "actor for reaching a won position.",
    )
    parser.add_argument(
        "--num-filters", type=int, default=DEFAULT_NUM_FILTERS,
        help="Residual tower channel width. Must match --init-from's architecture.",
    )
    parser.add_argument(
        "--num-residual-blocks", type=int, default=DEFAULT_NUM_RESIDUAL_BLOCKS,
        help="Number of residual blocks. Must match --init-from's architecture.",
    )
    parser.add_argument(
        "--tablebase-terminate-prob", type=float, default=0.25,
        help="Per-game probability of auto-terminating with the WDL result "
             "the instant a <=5-man tablebase position is reached. Set to 0 "
             "to always play out endgames (e.g. when using "
             "--tb-value-aux-weight, so the actor never gets an injected "
             "terminal reward for merely reaching a won position).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if torch.version.hip is not None:
        os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")

    run_config = {
        "model_name": args.model_name,
        "init_from": args.init_from,
        "num_batches": args.num_batches,
        "eval_interval": args.eval_interval,
        "eval_games": args.eval_games,
        "seed": args.seed,
        "device": run_device,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "precision": "fp32",
        "miopen_find_mode": os.environ.get("MIOPEN_FIND_MODE"),
        "trainee_search": args.trainee_search,
        "lookahead_alpha": args.lookahead_alpha,
        "value_weight": args.value_weight,
        "dtz_shaping_weight": args.dtz_shaping_weight,
        "tb_value_aux_weight": args.tb_value_aux_weight,
        "tablebase_terminate_prob": args.tablebase_terminate_prob,
        "num_filters": args.num_filters,
        "num_residual_blocks": args.num_residual_blocks,
        "distill_weight": 0.0,
    }
    print("[INFO] training configuration")
    for key, value in run_config.items():
        print(f"  {key}={value}")

    if not os.path.exists(args.init_from):
        raise SystemExit(f"Initial checkpoint not found: {args.init_from}")

    actor_critic_net = ActorCriticResNet(
        use_se=False,
        num_filters=args.num_filters,
        num_residual_blocks=args.num_residual_blocks,
    ).to(run_device)
    print(f"Loading initial weights from: {args.init_from}")
    incompatible = load_actor_critic_state_dict(
        actor_critic_net, torch.load(args.init_from, map_location=run_device)
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Initial checkpoint did not load at full fidelity: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )

    optimizer = torch.optim.AdamW(
        actor_critic_net.parameters(), lr=5e-5, betas=(0.9, 0.999), weight_decay=0.0
    )

    train_actor_critic(
        actor_critic_net=actor_critic_net,
        model_name=args.model_name,
        optimizer=optimizer,
        device=run_device,
        num_batches=args.num_batches,
        eval_interval=args.eval_interval,
        eval_games=args.eval_games,
        gamma=0.98,
        gae_lamb=0.95,
        critic_loss_weight=0.5,
        entropy_weight=0.005,
        batch_size=32,
        opening_prob=0.6,
        temperature=1.0,
        ppo_clip_ratio=0.2,
        ppo_epochs=4,
        ppo_minibatch_size=256,
        target_kl=0.015,
        opponent_ratio=0.45,
        opponent_temperature=1.0,
        engine_ratio=0.1,
        engine_path="./stockfish/stockfish",
        engine_pool_size=4,
        engine_move_time=0.01,
        engine_skill_level=0,
        step_penalty=0.001,
        draw_penalty=0.1,
        material_shaping_per_pawn=0.025,
        dtz_shaping_weight=args.dtz_shaping_weight,
        tb_value_aux_weight=args.tb_value_aux_weight,
        lookahead_k=4,
        lookahead_alpha=args.lookahead_alpha,
        lookahead_value_weight=args.value_weight,
        trainee_search=args.trainee_search,
        distill_weight=0.0,
        tablebase_terminate_prob=args.tablebase_terminate_prob,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
