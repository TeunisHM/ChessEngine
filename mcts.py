"""AlphaZero-style PUCT Monte Carlo Tree Search.

Usage:
    tree = MCTS(actor_critic_net, device, c_puct=1.5, dirichlet_alpha=0.3,
                dirichlet_frac=0.25)
    move, pi, root_value = tree.run(board, num_simulations=200, temperature=1.0)

The returned `pi` is an action-space-sized vector of visit-count probabilities
(proportional to N(s,a)^(1/temperature)) suitable as a supervised policy
target.
"""
import math
from typing import Optional

import chess
import torch
import torch.nn.functional as F

from helper import ACTION_SPACE_SIZE, board_to_tensor, index_to_move, legal_moves_mask, move_to_index


class _Node:
    __slots__ = ("board", "parent", "prior", "children", "N", "W", "Q", "is_terminal", "terminal_value", "expanded")

    def __init__(self, board: chess.Board, parent: Optional["_Node"] = None, prior: float = 0.0):
        self.board = board
        self.parent = parent
        self.prior = prior
        self.children: dict = {}  # action_idx -> _Node
        self.N = 0
        self.W = 0.0
        self.Q = 0.0
        self.is_terminal = False
        self.terminal_value = 0.0
        self.expanded = False


class MCTS:
    def __init__(self,
                 actor_critic_net,
                 device: str = "cpu",
                 c_puct: float = 1.5,
                 dirichlet_alpha: float = 0.3,
                 dirichlet_frac: float = 0.25):
        self.net = actor_critic_net
        self.device = device
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_frac = dirichlet_frac

    @torch.no_grad()
    def _evaluate(self, board: chess.Board):
        """Run the network on `board` and return (policy_probs over the action space, value).

        `value` is from the perspective of the side to move at `board`.
        """
        was_training = self.net.training
        self.net.eval()
        try:
            state = board_to_tensor(board).unsqueeze(0).to(self.device)
            logits, value = self.net(state)
            mask = legal_moves_mask(board).to(self.device)
            if mask.sum() == 0:
                # Shouldn't happen unless game is over (handled elsewhere).
                probs = torch.zeros_like(logits[0])
            else:
                probs = F.softmax(logits[0].masked_fill(~mask, -1e9), dim=0)
            return probs.cpu(), float(value.item())
        finally:
            if was_training:
                self.net.train()

    def _expand(self, node: _Node, add_dirichlet_noise: bool = False):
        board = node.board
        if board.is_game_over():
            node.is_terminal = True
            result = board.result()
            # Value from side-to-move's perspective at this terminal node.
            if result == "1-0":
                val = 1.0 if board.turn == chess.WHITE else -1.0
            elif result == "0-1":
                val = 1.0 if board.turn == chess.BLACK else -1.0
            else:
                val = 0.0
            node.terminal_value = val
            node.expanded = True
            return val

        probs, value = self._evaluate(board)

        legal_indices = []
        legal_priors = []
        for mv in board.legal_moves:
            try:
                idx = move_to_index(mv, board)
            except Exception:
                continue
            legal_indices.append(idx)
            legal_priors.append(float(probs[idx].item()))

        # Fallback to uniform if the network assigns zero mass everywhere.
        total = sum(legal_priors)
        if total <= 1e-12:
            legal_priors = [1.0 / max(1, len(legal_indices))] * len(legal_indices)
        else:
            legal_priors = [p / total for p in legal_priors]

        if add_dirichlet_noise and legal_priors:
            noise = torch.distributions.Dirichlet(
                torch.full((len(legal_priors),), self.dirichlet_alpha)
            ).sample().tolist()
            legal_priors = [
                (1 - self.dirichlet_frac) * p + self.dirichlet_frac * n
                for p, n in zip(legal_priors, noise)
            ]

        for idx, prior in zip(legal_indices, legal_priors):
            child_board = board.copy(stack=False)
            move = index_to_move(idx, child_board)
            if move is None or move not in child_board.legal_moves:
                continue
            child_board.push(move)
            node.children[idx] = _Node(child_board, parent=node, prior=prior)

        node.expanded = True
        return value

    def _select_child(self, node: _Node):
        """PUCT selection. Returns (action_idx, child_node)."""
        total_N = max(1, node.N)
        sqrt_total = math.sqrt(total_N)
        best_score = -float("inf")
        best_action = None
        best_child = None
        for action, child in node.children.items():
            # Q(s,a) from *this* node's perspective.
            # Child's value was computed from child's side-to-move (opponent).
            # So from current player's perspective it's -child.Q.
            q = -child.Q if child.N > 0 else 0.0
            u = self.c_puct * child.prior * sqrt_total / (1 + child.N)
            score = q + u
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        return best_action, best_child

    def run(self,
            board: chess.Board,
            num_simulations: int = 100,
            temperature: float = 1.0,
            add_root_noise: bool = True):
        """Run num_simulations PUCT simulations from board.

        Returns (move, pi_vector, root_value)
          - move: selected chess.Move
          - pi_vector: action-space-sized torch.Tensor of visit-count probabilities
            ∝ N(s,a)^(1/temperature). Zero at illegal moves.
          - root_value: approximate state value from side-to-move perspective.
        """
        root = _Node(board.copy(stack=False), parent=None, prior=0.0)
        self._expand(root, add_dirichlet_noise=add_root_noise)

        for _ in range(max(1, num_simulations)):
            node = root
            path = [node]
            while node.expanded and not node.is_terminal and node.children:
                _, child = self._select_child(node)
                if child is None:
                    break
                node = child
                path.append(node)

            if not node.expanded:
                value = self._expand(node)
            elif node.is_terminal:
                value = node.terminal_value
            else:
                # No legal moves but not terminal (can't happen in chess).
                value = 0.0

            # Backpropagate. `value` is from `node`'s side-to-move perspective.
            # Each ancestor is the opponent of its child, so flip sign per step.
            for ancestor in reversed(path):
                ancestor.N += 1
                ancestor.W += value
                ancestor.Q = ancestor.W / ancestor.N
                value = -value

        # Build pi from visit counts.
        pi = torch.zeros(ACTION_SPACE_SIZE, dtype=torch.float32)
        for action, child in root.children.items():
            pi[action] = child.N

        if pi.sum() <= 0:
            # Shouldn't happen, but fall back to uniform over legal moves.
            legal = legal_moves_mask(board).float()
            pi = legal / max(1.0, legal.sum().item())
        elif temperature is None or temperature <= 1e-6:
            # Greedy: one-hot on most-visited.
            pi = torch.zeros(ACTION_SPACE_SIZE, dtype=torch.float32)
            best = max(root.children.items(), key=lambda kv: kv[1].N)
            pi[best[0]] = 1.0
        else:
            counts = pi.pow(1.0 / float(temperature))
            pi = counts / counts.sum()

        # Sample move from pi.
        if temperature is None or temperature <= 1e-6:
            action = int(pi.argmax().item())
        else:
            action = int(torch.multinomial(pi, num_samples=1).item())

        move = index_to_move(action, board)
        # Root value from side-to-move perspective.
        root_value = root.Q
        return move, pi, root_value, action
