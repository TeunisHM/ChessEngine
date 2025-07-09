import torch
import torch.nn as nn
import torch.optim as optim
import chess
import chess.pgn
import random

# === 1. Define Policy Network ===

class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(773, 256),  # board state input size
            nn.ReLU(),
            nn.Linear(256, 4672)  # total legal UCI moves in chess (e.g., e2e4)
        )

    def forward(self, x):
        return self.net(x)

# === 2. Helper Functions ===

def board_to_tensor(board):
    """Encode board into a flat vector (simplified)."""
    piece_map = board.piece_map()
    board_tensor = torch.zeros(64 * 12)
    for square, piece in piece_map.items():
        offset = "PNBRQKpnbrqk".index(piece.symbol())
        board_tensor[64 * offset + square] = 1
    turn_tensor = torch.tensor([board.turn], dtype=torch.float32)
    return torch.cat([board_tensor, turn_tensor])  # shape (773,)

def legal_moves_mask(board):
    """Binary mask over all possible UCI moves."""
    mask = torch.zeros(4672)
    for move in board.legal_moves:
        idx = move_to_index(move)
        if idx is not None:
            mask[idx] = 1
    return mask

# === 3. Move Encoding (UCI indexing) ===

uci_to_index = {}
index_to_uci = []
for from_sq in chess.SQUARES:
    for to_sq in chess.SQUARES:
        for promo in [None, 'q', 'r', 'b', 'n']:
            try:
                move = chess.Move(from_sq, to_sq, promotion=chess.Piece.from_symbol(promo).piece_type if promo else None)
                uci = move.uci()
                if uci not in uci_to_index:
                    uci_to_index[uci] = len(index_to_uci)
                    index_to_uci.append(uci)
            except:
                continue

def move_to_index(move):
    return uci_to_index.get(move.uci(), None)

def index_to_move(idx):
    return chess.Move.from_uci(index_to_uci[idx])

# === 4. Play a Self-Game and Store Trajectory ===

def self_play_game(policy_net):
    board = chess.Board()
    trajectory = []

    while not board.is_game_over():
        state = board_to_tensor(board)
        logits = policy_net(state)
        mask = legal_moves_mask(board)
        probs = torch.softmax(logits.masked_fill(mask == 0, -1e9), dim=0)
        dist = torch.distributions.Categorical(probs)
        move_idx = dist.sample().item()

        move = index_to_move(move_idx)
        if move not in board.legal_moves:
            continue  # skip illegal (should be rare)
        trajectory.append((state, move_idx))
        board.push(move)

    result = board.result()
    if result == "1-0":
        reward = 1
    elif result == "0-1":
        reward = -1
    else:
        reward = 0
    return trajectory, reward

# === 5. Train with REINFORCE ===

def train(policy_net, optimizer, num_games=100):
    for game in range(num_games):
        trajectory, reward = self_play_game(policy_net)
        loss = 0
        for state, move_idx in trajectory:
            logits = policy_net(state)
            log_prob = torch.log_softmax(logits, dim=0)[move_idx]
            loss -= log_prob * reward  # REINFORCE update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Game {game+1}, reward: {reward}, moves: {len(trajectory)}")

# === 6. Run Training ===

policy_net = PolicyNet()
optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)

train(policy_net, optimizer, num_games=1000)
