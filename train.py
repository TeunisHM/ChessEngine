import torch
import torch.nn as nn
import torch.optim as optim
import chess
import chess.pgn
import random
from helper import move_to_index, index_to_move, board_to_tensor, legal_moves_mask

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
    trajectory = []

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
            print("Illegal move detected, skipping")
            continue
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

def train(policy_net, optimizer, num_games=100):
    for game in range(num_games):
        trajectory, reward = self_play_game(policy_net)
        loss = 0
        for i, (state, move_idx) in enumerate(trajectory):
            logits = policy_net(state)
            log_prob = torch.log_softmax(logits, dim=0)[move_idx]

            # Alternate reward: even = player who won, odd = opponent
            signed_reward = reward if i % 2 == 0 else -reward

            loss -= log_prob * signed_reward  # REINFORCE update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Game {game+1}, reward: {reward}, moves: {len(trajectory)}")

if __name__ == "__main__":
    policy_net = PolicyNet()
    optimizer = optim.Adam(policy_net.parameters(), lr=1.5e-3)

    train(policy_net, optimizer, num_games=10)
    torch.save(policy_net.state_dict(), "policy.pt")