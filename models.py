import torch
import torch.nn as nn
import torch.nn.functional as F

from helper import ACTION_SPACE_SIZE, BOARD_TENSOR_PLANES

DEFAULT_INPUT_CHANNELS = BOARD_TENSOR_PLANES
DEFAULT_NUM_RESIDUAL_BLOCKS = 4
DEFAULT_NUM_FILTERS = 64


class ResidualBlock(nn.Module):
    """A standard residual block for a ResNet."""

    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)  
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = F.relu(out)
        return out


class ActorCriticResNet(nn.Module):
    def __init__(
        self,
        num_input_channels=DEFAULT_INPUT_CHANNELS,
        num_residual_blocks=DEFAULT_NUM_RESIDUAL_BLOCKS,
        num_filters=DEFAULT_NUM_FILTERS,
    ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(num_input_channels, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
        )

        self.residual_tower = nn.Sequential(
            *[ResidualBlock(num_filters) for _ in range(num_residual_blocks)]
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, ACTION_SPACE_SIZE),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(num_filters, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        out = self.stem(x)
        out = self.residual_tower(out)
        policy_logits = self.policy_head(out)
        state_value = self.value_head(out)
        return policy_logits, state_value

def load_actor_critic_state_dict(model, state_dict):
    return model.load_state_dict(state_dict)
