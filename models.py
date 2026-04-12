import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_INPUT_CHANNELS = 20


class ResidualBlock(nn.Module):
    """A standard residual block for a ResNet."""

    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)  # Batch Norm is crucial
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
    def __init__(self, num_input_channels=DEFAULT_INPUT_CHANNELS, num_residual_blocks=4, num_filters=64):
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
            nn.Linear(2 * 8 * 8, 4672),
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


def adapt_legacy_state_dict(state_dict, target_in_channels=DEFAULT_INPUT_CHANNELS):
    """
    Expand or trim the input stem to keep older checkpoints loadable after
    encoder plane count changes.
    """
    adapted = dict(state_dict)
    stem_key = "stem.0.weight"
    stem_weight = adapted.get(stem_key)

    if stem_weight is None or stem_weight.ndim != 4:
        return adapted

    current_in_channels = stem_weight.shape[1]
    if current_in_channels == target_in_channels:
        return adapted

    if current_in_channels < target_in_channels:
        extra_channels = target_in_channels - current_in_channels
        padding = torch.zeros(
            stem_weight.shape[0],
            extra_channels,
            stem_weight.shape[2],
            stem_weight.shape[3],
            dtype=stem_weight.dtype,
            device=stem_weight.device,
        )
        adapted[stem_key] = torch.cat((stem_weight, padding), dim=1)
        return adapted

    adapted[stem_key] = stem_weight[:, :target_in_channels, :, :].clone()
    return adapted


def load_actor_critic_state_dict(model, state_dict, strict=True):
    return model.load_state_dict(adapt_legacy_state_dict(state_dict), strict=strict)
