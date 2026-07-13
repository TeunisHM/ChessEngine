import torch
import torch.nn as nn
import torch.nn.functional as F

from helper import ACTION_SPACE_SIZE, BOARD_TENSOR_PLANES

DEFAULT_INPUT_CHANNELS = BOARD_TENSOR_PLANES
DEFAULT_NUM_RESIDUAL_BLOCKS = 8
DEFAULT_NUM_FILTERS = 128
DEFAULT_TRANSFORMER_HEADS = 4

def _norm2d(num_channels):
    # GroupNorm does not depend on running batch statistics, so PPO rollouts
    # and PPO updates see the same normalization behavior in eval/train modes.
    num_groups = min(32, num_channels)
    while num_channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups, num_channels)


class ResidualBlock(nn.Module):
    """A standard residual block for a ResNet."""

    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = _norm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = _norm2d(num_channels)

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


class BoardTransformerLayer(nn.Module):
    """Pre-LN self-attention + FFN over the 64 board squares as tokens."""

    def __init__(self, num_channels, num_heads=DEFAULT_TRANSFORMER_HEADS,
                 dim_feedforward=None, dropout=0.0):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * num_channels
        self.pos_embed = nn.Parameter(torch.zeros(64, num_channels))
        self.layer = nn.TransformerEncoderLayer(
            d_model=num_channels,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2) + self.pos_embed
        tokens = self.layer(tokens)
        return tokens.transpose(1, 2).reshape(B, C, H, W)


class ConvPolicyHead(nn.Module):
    """Spatial policy head: the logit for action (from_square s, move plane p)
    is channel p of the 1x1 conv output at position s. Move geometry is learned
    once and shared across squares, and each square's logits are computed from
    that square's full feature vector — no global bottleneck.

    Output is flattened square-major to match move_to_index = s * 73 + p
    (board_to_tensor is [plane, rank, file] with s = rank * 8 + file).
    """

    def __init__(self, num_filters):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.norm = _norm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, ACTION_SPACE_SIZE // 64, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.norm(self.conv1(x)))
        out = self.conv2(out)  # (B, 73, 8, 8)
        return out.flatten(2).transpose(1, 2).reshape(x.shape[0], ACTION_SPACE_SIZE)


class ActorCriticResNet(nn.Module):
    def __init__(
        self,
        num_input_channels=DEFAULT_INPUT_CHANNELS,
        num_residual_blocks=DEFAULT_NUM_RESIDUAL_BLOCKS,
        num_filters=DEFAULT_NUM_FILTERS,
        transformer_heads=DEFAULT_TRANSFORMER_HEADS,
        policy_head_style="conv",
    ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(num_input_channels, num_filters, kernel_size=3, padding=1),
            _norm2d(num_filters),
            nn.ReLU(),
        )

        self.residual_tower = nn.Sequential(
            *[ResidualBlock(num_filters) for _ in range(num_residual_blocks)]
        )

        self.transformer = BoardTransformerLayer(num_filters, num_heads=transformer_heads)

        if policy_head_style == "conv":
            self.policy_head = ConvPolicyHead(num_filters)
        elif policy_head_style == "dense":
            # Legacy AlphaGo-Zero-style head (checkpoints v10 and earlier).
            self.policy_head = nn.Sequential(
                nn.Conv2d(num_filters, 2, kernel_size=1),
                _norm2d(2),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(2 * 8 * 8, ACTION_SPACE_SIZE),
            )
        else:
            raise ValueError(f"unknown policy_head_style: {policy_head_style!r}")

        self.value_head = nn.Sequential(
            nn.Conv2d(num_filters, 1, kernel_size=1),
            _norm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        out = self.stem(x)
        out = self.residual_tower(out)
        out = self.transformer(out)
        policy_logits = self.policy_head(out)
        state_value = self.value_head(out)
        return policy_logits, state_value

def load_actor_critic_state_dict(model, state_dict):
    bn_stat_suffixes = (".running_mean", ".running_var", ".num_batches_tracked")
    model_sd = model.state_dict()
    filtered = {}
    dropped = []
    for key, value in state_dict.items():
        if key.endswith(bn_stat_suffixes):
            continue
        if key in model_sd and model_sd[key].shape != value.shape:
            dropped.append(key)
            continue
        filtered[key] = value
    if dropped:
        print(
            f"[WARN] load: dropped {len(dropped)} shape-mismatched keys "
            f"({dropped[0]} ...); those modules keep fresh init"
        )
    return model.load_state_dict(filtered, strict=False)


def infer_policy_head_style(state_dict) -> str:
    """Legacy dense-head checkpoints carry the Linear at policy_head.4."""
    return "dense" if "policy_head.4.weight" in state_dict else "conv"


def net_from_state_dict(state_dict, device="cpu") -> ActorCriticResNet:
    """Build a net whose policy-head architecture matches the checkpoint, so
    old (dense-head) and new (conv-head) checkpoints both load at full fidelity.
    """
    net = ActorCriticResNet(policy_head_style=infer_policy_head_style(state_dict)).to(device)
    load_actor_critic_state_dict(net, state_dict)
    return net
