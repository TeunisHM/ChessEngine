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


class SqueezeExcitation(nn.Module):
    """Channel-wise attention conditioned on a global board summary.

    Squeeze: per-channel spatial mean (a global census of each feature).
    Excitation: bottleneck MLP -> sigmoid gates in [0,1], one per channel.
    The block's feature maps are rescaled by these gates, letting global
    facts (material balance, king danger) modulate local conv features.
    """

    def __init__(self, num_channels, reduction=8):
        super().__init__()
        hidden = max(1, num_channels // reduction)
        self.fc1 = nn.Linear(num_channels, hidden)
        self.fc2 = nn.Linear(hidden, num_channels)

    def forward(self, x):
        s = x.mean(dim=(2, 3))
        g = torch.sigmoid(self.fc2(F.relu(self.fc1(s))))
        return x * g.unsqueeze(-1).unsqueeze(-1)


class ResidualBlock(nn.Module):
    """A standard residual block for a ResNet, optionally with SE gating."""

    def __init__(self, num_channels, use_se=False):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = _norm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = _norm2d(num_channels)
        self.se = SqueezeExcitation(num_channels) if use_se else None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.se is not None:
            out = self.se(out)
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
        use_se=False,
    ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(num_input_channels, num_filters, kernel_size=3, padding=1),
            _norm2d(num_filters),
            nn.ReLU(),
        )

        self.residual_tower = nn.Sequential(
            *[ResidualBlock(num_filters, use_se=use_se) for _ in range(num_residual_blocks)]
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

        # Separate WDL evaluator head with its own conv adapter. Trained on
        # objective game outcome (win/draw/loss) with the shared trunk DETACHED,
        # so it gives search a calibrated zero-sum evaluator without dragging the
        # policy — unlike supervising the shared value scalar, which backprops
        # into the trunk and destabilises PPO.
        self.wdl_head = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            _norm2d(num_filters),
            nn.ReLU(),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            _norm2d(num_filters),
            nn.ReLU(),
            nn.Conv2d(num_filters, 1, kernel_size=1),
            _norm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 3),          # logits: [win, draw, loss], mover POV
        )

    def forward(self, x, with_wdl=False, wdl_detach=True):
        out = self.stem(x)
        out = self.residual_tower(out)
        out = self.transformer(out)
        policy_logits = self.policy_head(out)
        state_value = self.value_head(out)
        if with_wdl:
            # Detach by default: WDL loss must not update the shared trunk/policy.
            feat = out.detach() if wdl_detach else out
            return policy_logits, state_value, self.wdl_head(feat)
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


def infer_use_se(state_dict) -> bool:
    return any(".se." in key for key in state_dict)


def infer_num_filters(state_dict) -> int:
    return state_dict["stem.0.weight"].shape[0]


def infer_num_residual_blocks(state_dict) -> int:
    prefix = "residual_tower."
    indices = set()
    for key in state_dict:
        if key.startswith(prefix) and key.endswith(".conv1.weight"):
            indices.add(int(key[len(prefix):].split(".")[0]))
    return max(indices) + 1 if indices else DEFAULT_NUM_RESIDUAL_BLOCKS


def net_from_state_dict(state_dict, device="cpu") -> ActorCriticResNet:
    """Build a net whose architecture matches the checkpoint (policy-head style,
    SE blocks, filter width, and residual depth all auto-detected), so
    checkpoints from every generation and every model size load at full
    fidelity and can safely mix in the same opponent pool.
    """
    net = ActorCriticResNet(
        policy_head_style=infer_policy_head_style(state_dict),
        use_se=infer_use_se(state_dict),
        num_filters=infer_num_filters(state_dict),
        num_residual_blocks=infer_num_residual_blocks(state_dict),
    ).to(device)
    load_actor_critic_state_dict(net, state_dict)
    return net
