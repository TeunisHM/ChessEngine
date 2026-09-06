import torch
import torch.nn as nn
import torch.nn.functional as F

from helper import ACTION_SPACE_SIZE, BOARD_TENSOR_PLANES

DEFAULT_INPUT_CHANNELS = BOARD_TENSOR_PLANES
DEFAULT_NUM_RESIDUAL_BLOCKS = 8
DEFAULT_NUM_FILTERS = 128
DEFAULT_TRANSFORMER_HEADS = 4
DEFAULT_NUM_ATTENTION_LAYERS = 1     # 1 == legacy: a single layer after the tower

# (dRank, dFile) each in [-7, 7] -> 15*15 = 225 relative-position buckets.
_REL_BUCKETS = 225


def _relative_index() -> torch.Tensor:
    """(64,64) long tensor mapping square pairs to a (dRank, dFile) bucket.

    A full relative-position table subsumes the chess relations that matter --
    same rank (dRank=0), same file (dFile=0), diagonals (|dRank|==|dFile|),
    knight jumps, and distance -- without hand-coding any of them, at 225
    parameters per head.
    """
    ar = torch.arange(64)
    r, f = ar // 8, ar % 8
    dr = r[None, :] - r[:, None] + 7
    df = f[None, :] - f[:, None] + 7
    return (dr * 15 + df).long()

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
                 dim_feedforward=None, dropout=0.0, rel_bias=False):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * num_channels
        self.num_heads = num_heads
        self.pos_embed = nn.Parameter(torch.zeros(64, num_channels))
        if rel_bias:
            # Per-head additive bias on attention logits, indexed by the
            # (dRank, dFile) offset between squares. Gives attention the board
            # geometry directly instead of making it rediscover that a1 and h1
            # share a rank from a free-form pos_embed.
            self.register_buffer("rel_index", _relative_index(), persistent=False)
            self.rel_bias = nn.Parameter(torch.zeros(num_heads, _REL_BUCKETS))
        else:
            self.rel_bias = None
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
        mask = None
        if self.rel_bias is not None:
            # (heads,64,64) -> (B*heads,64,64), the layout MultiheadAttention
            # expects for a 3D float attn_mask (batch-major, head-minor).
            bias = self.rel_bias[:, self.rel_index]
            mask = bias.unsqueeze(0).expand(B, -1, -1, -1).reshape(
                B * self.num_heads, 64, 64)
        if mask is None:
            tokens = self.layer(tokens)
        else:
            # nn.TransformerEncoderLayer's fused kernel returns NaN for a 3D
            # float src_mask on this ROCm build (raw MultiheadAttention with the
            # same mask is fine), so run the pre-LN block explicitly. Uses the
            # layer's own submodules, so parameter names -- and therefore
            # checkpoint compatibility -- are unchanged.
            lyr = self.layer
            h = lyr.norm1(tokens)
            attn, _ = lyr.self_attn(h, h, h, attn_mask=mask, need_weights=False)
            tokens = tokens + lyr.dropout1(attn)
            h = lyr.norm2(tokens)
            ff = lyr.linear2(lyr.dropout(lyr.activation(lyr.linear1(h))))
            tokens = tokens + lyr.dropout2(ff)
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
        num_attention_layers=DEFAULT_NUM_ATTENTION_LAYERS,
        rel_bias=False,
        attn_after_stem=False,
        attn_positions=None,
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

        # Attention is cheap over 64 tokens (~69% of a residual block in
        # compute), so it is interleaved rather than bolted on once at the end:
        # with a single trailing layer the whole tower runs on purely local
        # information. `mid_attn` holds the interleaved layers; `transformer`
        # stays the final one so pre-v28 checkpoints keep loading unchanged.
        n_mid = max(0, int(num_attention_layers) - 1)
        self.mid_attn = nn.ModuleList([
            BoardTransformerLayer(num_filters, num_heads=transformer_heads,
                                  rel_bias=rel_bias)
            for _ in range(n_mid)
        ])
        # Placement, in "after this many residual blocks"; 0 == straight after
        # the stem, so attention sees raw per-square piece identity before any
        # convolution has mixed it. Saved as a buffer so the exact layout is
        # recoverable from the checkpoint rather than re-derived by rule.
        if attn_positions is not None:
            self._attn_after = [int(v) for v in attn_positions]
        elif attn_after_stem and n_mid > 0:
            rest = n_mid - 1
            self._attn_after = [0] + [
                round((j + 1) * num_residual_blocks / (rest + 1)) for j in range(rest)
            ]
        else:
            self._attn_after = [
                round((j + 1) * num_residual_blocks / (n_mid + 1)) for j in range(n_mid)
            ]
        self._validate_attn_positions(self._attn_after)
        self.register_buffer("attn_positions",
                             torch.tensor(self._attn_after, dtype=torch.long),
                             persistent=True)
        # num_attention_layers=0 removes attention entirely (pure conv tower),
        # the control arm for "does attention earn its place at all".
        self.transformer = (
            BoardTransformerLayer(num_filters, num_heads=transformer_heads,
                                  rel_bias=rel_bias)
            if int(num_attention_layers) > 0 else None
        )

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

    def _validate_attn_positions(self, positions):
        if len(positions) != len(self.mid_attn):
            raise ValueError(
                f"expected {len(self.mid_attn)} interleaved attention positions, "
                f"got {len(positions)}"
            )
        n_blocks = len(self.residual_tower)
        if any(p < 0 or p > n_blocks for p in positions):
            raise ValueError(
                f"attention positions must be in [0, {n_blocks}], got {positions}"
            )
        if positions != sorted(set(positions)):
            raise ValueError(
                f"attention positions must be strictly increasing, got {positions}"
            )

    def load_state_dict(self, state_dict, *args, **kwargs):
        result = super().load_state_dict(state_dict, *args, **kwargs)
        # forward() uses this cached Python list to avoid a GPU->CPU sync on
        # every call. Refresh it after any direct state-dict load so the saved
        # placement buffer remains the single source of truth.
        positions = [int(v) for v in self.attn_positions.detach().cpu().tolist()]
        self._validate_attn_positions(positions)
        self._attn_after = positions
        return result

    def forward(self, x, with_wdl=False, wdl_detach=True):
        out = self.stem(x)
        if self.mid_attn:
            ai = 0
            if self._attn_after and self._attn_after[0] == 0:
                out = self.mid_attn[0](out)      # raw piece info, pre-convolution
                ai = 1
            for i, blk in enumerate(self.residual_tower):
                out = blk(out)
                if ai < len(self.mid_attn) and (i + 1) == self._attn_after[ai]:
                    out = self.mid_attn[ai](out)
                    ai += 1
        else:
            out = self.residual_tower(out)
        if self.transformer is not None:
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


def infer_num_attention_layers(state_dict) -> int:
    """Interleaved `mid_attn.N.*` layers plus the trailing one, if present.

    Returns 0 for a pure-conv checkpoint (no attention at all).
    """
    idx = {int(k.split(".")[1]) for k in state_dict if k.startswith("mid_attn.")}
    has_final = any(k.startswith("transformer.") for k in state_dict)
    return len(idx) + (1 if has_final else 0)


def infer_rel_bias(state_dict) -> bool:
    return any(k.endswith(".rel_bias") for k in state_dict)


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
        num_attention_layers=infer_num_attention_layers(state_dict),
        rel_bias=infer_rel_bias(state_dict),
        attn_positions=(state_dict["attn_positions"].tolist()
                        if "attn_positions" in state_dict else None),
    ).to(device)
    load_actor_critic_state_dict(net, state_dict)
    return net
