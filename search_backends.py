"""Single seam for choosing which search turns a position into a move.

Both backends return the same 10-tuple
    (action_idx, log_pi, masks, root_values, states,
     log_b_chosen, kl_b_pi, pick_rank, topk_idx, log_b_topk)
so call sites differ by one argument, not by a branch.

    quiescence : lookahead.select_moves_with_lookahead -- top-k(pi) widened with
                 captures/checks, scored by batched alpha-beta quiescence.
                 The default; every result before 2026-09 was produced with it.
    gumbel     : gumbel_search.select_moves_with_gumbel -- Gumbel AlphaZero
                 (Danihelka et al. 2022), fixed simulation budget.
"""
from typing import List

import chess

from gumbel_search import C_SCALE, C_VISIT, select_moves_with_gumbel
from lookahead import select_moves_with_lookahead

BACKENDS = ("quiescence", "gumbel")
DEFAULT_BACKEND = "quiescence"


def add_search_args(parser):
    """Attach the backend switch and the gumbel knobs to an argparse parser.

    The quiescence knobs (--lookahead-k, --max-qdepth, ...) stay where they are;
    this only adds what is new, so existing command lines keep working.
    """
    parser.add_argument(
        "--search-backend", choices=BACKENDS, default=DEFAULT_BACKEND,
        help="Which search produces moves. 'quiescence' is the historical "
             "default and reproduces every pre-2026-09 result.",
    )
    parser.add_argument(
        "--opponent-search-backend", choices=("same",) + BACKENDS, default="same",
        help="Search used by frozen checkpoint opponents. 'same' follows "
             "--search-backend (historical behaviour). Pin it to 'quiescence' to "
             "change only the trainee's search and hold the curriculum fixed.",
    )
    parser.add_argument(
        "--gumbel-m", type=int, default=16,
        help="Gumbel AZ: root actions sampled without replacement (paper's m).",
    )
    parser.add_argument(
        "--gumbel-sims", type=int, default=32,
        help="Gumbel AZ: simulation budget per move (paper's n). Net evaluations "
             "per move are gumbel_sims + 1, independent of --gumbel-m.",
    )
    parser.add_argument(
        "--gumbel-c-visit", type=float, default=C_VISIT,
        help="Gumbel AZ: c_visit in sigma(q) = (c_visit + max N) * c_scale * q.",
    )
    parser.add_argument(
        "--gumbel-c-scale", type=float, default=C_SCALE,
        help="Gumbel AZ: c_scale in sigma(q).",
    )
    return parser


def search_config(args) -> dict:
    """Pull the backend settings off a parsed argparse namespace."""
    return {
        "backend": getattr(args, "search_backend", DEFAULT_BACKEND),
        "gumbel_m": getattr(args, "gumbel_m", 16),
        "gumbel_sims": getattr(args, "gumbel_sims", 32),
        "c_visit": getattr(args, "gumbel_c_visit", C_VISIT),
        "c_scale": getattr(args, "gumbel_c_scale", C_SCALE),
    }


def opponent_search_config(args) -> dict:
    """Search settings for frozen checkpoint opponents. Defaults to the
    trainee's, so an unspecified run behaves exactly as before."""
    cfg = search_config(args)
    choice = getattr(args, "opponent_search_backend", "same")
    if choice != "same":
        cfg["backend"] = choice
    return cfg


def select_moves(
    net,
    boards: List[chess.Board],
    device,
    *,
    backend: str = DEFAULT_BACKEND,
    temperature: float = 0.0,
    value_weight: float = 1.0,
    use_wdl: bool = False,
    # quiescence knobs
    top_k: int = 8,
    alpha: float = 0.33,
    max_qdepth: int = 2,
    check_budget: int = 1,
    # gumbel knobs
    gumbel_m: int = 16,
    gumbel_sims: int = 32,
    c_visit: float = C_VISIT,
    c_scale: float = C_SCALE,
):
    if backend == "gumbel":
        return select_moves_with_gumbel(
            net, boards, device, m=gumbel_m, sims=gumbel_sims,
            temperature=temperature, c_visit=c_visit, c_scale=c_scale,
            value_weight=value_weight, use_wdl=use_wdl,
        )
    if backend != "quiescence":
        raise ValueError(f"unknown search backend {backend!r}; expected one of {BACKENDS}")
    return select_moves_with_lookahead(
        net, boards, device, top_k=top_k, alpha=alpha, temperature=temperature,
        max_qdepth=max_qdepth, check_budget=check_budget,
        value_weight=value_weight, use_wdl=use_wdl,
    )
