"""Does a search configuration actually play better chess than no search?

Cheap, mandatory gate before committing GPU to a training run with a new search.
Runs each configuration through the validated evaluate_vs_engine.py harness on
the same weights and the same opening book, then reports scores with standard
errors and flags anything that fails to beat the raw policy.

The lesson this encodes: a search can pass every contract and cost test and
still play worse than not searching at all. Gumbel m=16/n=32 at the paper's
default c_visit=50 scored 0.062 against the quiescence search's 0.422 on
identical weights, because sigma's ~50x multiplier hands the move choice
entirely to an unvalidated value head.

    venv-rocm/bin/python search_strength_gate.py --model models/seed.pt
"""
import argparse
import math
import re
import subprocess
import sys

# label -> extra evaluate_vs_engine.py flags
DEFAULT_ARMS = [
    ("raw policy",                 ["--raw"]),
    ("quiescence k=4 a=1.0",       ["--lookahead-k", "4", "--lookahead-alpha", "1.0"]),
    ("gumbel n=32 c_visit=50",     ["--search-backend", "gumbel", "--gumbel-m", "16",
                                    "--gumbel-sims", "32", "--gumbel-c-visit", "50"]),
    ("gumbel n=32 c_visit=2",      ["--search-backend", "gumbel", "--gumbel-m", "16",
                                    "--gumbel-sims", "32", "--gumbel-c-visit", "2"]),
    ("gumbel n=32 c_visit=0.5",    ["--search-backend", "gumbel", "--gumbel-m", "16",
                                    "--gumbel-sims", "32", "--gumbel-c-visit", "0.5"]),
    ("gumbel n=64 c_visit=1",      ["--search-backend", "gumbel", "--gumbel-m", "16",
                                    "--gumbel-sims", "64", "--gumbel-c-visit", "1"]),
    ("gumbel n=64 c_visit=0.5",    ["--search-backend", "gumbel", "--gumbel-m", "16",
                                    "--gumbel-sims", "64", "--gumbel-c-visit", "0.5"]),
    ("gumbel n=128 c_visit=0.5",   ["--search-backend", "gumbel", "--gumbel-m", "16",
                                    "--gumbel-sims", "128", "--gumbel-c-visit", "0.5"]),
]

# Narrowed set for a high-power re-run once the sweep has thinned the field.
# 64 games leaves ~0.06 SE, which cannot separate arms a few points apart.
FINALIST_ARMS = [
    ("raw policy",               ["--raw"]),
    ("quiescence k=4 a=1.0",     ["--lookahead-k", "4", "--lookahead-alpha", "1.0"]),
    ("gumbel n=32 c_visit=0.5",  ["--search-backend", "gumbel", "--gumbel-m", "16",
                                  "--gumbel-sims", "32", "--gumbel-c-visit", "0.5"]),
    ("gumbel n=64 c_visit=1",    ["--search-backend", "gumbel", "--gumbel-m", "16",
                                  "--gumbel-sims", "64", "--gumbel-c-visit", "1"]),
    ("gumbel n=64 c_visit=0.5",  ["--search-backend", "gumbel", "--gumbel-m", "16",
                                  "--gumbel-sims", "64", "--gumbel-c-visit", "0.5"]),
    ("gumbel n=128 c_visit=1",   ["--search-backend", "gumbel", "--gumbel-m", "16",
                                  "--gumbel-sims", "128", "--gumbel-c-visit", "1"]),
]

PRESETS = {"sweep": DEFAULT_ARMS, "finalists": FINALIST_ARMS}

_RESULT = re.compile(r"Wins:\s*(\d+)\s*\|\s*Draws:\s*(\d+)\s*\|\s*Losses:\s*(\d+)")


def run_arm(python, model, flags, games, skill, move_time):
    cmd = [python, "-u", "evaluate_vs_engine.py", "--model", model,
           "--games", str(games), "--engine-skill-level", str(skill),
           "--engine-move-time", str(move_time), "--paired-openings"] + flags
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    m = None
    for m in _RESULT.finditer(out):
        pass
    if m is None:
        return None
    w, d, l = (int(x) for x in m.groups())
    n = w + d + l
    return (w, d, l, n, (w + 0.5 * d) / max(1, n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--games", type=int, default=64)
    ap.add_argument("--engine-skill", type=int, default=0)
    ap.add_argument("--engine-move-time", type=float, default=0.01)
    ap.add_argument("--preset", choices=sorted(PRESETS), default="sweep",
                    help="'sweep' scans c_visit/budget broadly; 'finalists' re-runs "
                         "the survivors, meant to be paired with a larger --games.")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    print(f"model: {args.model}")
    print(f"protocol: SF skill {args.engine_skill}, {args.engine_move_time}s/move, "
          f"{args.games} paired-opening games per arm | preset={args.preset}\n")
    print(f"{'arm':<26} {'W':>4} {'D':>4} {'L':>4} {'score':>8} {'+/-SE':>7}  verdict")

    baseline = None
    for label, flags in PRESETS[args.preset]:
        res = run_arm(args.python, args.model, flags, args.games,
                      args.engine_skill, args.engine_move_time)
        if res is None:
            print(f"{label:<26} {'FAILED TO PARSE RESULT':>40}")
            continue
        w, d, l, n, score = res
        se = math.sqrt(max(score * (1 - score), 1e-9) / max(1, n))
        if baseline is None:
            baseline = (score, se)
            verdict = "baseline (no search)"
        else:
            z = (score - baseline[0]) / math.sqrt(se ** 2 + baseline[1] ** 2)
            verdict = (f"z={z:+.2f} vs raw  "
                       + ("PASS" if z > 1.0 else "FAIL - no better than not searching"))
        print(f"{label:<26} {w:>4} {d:>4} {l:>4} {score:>8.3f} {se:>7.3f}  {verdict}", flush=True)

    print("\nA search arm should not enter a training run unless it clearly beats raw policy.")


if __name__ == "__main__":
    main()
