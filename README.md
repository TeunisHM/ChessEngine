# Chess Engine with PPO and Lookahead Search

This project implements a chess engine trained with PPO, supervised PGN pretraining,
checkpoint/engine opponents, and value-guided quiescence search. It uses
`python-chess` for board logic and PyTorch for the policy/value network.

## Project Structure

- **`train.py`**: PPO training, rollout generation, and diagnostics.
- **`models.py`**: Policy/value network definitions and checkpoint-compatible loading.
- **`lookahead.py`**: Widened policy-candidate lookahead and quiescence search.
- **`helper.py`**: Contains helper functions for converting the chess board to a tensor, encoding and decoding moves, and creating a legal moves mask.
- **`test_helper.py`**: Unit tests for the functions in `helper.py` to ensure the board representation and move encoding/decoding are correct.
- **`requirements.txt`**: A list of the Python packages required to run the project.
- **`logs/`**: Evaluation, console, and H2H logs.
- **`models/`**: Supervised seeds and PPO checkpoints.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

## Training

On the Radeon 8060S, use the isolated `venv-rocm` environment. The entry point
selects MIOpen `FAST` fallback and keeps AMP off because FP16 gradients are not
finite on the current gfx1151 stack. Specify the initializer and run identity
explicitly:

```bash
MIOPEN_FIND_MODE=FAST venv-rocm/bin/python train.py \
    --init-from models/ppo_search_v13_ppo_control_seed1301_checkpoint_99.pt \
    --model-name ppo_search_v14 \
    --num-batches 400 \
    --eval-interval 50 \
    --seed 1401
```

This loads model weights but starts a fresh optimizer and 400-batch cosine
scheduler. The normal entrypoint uses raw-policy PPO rollouts, no distillation,
and FP32. Use a distinct model name for every phase.

## Completed v13 Experiment

The pure-PPO control beat v11@399 70W/29D/2L and beat the search-training arm
82W/16D/3L. The search arm lost to v11@399 17W/45D/39L. Both sides used the
same lookahead search during H2H evaluation; "pure PPO" refers only to rollout
generation. See `logs/v13_matched_experiment_final_report_20260715.md` for the
full analysis.

## Supervised Pretraining from PGNs

You can optionally warm-start the model on real games before reinforcement learning by running `pretrain_from_pgn.py`. Point it at one or more PGN files and it will optimize the policy head to mimic the human moves while teaching the value head to predict the eventual outcome from each position.

```bash
python pretrain_from_pgn.py --pgn /path/to/games.pgn --max-games 5000 --epochs 5 --output-model pretrained.pt
```

Supply a resulting checkpoint to `train.py` with `--init-from` when starting a
new run.

### Monitoring Training

Training prints rollout, PPO, and opponent outcome diagnostics to the console.
Periodic evaluation summaries are written to CSV files under `logs/`.

## Testing

To ensure the core components of the project are working correctly, you can run the unit tests for the helper functions:

```bash
python -m unittest test_helper.py
```

These tests verify board/move encoding and legal masks.
