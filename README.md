# Chess Engine with A2C

This project implements a chess engine that learns to play chess through self-play using an Actor-Critic (A2C) reinforcement learning algorithm. The engine is built in Python and uses the `python-chess` library for board logic and `PyTorch` for the neural network.

## Project Structure

```
/home/teunis/python/ChessEngine/
├───.gitignore
├───helper.py
├───logger.py
├───train.py
├───Visualize.py
├───test_helper.py
├───requirements.txt
├───__pycache__/
├───.git/...
├───conv_white_capture/
├───decent_models/
├───eval_logs/...
├───runs/...
└───venv/
```

- **`train.py`**: The main script for training the chess agent. It contains the A2C training loop, the model definition, and the self-play logic.
- **`helper.py`**: Contains helper functions for converting the chess board to a tensor, encoding and decoding moves, and creating a legal moves mask.
- **`test_helper.py`**: Unit tests for the functions in `helper.py` to ensure the board representation and move encoding/decoding are correct.
- **`Visualize.py`**: A utility for visualizing games in the console.
- **`requirements.txt`**: A list of the Python packages required to run the project.
- **`runs/`**: The directory where TensorBoard logs are stored.
- **`eval_logs/`**: The directory where evaluation logs are stored.
- **`decent_models/`**: A directory to store trained model checkpoints.

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

## Training the Agent

To start the training process, run the `train.py` script:

```bash
python3 train.py
```

This will initialize a new model (or load a pre-trained one if a checkpoint is found) and begin the self-play training loop. Training progress, including losses, rewards, and evaluation metrics, will be logged to the `runs/` directory.

### Monitoring Training

You can monitor the training progress in real-time using TensorBoard. To launch TensorBoard, run the following command in a separate terminal:

```bash
tensorboard --logdir=runs
```

This will start a web server where you can view various training metrics, such as the total loss, actor and critic losses, entropy, and evaluation results.

## Testing

To ensure the core components of the project are working correctly, you can run the unit tests for the helper functions:

```bash
python3 -m unittest test_helper.py
```

These tests verify the board-to-tensor conversion, move encoding/decoding, and legal moves mask generation.
