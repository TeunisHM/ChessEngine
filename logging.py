import json
import os
import glob
from statistics import mean
import matplotlib.pyplot as plt

def load_logs(log_dir="eval_logs"):
    files = sorted(glob.glob(os.path.join(log_dir, "*.json")))
    logs = []
    for file in files:
        with open(file, "r") as f:
            logs.append(json.load(f))
    return logs

def plot_metric(logs, key, label=None):
    values = [log[key] for log in logs]
    plt.plot(values, label=label or key)

def main():
    logs = load_logs()
    if not logs:
        print("No logs found.")
        return

    # Plotting
    plt.figure(figsize=(10, 6))
    plot_metric(logs, "policy_white_wins", label="White Wins (Policy)")
    plot_metric(logs, "policy_black_wins", label="Black Wins (Policy)")
    plot_metric(logs, "avg_game_length", label="Avg Game Length")
    plot_metric(logs, "avg_entropy", label="Avg Entropy")
    plt.title("Evaluation Over Time")
    plt.xlabel("Evaluation Step")
    plt.ylabel("Metric")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
