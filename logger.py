import json
import os
import glob
from statistics import mean
import matplotlib.pyplot as plt

def load_logs(log_dir="eval_logs2"):
    files = sorted(glob.glob(os.path.join(log_dir, "*.json")))
    logs = []
    for file in files:
        with open(file, "r") as f:
            logs.append(json.load(f))
    return logs

def plot_metric(logs, key, label=None, ax=None, **kwargs):
    values = [log[key] for log in logs]
    plt.plot(values, label=label or key, **kwargs)

def main(model_name):
    import matplotlib
    matplotlib.use("Agg")  # Use non-GUI backend (no X required)
    logs = load_logs('eval_logs/'+model_name)
    if not logs:
        print("No logs found.")
        return

    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))
    plot_metric(logs, "wins", label="Wins", ax=ax1)
    plot_metric(logs, "losses", label="Losses", ax=ax1)
    #plot_metric(logs, "avg_game_length", label="Avg Game Length")
    ax2 = ax1.twinx()
    plot_metric(logs, "avg_entropy", label="Avg Entropy", ax=ax2, color='red')
    # Titles and labels
    ax1.set_title("Evaluation Over Time")
    ax1.set_xlabel("Evaluation Step")
    ax1.set_ylabel("Wins / Losses")  # Primary y-axis
    ax2.set_ylabel("Avg Entropy")    # Secondary y-axis
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    # Style
    ax1.grid(True)
    fig.tight_layout()
    plt.savefig(f"eval_plot_{model_name}.png")

if __name__ == "__main__":
    model_name = "conv_white_capture"
    main(model_name)
