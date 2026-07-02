import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent
OUT_PATH = RESULTS_DIR / "metrics.png"


def load_series(path, x_key, y_key):
    series = defaultdict(lambda: ([], []))  # connector -> (x, y)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            x, y = series[row["connector"]]
            x.append(int(row[x_key]))
            y.append(float(row[y_key]))
    return series


def load_efficiency(path):
    names, values = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            names.append(row["connector"])
            values.append(float(row["anls_per_million_params"]))
    return names, values


def main():
    train_loss = load_series(RESULTS_DIR / "train_loss.csv", "step", "loss")
    val_loss = load_series(RESULTS_DIR / "val_loss.csv", "epoch", "loss")
    val_anls = load_series(RESULTS_DIR / "val_anls.csv", "epoch", "anls")
    eff_names, eff_values = load_efficiency(RESULTS_DIR / "efficiency.csv")

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))

    for c, (steps, loss) in train_loss.items():
        axes[0].plot(steps, loss, label=c, alpha=0.8)
    axes[0].set(title="Training loss", xlabel="step", ylabel="loss")

    for c, (epochs, loss) in val_loss.items():
        axes[1].plot(epochs, loss, marker="o", label=c)
    axes[1].set(title="Validation loss", xlabel="epoch", ylabel="loss")

    for c, (epochs, score) in val_anls.items():
        axes[2].plot(epochs, score, marker="o", label=c)
    axes[2].set(title="Validation ANLS", xlabel="epoch", ylabel="ANLS")

    axes[3].bar(eff_names, eff_values, color=["tab:blue", "tab:orange", "tab:green"])
    axes[3].set(
        title="Efficiency: ANLS per\nmillion trainable params",
        xlabel="connector",
        ylabel="ANLS / 1M params",
    )

    for ax in axes[:3]:
        ax.legend()
        ax.grid(True, alpha=0.3)
    axes[3].grid(True, alpha=0.3, axis="y")

    fig.suptitle("VLM fusion mechanism ablation")
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
