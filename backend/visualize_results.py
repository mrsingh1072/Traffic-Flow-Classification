
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "backend" / "results" / "evaluation"
RESULTS_DIR = ROOT / "backend" / "results"
OUTPUT_DIR = RESULTS_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_confusion_matrix():
    path = EVAL_DIR / "confusion_matrix.csv"
    matrix = pd.read_csv(path, index_col=0)

    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(matrix.values, interpolation="nearest", aspect="auto")
    fig.colorbar(image, ax=ax, label="Number of records")

    ax.set(
        title="UNSW-NB15: CNN Confusion Matrix",
        xlabel="Predicted class",
        ylabel="Actual class",
    )
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=200)
    plt.close(fig)


def save_class_f1():
    path = EVAL_DIR / "classification_report.csv"
    report = pd.read_csv(path, index_col=0)

    if "f1-score" not in report.columns:
        raise ValueError("The classification report has no f1-score column.")

    class_names = [
        "Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers",
        "Generic", "Normal", "Reconnaissance", "Shellcode", "Worms",
    ]

    available = report.loc[report.index.intersection(class_names)]
    available = available.reindex(
        [name for name in class_names if name in available.index]
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(available.index, available["f1-score"])

    ax.set(
        title="UNSW-NB15: F1-Score by Traffic Class",
        xlabel="Traffic class",
        ylabel="F1-score",
        ylim=(0, 1),
    )
    ax.tick_params(axis="x", rotation=40)

    for bar, value in zip(bars, available["f1-score"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{value:.2f}",
            ha="center",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "class_f1_scores.png", dpi=200)
    plt.close(fig)


def save_label_fraction_comparison():
    path = RESULTS_DIR / "label_fraction_results.csv"
    results = pd.read_csv(path)

    required = {"dataset", "label_fraction", "accuracy", "f1_macro"}
    if not required.issubset(results.columns):
        raise ValueError(
            f"Results CSV must contain: {sorted(required)}"
        )

    data = results[results["dataset"] == "UNSW_NB15"].sort_values(
        "label_fraction"
    )

    if data.empty:
        raise ValueError("No UNSW_NB15 rows found in the results CSV.")

    x = data["label_fraction"] * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, data["accuracy"], marker="o", label="Accuracy")
    ax.plot(x, data["f1_macro"], marker="s", label="Macro F1-score")

    ax.set(
        title="UNSW-NB15: Performance vs. Labeled Training Data",
        xlabel="Percentage of labeled training data (%)",
        ylabel="Score",
        ylim=(0, 1),
        xticks=x,
    )
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "label_fraction_comparison.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    save_confusion_matrix()
    save_class_f1()
    save_label_fraction_comparison()

    print("All evaluation figures generated successfully.")
    print(f"Saved to: {OUTPUT_DIR}")
