
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "datasets" / "processed"
MODEL_DIR = ROOT / "backend" / "models"
OUTPUT_DIR = ROOT / "backend" / "results" / "all_datasets_evaluation"

DATASETS = [
    "UNSW_NB15",
    "ISCX_15s",
    "ISCX_30s",
    "ISCX_60s",
    "ISCX_120s",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 512


class TrafficCNN(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, ceil_mode=True),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.30),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.net(x.transpose(1, 2))


def evaluate_dataset(name):
    data_path = DATA_DIR / name / "data.npz"
    model_path = MODEL_DIR / f"{name}_100pct" / "model.pt"

    if not data_path.exists():
        raise FileNotFoundError(f"Processed data missing: {data_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint missing: {model_path}")

    with np.load(data_path, allow_pickle=True) as data:
        X_test = np.asarray(data["X_test"], dtype=np.float32)
        y_test = np.asarray(data["y_test"]).reshape(-1)
        classes = [str(c) for c in data["classes"].tolist()]

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=False,
    )

    checkpoint_classes = [str(c) for c in checkpoint["classes"]]
    if classes != checkpoint_classes:
        raise ValueError(f"Class order mismatch for {name}")

    if X_test.ndim != 3 or X_test.shape[1] != checkpoint["num_features"]:
        raise ValueError(
            f"Unexpected test feature shape for {name}: {X_test.shape}"
        )

    model = TrafficCNN(
        int(checkpoint["num_features"]),
        int(checkpoint["num_classes"]),
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    predictions = []

    with torch.no_grad():
        for start in range(0, len(X_test), BATCH_SIZE):
            batch = torch.from_numpy(
                X_test[start:start + BATCH_SIZE]
            ).to(DEVICE)

            logits = model(batch)
            predictions.extend(
                logits.argmax(dim=1).cpu().numpy().tolist()
            )

    y_pred = np.asarray(predictions)
    y_true = y_test.astype(int)

    if len(y_true) != len(y_pred):
        raise ValueError(f"Prediction count mismatch for {name}")

    labels = list(range(len(classes)))

    metrics = {
        "dataset": name,
        "test_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(
                y_true, y_pred, labels=labels,
                average="macro", zero_division=0
            )
        ),
        "recall_macro": float(
            recall_score(
                y_true, y_pred, labels=labels,
                average="macro", zero_division=0
            )
        ),
        "f1_macro": float(
            f1_score(
                y_true, y_pred, labels=labels,
                average="macro", zero_division=0
            )
        ),
        "f1_weighted": float(
            f1_score(
                y_true, y_pred, labels=labels,
                average="weighted", zero_division=0
            )
        ),
    }

    output_dir = OUTPUT_DIR / name
    output_dir.mkdir(parents=True, exist_ok=True)

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(report).transpose().to_csv(
        output_dir / "classification_report.csv"
    )

    pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=labels),
        index=classes,
        columns=classes,
    ).to_csv(output_dir / "confusion_matrix.csv")

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    return metrics


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for name in DATASETS:
        try:
            metrics = evaluate_dataset(name)
            results.append(metrics)

            print(
                f"PASS | {name:12s} | "
                f"Accuracy: {metrics['accuracy']:.4f} | "
                f"Macro F1: {metrics['f1_macro']:.4f} | "
                f"Test samples: {metrics['test_samples']}"
            )
        except Exception as exc:
            print(f"FAIL | {name} | {exc}")
            raise

    summary = pd.DataFrame(results)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)

    print("\n========== ALL DATASET EVALUATION ==========")
    print(summary.to_string(index=False))
    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
