
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "backend" / "models" / "UNSW_NB15_100pct" / "model.pt"
PREPROCESSOR_PATH = ROOT / "datasets" / "processed" / "UNSW_NB15" / "preprocessor.joblib"
RESULTS_DIR = ROOT / "backend" / "results"
OUTPUT_DIR = RESULTS_DIR / "evaluation"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    artifact = joblib.load(PREPROCESSOR_PATH)
    preprocessor = artifact["preprocessor"]
    feature_columns = list(artifact["feature_columns"])
    classes = [str(c) for c in artifact["classes"]]

    numeric_columns = []
    categorical_columns = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == "numeric":
            numeric_columns = list(columns)
        elif name == "categorical":
            categorical_columns = list(columns)

    checkpoint = torch.load(
        MODEL_PATH, map_location=DEVICE, weights_only=False
    )

    model = TrafficCNN(
        int(checkpoint["num_features"]),
        int(checkpoint["num_classes"]),
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    csv_path = (
        ROOT / "datasets" / "CSV Files" /
        "Training and Testing Sets" / "UNSW_NB15_testing-set.csv"
    )

    if not csv_path.exists():
        raise FileNotFoundError(f"Test dataset not found: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)

    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required input columns: {missing}")

    if "attack_cat" not in df.columns:
        raise ValueError("The test CSV does not contain attack_cat labels.")

    X = df[feature_columns].copy()

    for column in numeric_columns:
        X[column] = pd.to_numeric(X[column], errors="coerce")

    for column in categorical_columns:
        values = X[column].astype("string")
        X[column] = values.astype(object).where(values.notna(), np.nan)

    transformed = np.asarray(
        preprocessor.transform(X), dtype=np.float32
    )

    if not np.isfinite(transformed).all():
        raise ValueError("Transformed test features contain non-finite values.")

    y_true = df["attack_cat"].astype(str).to_numpy()
    class_to_id = {name: i for i, name in enumerate(classes)}

    unknown_labels = sorted(set(y_true) - set(class_to_id))
    if unknown_labels:
        raise ValueError(f"Labels absent from model classes: {unknown_labels}")

    y_true_ids = np.array([class_to_id[label] for label in y_true])

    batch_size = 512
    predictions = []
    probabilities = []

    with torch.no_grad():
        for start in range(0, len(transformed), batch_size):
            batch = transformed[start:start + batch_size]
            tensor = torch.from_numpy(batch).unsqueeze(-1).to(DEVICE)

            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

            probabilities.append(probs)
            predictions.extend(np.argmax(probs, axis=1).tolist())

    y_pred_ids = np.asarray(predictions)
    y_pred = [classes[i] for i in y_pred_ids]
    all_probabilities = np.vstack(probabilities)

    metrics = {
        "dataset": "UNSW_NB15",
        "model": "TrafficCNN-1D",
        "test_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true_ids, y_pred_ids)),
        "precision_macro": float(
            precision_score(y_true_ids, y_pred_ids, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true_ids, y_pred_ids, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true_ids, y_pred_ids, average="macro", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true_ids, y_pred_ids, average="weighted", zero_division=0)
        ),
    }

    report = classification_report(
        y_true_ids,
        y_pred_ids,
        labels=list(range(len(classes))),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true_ids, y_pred_ids, labels=list(range(len(classes)))
    )

    pd.DataFrame(
        matrix, index=classes, columns=classes
    ).to_csv(OUTPUT_DIR / "confusion_matrix.csv")

    pd.DataFrame(report).transpose().to_csv(
        OUTPUT_DIR / "classification_report.csv"
    )

    pd.DataFrame({
        "actual_class": y_true,
        "predicted_class": y_pred,
        "confidence": all_probabilities.max(axis=1),
    }).to_csv(OUTPUT_DIR / "predictions.csv", index=False)

    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print("\n========== TEST SET EVALUATION ==========")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:20s}: {value:.4f}")
        else:
            print(f"{key:20s}: {value}")

    print(f"\nSaved evaluation files to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
