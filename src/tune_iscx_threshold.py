
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "datasets" / "ISCX-VPN-nonVPN" / "processed"
MODEL_PATH = ROOT / "models" / "iscx_cnn_best.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


class TrafficCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.30),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x)).squeeze(1)


data = np.load(DATA_DIR / "train_processed.npz")
X = data["X"].astype(np.float32)
y = data["y"].astype(np.int64)

_, X_val, _, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

val_loader = DataLoader(
    TensorDataset(
        torch.from_numpy(np.ascontiguousarray(X_val)).unsqueeze(1),
        torch.from_numpy(y_val),
    ),
    batch_size=512,
    shuffle=False,
)

model = TrafficCNN().to(device)
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device, weights_only=True)
)
model.eval()

probabilities = []

with torch.no_grad():
    for features, _ in val_loader:
        features = features.to(device)
        probs = torch.sigmoid(model(features))
        probabilities.extend(probs.cpu().numpy())

probabilities = np.asarray(probabilities)

print("\nISCX VALIDATION THRESHOLD COMPARISON")
print("Threshold | Precision | Recall | F1-score | False-positive rate")
print("-" * 70)

results = []

for threshold in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val, predictions, labels=[0, 1]
    ).ravel()

    precision = precision_score(
        y_val, predictions, zero_division=0
    )
    recall = recall_score(
        y_val, predictions, zero_division=0
    )
    f1 = f1_score(y_val, predictions, zero_division=0)
    fpr = fp / (fp + tn)

    results.append({
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
    })

    print(
        f"{threshold:.2f}      | {precision:.4f}    | "
        f"{recall:.4f} | {f1:.4f}    | {fpr:.4f}"
    )

best = max(results, key=lambda row: row["f1"])

print("\nBest threshold by validation F1:", best["threshold"])
print("Best validation F1:", round(best["f1"], 4))
