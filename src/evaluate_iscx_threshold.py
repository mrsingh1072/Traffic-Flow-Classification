
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "datasets" / "ISCX-VPN-nonVPN" / "processed"
MODEL_PATH = ROOT / "models" / "iscx_cnn_best.pth"
RESULT_PATH = ROOT / "results" / "iscx_cnn_threshold_metrics.json"

THRESHOLD = 0.60

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


# Load the untouched official test set
data = np.load(DATA_DIR / "test_processed.npz")
X_test = data["X"].astype(np.float32)
y_test = data["y"].astype(np.int64)

print("Test features:", X_test.shape)
print("Test labels:", y_test.shape)
print("Threshold:", THRESHOLD)

test_loader = DataLoader(
    TensorDataset(
        torch.from_numpy(np.ascontiguousarray(X_test)).unsqueeze(1),
        torch.from_numpy(y_test),
    ),
    batch_size=512,
    shuffle=False,
)

# Load the previously trained model; do not retrain
model = TrafficCNN().to(device)
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device, weights_only=True)
)
model.eval()

all_probabilities = []

with torch.no_grad():
    for features, _ in test_loader:
        features = features.to(device)
        probabilities = torch.sigmoid(model(features))
        all_probabilities.extend(probabilities.cpu().numpy())

probabilities = np.asarray(all_probabilities)
predictions = (probabilities >= THRESHOLD).astype(int)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, zero_division=0)
recall = recall_score(y_test, predictions, zero_division=0)
f1 = f1_score(y_test, predictions, zero_division=0)

tn, fp, fn, tp = confusion_matrix(
    y_test, predictions, labels=[0, 1]
).ravel()

false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0

print("\nISCX OFFICIAL TEST RESULTS")
print("=" * 50)
print(f"Threshold:          {THRESHOLD:.2f}")
print(f"Accuracy:           {accuracy:.4f}")
print(f"Precision (VPN):    {precision:.4f}")
print(f"Recall (VPN):       {recall:.4f}")
print(f"F1-score (VPN):     {f1:.4f}")
print(f"False-positive rate:{false_positive_rate:.4f}")

print("\nConfusion Matrix [[TN, FP], [FN, TP]]:")
print([[int(tn), int(fp)], [int(fn), int(tp)]])

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Non-VPN", "VPN"],
        zero_division=0,
    )
)

metrics = {
    "dataset": "ISCX VPN-nonVPN",
    "evaluation_split": "official_test",
    "threshold": THRESHOLD,
    "accuracy": float(accuracy),
    "precision_vpn": float(precision),
    "recall_vpn": float(recall),
    "f1_vpn": float(f1),
    "false_positive_rate": float(false_positive_rate),
    "confusion_matrix": [
        [int(tn), int(fp)],
        [int(fn), int(tp)],
    ],
}

RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(RESULT_PATH, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

print("\nMetrics saved to:", RESULT_PATH)
