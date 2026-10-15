
import json
from pathlib import Path

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
DATA_DIR = ROOT / "datasets" / "ISCX-VPN-nonVPN" / "processed_grouped"
MODEL_PATH = ROOT / "models" / "iscx_cnn_v2_best.pth"
RESULT_PATH = ROOT / "results" / "iscx_cnn_v2_metrics.json"

THRESHOLD = 0.50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


class TrafficCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
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


# Load only the grouped official test set
data = np.load(DATA_DIR / "test_processed.npz")
X_test = data["X"].astype(np.float32)
y_test = data["y"].astype(np.int64)

test_loader = DataLoader(
    TensorDataset(
        torch.from_numpy(np.ascontiguousarray(X_test)).unsqueeze(1),
        torch.from_numpy(y_test),
    ),
    batch_size=512,
    shuffle=False,
)

model = TrafficCNN().to(device)
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device, weights_only=True)
)
model.eval()

all_probabilities = []

with torch.no_grad():
    for features, _ in test_loader:
        features = features.to(device)
        probs = torch.sigmoid(model(features))
        all_probabilities.extend(probs.cpu().numpy())

probabilities = np.asarray(all_probabilities)
predictions = (probabilities >= THRESHOLD).astype(int)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, zero_division=0)
recall = recall_score(y_test, predictions, zero_division=0)
f1 = f1_score(y_test, predictions, zero_division=0)

cm = confusion_matrix(y_test, predictions, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()
fpr = fp / (fp + tn) if (fp + tn) else 0.0

print("\n========== ISCX CNN V2 TEST RESULTS ==========")
print("Threshold:", THRESHOLD)
print(f"Accuracy:            {accuracy:.4f}")
print(f"Precision (VPN):     {precision:.4f}")
print(f"Recall (VPN):        {recall:.4f}")
print(f"F1-score (VPN):      {f1:.4f}")
print(f"False-positive rate: {fpr:.4f}")

print("\nConfusion matrix [[TN, FP], [FN, TP]]:")
print(cm)

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Non-VPN", "VPN"],
        digits=4,
        zero_division=0,
    )
)

metrics = {
    "dataset": "ISCX VPN-nonVPN",
    "model": "CNN V2",
    "evaluation_split": "grouped official test",
    "threshold": THRESHOLD,
    "accuracy": float(accuracy),
    "precision_vpn": float(precision),
    "recall_vpn": float(recall),
    "f1_vpn": float(f1),
    "false_positive_rate": float(fpr),
    "confusion_matrix": cm.tolist(),
}

with open(RESULT_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)

print("\nMetrics saved to:", RESULT_PATH)
