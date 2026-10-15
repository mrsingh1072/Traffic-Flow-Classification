
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
DATA_DIR = ROOT / "datasets" / "UNSW-NB15" / "processed"
MODEL_PATH = ROOT / "models" / "unsw_cnn_best.pth"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.60
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


# Load official test data
data = np.load(DATA_DIR / "test_processed.npz")
X_test = data["X"].astype(np.float32)
y_test = data["y"].astype(np.int64)

test_loader = DataLoader(
    TensorDataset(
        torch.from_numpy(np.ascontiguousarray(X_test)).unsqueeze(1),
        torch.from_numpy(y_test),
    ),
    batch_size=1024,
    shuffle=False,
)

# Load saved checkpoint
model = TrafficCNN().to(device)
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device, weights_only=True)
)
model.eval()

all_probs = []

with torch.no_grad():
    for features, _ in test_loader:
        features = features.to(device)
        probabilities = torch.sigmoid(model(features))
        all_probs.extend(probabilities.cpu().numpy())

predictions = (np.array(all_probs) >= THRESHOLD).astype(int)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, zero_division=0)
recall = recall_score(y_test, predictions, zero_division=0)
f1 = f1_score(y_test, predictions, zero_division=0)
cm = confusion_matrix(y_test, predictions, labels=[0, 1])

tn, fp, fn, tp = cm.ravel()
fpr = fp / (fp + tn)

metrics = {
    "dataset": "UNSW-NB15",
    "threshold": THRESHOLD,
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "false_positive_rate": float(fpr),
    "confusion_matrix": cm.tolist(),
}

print(f"Threshold: {THRESHOLD:.2f}")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"False-positive rate: {fpr:.4f}")
print("\nConfusion matrix [Normal, Attack]:")
print(cm)
print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["Normal", "Attack"],
    digits=4,
    zero_division=0,
))

with open(RESULTS_DIR / "unsw_cnn_threshold_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\nSaved:", RESULTS_DIR / "unsw_cnn_threshold_metrics.json")
