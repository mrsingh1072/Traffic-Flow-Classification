
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
    accuracy_score,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "datasets" / "ISCX-VPN-nonVPN" / "processed_grouped"
MODEL_PATH = ROOT / "models" / "iscx_cnn_v2_best.pth"

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


def get_probabilities(model, X):
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(
                np.ascontiguousarray(X, dtype=np.float32)
            ).unsqueeze(1)
        ),
        batch_size=512,
        shuffle=False,
    )

    results = []
    model.eval()

    with torch.no_grad():
        for (features,) in loader:
            features = features.to(device)
            probs = torch.sigmoid(model(features))
            results.extend(probs.cpu().numpy())

    return np.asarray(results)


# Recreate the exact validation split used during V2 training
train_data = np.load(DATA_DIR / "train_processed.npz")
X = train_data["X"].astype(np.float32)
y = train_data["y"].astype(np.int64)

_, X_val, _, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

model = TrafficCNN().to(device)
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device, weights_only=True)
)

val_probs = get_probabilities(model, X_val)

print("\nVALIDATION THRESHOLD COMPARISON")
print("Threshold | Precision | Recall | F1")
print("-" * 45)

results = []

for threshold in np.arange(0.30, 0.801, 0.05):
    pred = (val_probs >= threshold).astype(int)

    precision = precision_score(y_val, pred, zero_division=0)
    recall = recall_score(y_val, pred, zero_division=0)
    f1 = f1_score(y_val, pred, zero_division=0)

    results.append((threshold, precision, recall, f1))

    print(
        f"{threshold:.2f}      | {precision:.4f}    | "
        f"{recall:.4f} | {f1:.4f}"
    )

best = max(results, key=lambda row: row[3])
best_threshold = float(best[0])

print("\nSelected validation threshold:", round(best_threshold, 2))
print("Best validation F1:", round(best[3], 4))

# Evaluate the selected threshold ONCE on the grouped test set
test_data = np.load(DATA_DIR / "test_processed.npz")
X_test = test_data["X"].astype(np.float32)
y_test = test_data["y"].astype(np.int64)

test_probs = get_probabilities(model, X_test)
test_pred = (test_probs >= best_threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_test, test_pred, labels=[0, 1]
).ravel()

print("\nGROUPED TEST RESULTS")
print("Threshold:", round(best_threshold, 2))
print(f"Accuracy: {accuracy_score(y_test, test_pred):.4f}")
print(
    f"Precision (VPN): "
    f"{precision_score(y_test, test_pred, zero_division=0):.4f}"
)
print(
    f"Recall (VPN): "
    f"{recall_score(y_test, test_pred, zero_division=0):.4f}"
)
print(
    f"F1-score (VPN): "
    f"{f1_score(y_test, test_pred, zero_division=0):.4f}"
)
print(f"False-positive rate: {fp / (fp + tn):.4f}")
print("Confusion matrix [[TN, FP], [FN, TP]]:")
print([[int(tn), int(fp)], [int(fn), int(tp)]])
