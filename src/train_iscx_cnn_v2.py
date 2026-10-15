
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "datasets" / "ISCX-VPN-nonVPN" / "processed_grouped"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

data = np.load(DATA_DIR / "train_processed.npz")

X = data["X"].astype(np.float32)
y = data["y"].astype(np.int64)
groups = data["groups"]

splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

train_idx, val_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train, X_val = X[train_idx], X[val_idx]
y_train, y_val = y[train_idx], y[val_idx]

assert set(groups[train_idx]).isdisjoint(set(groups[val_idx]))

print("Group-aware validation split created.")

def create_loader(features, labels, batch_size, shuffle=False):
    X_tensor = torch.from_numpy(
        np.ascontiguousarray(features)
    ).unsqueeze(1)
    y_tensor = torch.from_numpy(
        np.ascontiguousarray(labels)
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    return DataLoader(
        TensorDataset(X_tensor, y_tensor),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator if shuffle else None,
    )

train_loader = create_loader(
    X_train, y_train, 256, shuffle=True
)
val_loader = create_loader(
    X_val, y_val, 512
)

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))

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

model = TrafficCNN().to(device)
criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0003,
    weight_decay=0.0001,
)

print("Model parameters:", sum(p.numel() for p in model.parameters()))

def validate():
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            labels_float = labels.to(device).float()

            logits = model(features)
            loss = criterion(logits, labels_float)

            total_loss += loss.item() * len(labels)
            total_samples += len(labels)

            predictions = (
                torch.sigmoid(logits) >= 0.5
            ).long()

            all_labels.extend(labels.numpy())
            all_predictions.extend(predictions.cpu().numpy())

    return (
        total_loss / total_samples,
        f1_score(all_labels, all_predictions, zero_division=0),
    )

EPOCHS = 30
PATIENCE = 5
best_val_f1 = -1.0
best_epoch = 0
epochs_without_improvement = 0

history = {
    "train_loss": [],
    "val_loss": [],
    "val_f1": [],
}

MODEL_PATH = MODEL_DIR / "iscx_cnn_v2_best.pth"
HISTORY_PATH = RESULTS_DIR / "iscx_cnn_v2_history.json"

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for features, labels in train_loader:
        features = features.to(device)
        labels = labels.to(device).float()

        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        total_samples += len(labels)

    train_loss = total_loss / total_samples
    val_loss, val_f1 = validate()

    history["train_loss"].append(float(train_loss))
    history["val_loss"].append(float(val_loss))
    history["val_f1"].append(float(val_f1))

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val F1: {val_f1:.4f}"
    )

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_epoch = epoch + 1
        epochs_without_improvement = 0

        torch.save(
            copy.deepcopy(model.state_dict()),
            MODEL_PATH,
        )
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print(
            f"\nEarly stopping at epoch {epoch + 1}. "
            f"Best epoch: {best_epoch}"
        )
        break

with open(HISTORY_PATH, "w", encoding="utf-8") as f:
    json.dump(history, f, indent=4)

summary = {
    "experiment": "ISCX CNN V2",
    "dataset": "ISCX VPN-nonVPN",
    "training_data": "processed_grouped",
    "learning_rate": 0.0003,
    "weight_decay": 0.0001,
    "max_epochs": EPOCHS,
    "patience": PATIENCE,
    "best_epoch": best_epoch,
    "best_validation_f1": float(best_val_f1),
    "model_path": str(MODEL_PATH),
    "history_path": str(HISTORY_PATH),
}

with open(
    RESULTS_DIR / "iscx_cnn_v2_training_summary.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(summary, f, indent=4)

print("\nCNN V2 training complete.")
print("Best epoch:", best_epoch)
print("Best validation F1:", round(best_val_f1, 4))
print("Model saved:", MODEL_PATH)
print("History saved:", HISTORY_PATH)
