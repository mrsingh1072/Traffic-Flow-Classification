
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# --------------------------------------------------
# 1. Paths and reproducibility
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "datasets" / "ISCX-VPN-nonVPN" / "processed"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# --------------------------------------------------
# 2. Load preprocessed data
# --------------------------------------------------
train_data = np.load(DATA_DIR / "train_processed.npz")
test_data = np.load(DATA_DIR / "test_processed.npz")

X = train_data["X"].astype(np.float32)
y = train_data["y"].astype(np.int64)

X_test = test_data["X"].astype(np.float32)
y_test = test_data["y"].astype(np.int64)

print("Full training data:", X.shape)
print("Official test data:", X_test.shape)

# Split only the training portion into train and validation.
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# --------------------------------------------------
# 3. Data loaders
# --------------------------------------------------
def create_loader(features, labels, batch_size, shuffle=False):
    X_tensor = torch.from_numpy(
        np.ascontiguousarray(features)
    ).unsqueeze(1)

    y_tensor = torch.from_numpy(
        np.ascontiguousarray(labels)
    )

    return DataLoader(
        TensorDataset(X_tensor, y_tensor),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


train_loader = create_loader(
    X_train, y_train, batch_size=256, shuffle=True
)

val_loader = create_loader(
    X_val, y_val, batch_size=512
)

test_loader = create_loader(
    X_test, y_test, batch_size=512
)

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Test samples:", len(X_test))

# --------------------------------------------------
# 4. Define the 1D CNN
# --------------------------------------------------
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
        x = self.features(x)
        return self.classifier(x).squeeze(1)


model = TrafficCNN().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Model parameters:", sum(p.numel() for p in model.parameters()))

# --------------------------------------------------
# 5. Validation helper
# --------------------------------------------------
def evaluate_validation(loader):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels_float = labels.to(device).float()

            logits = model(features)
            loss = criterion(logits, labels_float)

            total_loss += loss.item() * len(labels)
            total_samples += len(labels)

            predictions = (torch.sigmoid(logits) >= 0.5).long()
            all_labels.extend(labels.numpy())
            all_predictions.extend(predictions.cpu().numpy())

    return (
        total_loss / total_samples,
        f1_score(all_labels, all_predictions, zero_division=0),
    )


# --------------------------------------------------
# 6. Train and save the best validation checkpoint
# --------------------------------------------------
EPOCHS = 20
best_val_f1 = -1.0
history = {
    "train_loss": [],
    "val_loss": [],
    "val_f1": [],
}

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
    val_loss, val_f1 = evaluate_validation(val_loader)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_f1"].append(val_f1)

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val F1: {val_f1:.4f}"
    )

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1

        torch.save(
            copy.deepcopy(model.state_dict()),
            MODEL_DIR / "iscx_cnn_best.pth",
        )

print("\nTraining completed!")
print(f"Best validation F1: {best_val_f1:.4f}")

with open(RESULTS_DIR / "iscx_cnn_history.json", "w") as f:
    json.dump(history, f, indent=4)

# --------------------------------------------------
# 7. Evaluate on the untouched official test set
# --------------------------------------------------
model.load_state_dict(
    torch.load(
        MODEL_DIR / "iscx_cnn_best.pth",
        map_location=device,
        weights_only=True,
    )
)
model.eval()

all_true = []
all_predictions = []

with torch.no_grad():
    for features, labels in test_loader:
        features = features.to(device)
        logits = model(features)
        predictions = (torch.sigmoid(logits) >= 0.5).long()

        all_true.extend(labels.numpy())
        all_predictions.extend(predictions.cpu().numpy())

accuracy = accuracy_score(all_true, all_predictions)
precision = precision_score(all_true, all_predictions, zero_division=0)
recall = recall_score(all_true, all_predictions, zero_division=0)
f1 = f1_score(all_true, all_predictions, zero_division=0)
cm = confusion_matrix(all_true, all_predictions, labels=[0, 1])

metrics = {
    "dataset": "ISCX VPN-nonVPN",
    "task": "Non-VPN vs VPN",
    "threshold": 0.5,
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "best_validation_f1": float(best_val_f1),
    "confusion_matrix": cm.tolist(),
}

with open(RESULTS_DIR / "iscx_cnn_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\n========== ISCX OFFICIAL TEST RESULTS ==========")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print("\nConfusion matrix [Non-VPN, VPN]:")
print(cm)

print("\nClassification report:")
print(
    classification_report(
        all_true,
        all_predictions,
        labels=[0, 1],
        target_names=["Non-VPN", "VPN"],
        digits=4,
        zero_division=0,
    )
)

print("\nModel saved:", MODEL_DIR / "iscx_cnn_best.pth")
print("Metrics saved:", RESULTS_DIR / "iscx_cnn_metrics.json")
print("History saved:", RESULTS_DIR / "iscx_cnn_history.json")
