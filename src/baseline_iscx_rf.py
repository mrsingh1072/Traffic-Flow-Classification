
from pathlib import Path
import json

import numpy as np
from sklearn.ensemble import RandomForestClassifier
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
RESULT_PATH = ROOT / "results" / "iscx_rf_grouped_metrics.json"


# Load the prepared training and official test data
train = np.load(DATA_DIR / "train_processed.npz")
test = np.load(DATA_DIR / "test_processed.npz")

X_train = train["X"].astype(np.float32)
y_train = train["y"].astype(np.int64)
X_test = test["X"].astype(np.float32)
y_test = test["y"].astype(np.int64)

print("Training samples:", X_train.shape)
print("Test samples:", X_test.shape)

# Train only on the training set
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

print("\nTraining Random Forest...")
model.fit(X_train, y_train)

# Evaluate once on the official test set
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, zero_division=0)
recall = recall_score(y_test, predictions, zero_division=0)
f1 = f1_score(y_test, predictions, zero_division=0)

tn, fp, fn, tp = confusion_matrix(
    y_test, predictions, labels=[0, 1]
).ravel()

fpr = fp / (fp + tn) if (fp + tn) else 0.0

print("\nISCX RANDOM FOREST TEST RESULTS")
print("=" * 45)
print(f"Accuracy:            {accuracy:.4f}")
print(f"Precision (VPN):     {precision:.4f}")
print(f"Recall (VPN):        {recall:.4f}")
print(f"F1-score (VPN):      {f1:.4f}")
print(f"False-positive rate: {fpr:.4f}")
print("\nConfusion matrix [[TN, FP], [FN, TP]]:")
print([[int(tn), int(fp)], [int(fn), int(tp)]])

print("\nClassification report:")
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
    "model": "Random Forest",
    "evaluation_split": "official_test",
    "threshold": 0.5,
    "accuracy": float(accuracy),
    "precision_vpn": float(precision),
    "recall_vpn": float(recall),
    "f1_vpn": float(f1),
    "false_positive_rate": float(fpr),
    "confusion_matrix": [
        [int(tn), int(fp)],
        [int(fn), int(tp)],
    ],
}

RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(RESULT_PATH, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

print("\nMetrics saved to:", RESULT_PATH)
