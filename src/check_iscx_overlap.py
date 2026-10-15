
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "datasets" / "ISCX-VPN-nonVPN" / "processed"

train = np.load(DATA_DIR / "train_processed.npz")
test = np.load(DATA_DIR / "test_processed.npz")

X_train = train["X"]
y_train = train["y"]
X_test = test["X"]
y_test = test["y"]

# Convert each feature vector into a tuple for exact comparison
train_map = {}

for features, label in zip(X_train, y_train):
    key = tuple(features.tolist())
    train_map.setdefault(key, set()).add(int(label))

overlap_count = 0
conflicting_label_count = 0

for features, label in zip(X_test, y_test):
    key = tuple(features.tolist())

    if key in train_map:
        overlap_count += 1

        if int(label) not in train_map[key]:
            conflicting_label_count += 1

print("Training samples:", len(X_train))
print("Test samples:", len(X_test))
print("Test samples with identical training feature vectors:", overlap_count)
print("Overlapping samples with conflicting labels:", conflicting_label_count)

if overlap_count == 0:
    print("\nNo exact feature-vector overlap detected.")
else:
    print("\nExact overlap exists; investigate before finalizing results.")
