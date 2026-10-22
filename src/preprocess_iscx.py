
from pathlib import Path
import json
import io
import joblib

import arff
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    ROOT / "datasets" / "Scenario A1-ARFF"
    / "TimeBasedFeatures-Dataset-15s-VPN.arff"
)

# Keep the original experiment untouched
OUTPUT_DIR = (
    ROOT / "datasets" / "ISCX-VPN-nonVPN" / "processed_grouped"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load ARFF and repair malformed headers
with open(DATA_FILE, "r", encoding="utf-8", errors="replace") as f:
    lines = f.readlines()

cleaned_lines = []

for line in lines:
    stripped = line.strip()

    if stripped.lower().startswith("@relation"):
        relation = stripped.split(None, 1)[1].split(",")[0]
        cleaned_lines.append(f"@RELATION {relation}\n")
    elif stripped.lower().startswith("@attribute"):
        cleaned_lines.append(line.rstrip().rstrip(",") + "\n")
    else:
        cleaned_lines.append(line)

dataset = arff.load(io.StringIO("".join(cleaned_lines)))

df = pd.DataFrame(
    dataset["data"],
    columns=[attribute[0] for attribute in dataset["attributes"]],
)

print("Original dataset shape:", df.shape)

# 2. Encode labels: Non-VPN = 0, VPN = 1
df["class1"] = df["class1"].astype(str).str.strip()

label_mapping = {"Non-VPN": 0, "VPN": 1}
unexpected = set(df["class1"].unique()) - set(label_mapping)

if unexpected:
    raise ValueError(f"Unexpected labels: {unexpected}")

y = df["class1"].map(label_mapping).to_numpy(dtype=np.int64)

# 3. Prepare raw numerical features
X_df = df.drop(columns=["class1"]).apply(
    pd.to_numeric, errors="coerce"
)
X_df = X_df.replace([np.inf, -np.inf], np.nan)

feature_names = X_df.columns.tolist()

# 4. Group identical raw feature vectors BEFORE preprocessing.
# Missing values are represented consistently so identical rows
# containing missing values remain in the same group.
def canonical_row(row):
    return tuple(
        None if pd.isna(value) else float(value)
        for value in row
    )

group_lookup = {}
groups = np.empty(len(X_df), dtype=np.int64)

for i, row in enumerate(X_df.to_numpy(dtype=np.float64)):
    key = canonical_row(row)

    if key not in group_lookup:
        group_lookup[key] = len(group_lookup)

    groups[i] = group_lookup[key]

print("Unique feature-vector groups:", len(group_lookup))
print("Rows belonging to repeated feature vectors:",
      len(groups) - len(group_lookup))

# 5. Split by groups. No exact raw feature vector can appear
# in both training and test sets.
splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

train_idx, test_idx = next(
    splitter.split(X_df, y, groups=groups)
)

X_train = X_df.iloc[train_idx].to_numpy(dtype=np.float64)
X_test = X_df.iloc[test_idx].to_numpy(dtype=np.float64)

y_train = y[train_idx]
y_test = y[test_idx]

groups_train = groups[train_idx]
groups_test = groups[test_idx]

print("\nGrouped split:")
print("Training features:", X_train.shape)
print("Test features:", X_test.shape)
print("Training class counts:", np.bincount(y_train, minlength=2))
print("Test class counts:", np.bincount(y_test, minlength=2))

# 6. Fit preprocessing only on training data
preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

X_train_processed = preprocessor.fit_transform(
    X_train
).astype(np.float32)

X_test_processed = preprocessor.transform(
    X_test
).astype(np.float32)

if not np.isfinite(X_train_processed).all():
    raise ValueError("Training data contains non-finite values.")

if not np.isfinite(X_test_processed).all():
    raise ValueError("Test data contains non-finite values.")

# 7. Verify no raw feature-vector groups cross the split
overlap = set(groups_train).intersection(set(groups_test))

print("Groups shared between train and test:", len(overlap))

if overlap:
    raise ValueError("Group leakage detected!")

# 8. Save the new split separately
np.savez_compressed(
    OUTPUT_DIR / "train_processed.npz",
    X=X_train_processed,
    y=y_train,
    groups=groups[train_idx],
)

np.savez_compressed(
    OUTPUT_DIR / "test_processed.npz",
    X=X_test_processed,
    y=y_test,
    groups=groups[test_idx],
)

metadata = {
    "dataset": "ISCX VPN-nonVPN",
    "source_file": DATA_FILE.name,
    "target": "class1",
    "class_mapping": {"0": "Non-VPN", "1": "VPN"},
    "feature_names": feature_names,
    "feature_count": len(feature_names),
    "train_samples": int(len(y_train)),
    "test_samples": int(len(y_test)),
    "split_method": "StratifiedGroupKFold",
    "n_splits": 5,
    "random_state": 42,
    "unique_feature_vector_groups": int(len(group_lookup)),
    "train_groups": int(len(np.unique(groups_train))),
    "test_groups": int(len(np.unique(groups_test))),
    "preprocessing": [
        "Group identical raw feature vectors before splitting",
        "Stratified grouped five-fold split; first fold used as test",
        "Median imputation fitted on training data",
        "StandardScaler fitted on training data",
    ],
}

with open(OUTPUT_DIR / "metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4)

print("\nGrouped preprocessing complete!")
print("All training values finite:", np.isfinite(X_train_processed).all())
print("All test values finite:", np.isfinite(X_test_processed).all())
print("Saved to:", OUTPUT_DIR)

# Save fitted preprocessing pipeline for inference
joblib.dump(
    preprocessor,
    OUTPUT_DIR / "preprocessor.joblib"
)

print("Saved preprocessor:", OUTPUT_DIR / "preprocessor.joblib")