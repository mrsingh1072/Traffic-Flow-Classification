
from pathlib import Path
import json
import joblib

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# --------------------------------------------------
# 1. Paths
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "datasets" / "UNSW-NB15"
OUTPUT_DIR = DATA_DIR / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "UNSW_NB15_training-set.csv"
TEST_PATH = DATA_DIR / "UNSW_NB15_testing-set.csv"

# --------------------------------------------------
# 2. Load original datasets
# --------------------------------------------------
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("Original training shape:", train_df.shape)
print("Original testing shape:", test_df.shape)

# --------------------------------------------------
# 3. Separate target and input features
# --------------------------------------------------
TARGET = "label"
DROP_COLUMNS = ["id", "attack_cat", TARGET]

X_train = train_df.drop(columns=DROP_COLUMNS).copy()
X_test = test_df.drop(columns=DROP_COLUMNS).copy()

y_train = train_df[TARGET].astype("int64")
y_test = test_df[TARGET].astype("int64")

# --------------------------------------------------
# 4. Identify numerical and categorical features
# --------------------------------------------------
categorical_cols = ["proto", "service", "state"]
numeric_cols = [
    col for col in X_train.columns
    if col not in categorical_cols
]

# Convert numeric columns safely and replace infinities
for col in numeric_cols:
    X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
    X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

X_train[numeric_cols] = X_train[numeric_cols].replace(
    [np.inf, -np.inf], np.nan
)
X_test[numeric_cols] = X_test[numeric_cols].replace(
    [np.inf, -np.inf], np.nan
)

# --------------------------------------------------
# 5. Build preprocessing pipelines
# --------------------------------------------------
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_cols),
    ("categorical", categorical_pipeline, categorical_cols)
])

# Fit only on training data to prevent leakage
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# --------------------------------------------------
# 6. Convert sparse output to dense arrays
# --------------------------------------------------
X_train_processed = np.asarray(
    X_train_processed.toarray()
    if hasattr(X_train_processed, "toarray")
    else X_train_processed,
    dtype=np.float32
)

X_test_processed = np.asarray(
    X_test_processed.toarray()
    if hasattr(X_test_processed, "toarray")
    else X_test_processed,
    dtype=np.float32
)

y_train_array = y_train.to_numpy(dtype=np.int64)
y_test_array = y_test.to_numpy(dtype=np.int64)

# --------------------------------------------------
# 7. Save processed datasets
# --------------------------------------------------
np.savez_compressed(
    OUTPUT_DIR / "train_processed.npz",
    X=X_train_processed,
    y=y_train_array
)

np.savez_compressed(
    OUTPUT_DIR / "test_processed.npz",
    X=X_test_processed,
    y=y_test_array
)

# Save metadata for reproducibility
metadata = {
    "target": TARGET,
    "class_mapping": {
        "0": "Normal",
        "1": "Attack"
    },
    "dropped_columns": DROP_COLUMNS,
    "numeric_features": numeric_cols,
    "categorical_features": categorical_cols,
    "processed_feature_count": int(X_train_processed.shape[1]),
    "training_rows": int(X_train_processed.shape[0]),
    "testing_rows": int(X_test_processed.shape[0])
}

with open(OUTPUT_DIR / "metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4)

# --------------------------------------------------
# 8. Print summary
# --------------------------------------------------
print("\n--- PREPROCESSING COMPLETE ---")
print("Training matrix:", X_train_processed.shape)
print("Testing matrix:", X_test_processed.shape)
print("Training labels:", np.bincount(y_train_array))
print("Testing labels:", np.bincount(y_test_array))
print("Training dtype:", X_train_processed.dtype)
print("Testing dtype:", X_test_processed.dtype)
print("Saved files to:", OUTPUT_DIR)

# Save fitted preprocessing pipeline for inference
joblib.dump(
    preprocessor,
    OUTPUT_DIR / "preprocessor.joblib"
)

print("Saved preprocessor:", OUTPUT_DIR / "preprocessor.joblib")