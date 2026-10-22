
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# ---------- ISCX ----------
print("\n--- ISCX ---")

iscx_dir = ROOT / "datasets" / "ISCX-VPN-nonVPN" / "processed_grouped"
iscx_preprocessor = joblib.load(iscx_dir / "preprocessor.joblib")

with np.load(iscx_dir / "test_processed.npz") as data:
    X_iscx = data["X"]
    y_iscx = data["y"]

print("Saved preprocessor loaded")
print("Processed test shape:", X_iscx.shape)
print("Labels shape:", y_iscx.shape)
print("All values finite:", np.isfinite(X_iscx).all())

assert X_iscx.shape[1] == 23
assert np.isfinite(X_iscx).all()


# ---------- UNSW-NB15 ----------
print("\n--- UNSW-NB15 ---")

unsw_dir = ROOT / "datasets" / "UNSW-NB15"
processed_dir = unsw_dir / "processed"

unsw_preprocessor = joblib.load(processed_dir / "preprocessor.joblib")

raw_test_path = unsw_dir / "UNSW_NB15_testing-set.csv"
raw_test = pd.read_csv(raw_test_path)

# Match the input columns used during training preprocessing.
X_raw = raw_test.drop(columns=["id", "attack_cat", "label"])

X_transformed = unsw_preprocessor.transform(X_raw)

if hasattr(X_transformed, "toarray"):
    X_transformed = X_transformed.toarray()

X_transformed = np.asarray(X_transformed, dtype=np.float32)

with np.load(processed_dir / "test_processed.npz") as data:
    X_saved = data["X"]
    y_unsw = data["y"]

print("Saved preprocessor loaded")
print("Raw input shape:", X_raw.shape)
print("Newly transformed shape:", X_transformed.shape)
print("Saved processed shape:", X_saved.shape)
print("Labels shape:", y_unsw.shape)
print("All values finite:", np.isfinite(X_transformed).all())

assert X_transformed.shape[1] == 194
assert X_transformed.shape == X_saved.shape
max_difference = np.max(np.abs(X_transformed - X_saved))

print("Maximum difference from saved features:", max_difference)

assert np.allclose(X_transformed, X_saved, atol=1e-5), (
    "Transformed features do not match saved features!"
)
assert np.isfinite(X_transformed).all()

print("\nAll preprocessing checks passed!")
