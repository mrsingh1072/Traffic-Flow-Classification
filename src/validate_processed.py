
from pathlib import Path
import numpy as np
import json

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "datasets" / "UNSW-NB15" / "processed"

for name in ["train", "test"]:
    path = PROCESSED_DIR / f"{name}_processed.npz"

    with np.load(path) as data:
        X = data["X"]
        y = data["y"]

    print(f"\n--- {name.upper()} DATA ---")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Data type:", X.dtype)
    print("All values finite:", np.isfinite(X).all())
    print("Label values:", np.unique(y))
    print("Class counts:", np.bincount(y))
    print("Feature means: min =", X.mean(axis=0).min(),
          ", max =", X.mean(axis=0).max())

with open(PROCESSED_DIR / "metadata.json", encoding="utf-8") as f:
    metadata = json.load(f)

print("\nProcessed feature count:", metadata["processed_feature_count"])
print("Target:", metadata["target"])
print("Class mapping:", metadata["class_mapping"])
