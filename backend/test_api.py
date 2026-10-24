
from pathlib import Path
import json
import urllib.request
import urllib.error

import joblib
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "datasets"

PREPROCESSOR_PATH = (
    DATASET_DIR / "processed" / "UNSW_NB15" / "preprocessor.joblib"
)

API_URL = "http://127.0.0.1:8000/predict"


def main():
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessor not found: {PREPROCESSOR_PATH}"
        )

    artifact = joblib.load(PREPROCESSOR_PATH)
    feature_columns = artifact["feature_columns"]

    files = sorted(DATASET_DIR.rglob("UNSW_NB15_testing-set.csv"))

    if not files:
        raise FileNotFoundError(
            "Could not find UNSW_NB15_testing-set.csv inside datasets."
        )

    csv_path = files[0]
    print(f"Dataset: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)

    missing_columns = [
        col for col in feature_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required features: {missing_columns}"
        )

    # Select one complete, real record.
    row = df.iloc[0]
    features = {}

    for column in feature_columns:
        value = row[column]

        if pd.isna(value):
            features[column] = None
        elif isinstance(value, np.generic):
            features[column] = value.item()
        else:
            features[column] = value

    payload = json.dumps({"features": features}).encode("utf-8")

    request = urllib.request.Request(
        API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))

        print("\n========== REAL TRAFFIC PREDICTION ==========")
        print(f"Actual class:    {row.get('attack_cat', 'Not available')}")
        print(f"Predicted class: {result['predicted_class']}")
        print(f"Confidence:      {result['confidence']:.4f}")
        print(f"HTTP status:     200")
        print("\nClass probabilities:")

        for name, probability in result["probabilities"].items():
            print(f"  {name:20s} {probability:.6f}")

    except urllib.error.HTTPError as exc:
        print(f"API returned HTTP {exc.code}:")
        print(exc.read().decode("utf-8"))
        raise


if __name__ == "__main__":
    main()
