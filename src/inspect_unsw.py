
from pathlib import Path
import pandas as pd

DATASET_DIR = (
    Path(__file__).resolve().parent.parent
    / "datasets"
    / "UNSW-NB15"
)

train_file = DATASET_DIR / "UNSW_NB15_training-set.csv"
test_file = DATASET_DIR / "UNSW_NB15_testing-set.csv"

for file in [train_file, test_file]:
    print("\n" + "=" * 60)
    print("File:", file.name)

    if not file.exists():
        print("ERROR: File not found:", file)
        continue

    df = pd.read_csv(file)

    print("Shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    print("\nData types:")
    print(df.dtypes)

    for target in ["label", "attack_cat"]:
        if target in df.columns:
            print(f"\n{target} distribution:")
            print(df[target].value_counts(dropna=False))
