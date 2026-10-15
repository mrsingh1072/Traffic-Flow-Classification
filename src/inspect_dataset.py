
from pathlib import Path
import arff
import pandas as pd
import re

DATASET_DIR = (
    Path(__file__).resolve().parent.parent
    / "datasets"
    / "Scenario A1-ARFF"
)

dataset_file = DATASET_DIR / "TimeBasedFeatures-Dataset-15s-VPN.arff"

print("Loading:", dataset_file.name)

# Read the original file without modifying it.
with open(dataset_file, "r", encoding="utf-8-sig") as file:
    lines = file.readlines()

# Remove trailing commas from ARFF metadata lines.
cleaned_lines = []

for line in lines:
    stripped = line.strip()

    if stripped.upper().startswith("@RELATION"):
        relation_name = re.match(
            r"@RELATION\s+(.+)", stripped, re.IGNORECASE
        ).group(1).split(",")[0].strip()
        line = f"@RELATION {relation_name}\n"

    elif stripped.upper().startswith("@ATTRIBUTE"):
        line = re.sub(r",+\s*$", "", line.rstrip()) + "\n"

    cleaned_lines.append(line)

# Parse the cleaned content in memory.
import io

dataset = arff.load(io.StringIO("".join(cleaned_lines)))

df = pd.DataFrame(
    dataset["data"],
    columns=[attribute[0] for attribute in dataset["attributes"]]
)

print("\n--- Dataset Shape ---")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Column Names ---")
print(df.columns.tolist())

print("\n--- Data Types ---")
print(df.dtypes)

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Class Distribution ---")
print(df["class1"].value_counts(dropna=False))
