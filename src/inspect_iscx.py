
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    ROOT
    / "datasets"
    / "Scenario A1-ARFF"
    / "TimeBasedFeatures-Dataset-15s-VPN.arff"
)

print("Dataset path:", DATA_FILE)
print("File exists:", DATA_FILE.exists())

if not DATA_FILE.exists():
    raise FileNotFoundError(
        "ISCX ARFF file not found. Check the folder and filename."
    )

# Reuse the existing ARFF loader from inspect_dataset.py only
# if it exposes a reusable function. Otherwise, we'll parse
# the ARFF file directly in this script.
import arff

with open(DATA_FILE, "r", encoding="utf-8", errors="replace") as f:
    lines = f.readlines()

# Clean malformed trailing commas in the ARFF header.
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

dataset = arff.load(
    __import__("io").StringIO("".join(cleaned_lines))
)

df = pd.DataFrame(
    dataset["data"],
    columns=[attribute[0] for attribute in dataset["attributes"]],
)

print("\nDataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nClass distribution:")
print(df["class1"].value_counts(dropna=False))

print("\nMissing values:", int(df.isna().sum().sum()))
print("\nDuplicate rows:", int(df.duplicated().sum()))

print("\nFirst five rows:")
print(df.head())
