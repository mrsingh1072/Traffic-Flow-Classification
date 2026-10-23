
from pathlib import Path
import pandas as pd
from scipy.io import arff

ROOT = Path(__file__).parent

for path in sorted(ROOT.rglob("*")):
    if not path.is_file():
        continue

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, low_memory=False, encoding="cp1252")
        print(f"\nFILE: {path.relative_to(ROOT)}")
        print("Rows:", len(df))
        print("Columns:", len(df.columns))
        print("Column names:", df.columns.tolist())
        print("Missing values:", int(df.isna().sum().sum()))
        print("Duplicate rows:", int(df.duplicated().sum()))

    elif path.suffix.lower() == ".arff":
        print(f"\nFILE: {path.relative_to(ROOT)}")
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.strip().lower().startswith("@data"):
                    break
                if line.strip().lower().startswith("@attribute"):
                    print(line.strip())
