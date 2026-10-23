from pathlib import Path
import csv
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "processed"
SEED = 42
OUT.mkdir(parents=True, exist_ok=True)


def clean_df(df):
    """Normalize column names and decode byte-valued fields."""
    df = df.copy()
    df.columns = [
        str(c).replace("\ufeff", "").replace("ï»¿", "").strip()
        for c in df.columns
    ]

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(
                lambda x: x.decode("utf-8", errors="replace")
                if isinstance(x, bytes)
                else x
            )
    return df


def load_arff(path):
    """Read ARFF files, including blank/? values represented as NaN."""
    attributes, rows, in_data = [], [], False
    pattern = re.compile(
        r"""^\s*@attribute\s+(?:'([^']+)'|"([^"]+)"|(\S+))\s+(.+?)\s*$""",
        re.IGNORECASE,
    )

    with open(path, "r", encoding="utf-8-sig", errors="replace") as file:
        for line_no, line in enumerate(file, 1):
            line = line.strip()
            if not line or line.startswith("%"):
                continue

            if not in_data:
                if line.lower().startswith("@data"):
                    in_data = True
                    continue

                match = pattern.match(line)
                if match:
                    name = match.group(1) or match.group(2) or match.group(3)
                    attributes.append((name, match.group(4).strip().lower()))
                continue

            values = next(csv.reader([line], skipinitialspace=True))
            if len(values) != len(attributes):
                raise ValueError(
                    f"{path.name}, line {line_no}: expected "
                    f"{len(attributes)} values, found {len(values)}"
                )

            row = []
            for value, (_, kind) in zip(values, attributes):
                value = value.strip().strip("'").strip('"')
                if value in ("", "?"):
                    row.append(np.nan)
                elif kind.startswith(("numeric", "real", "integer")):
                    row.append(float(value))
                else:
                    row.append(value)
            rows.append(row)

    if not in_data or not attributes:
        raise ValueError(f"Invalid ARFF file: {path}")

    return clean_df(pd.DataFrame(rows, columns=[a[0] for a in attributes]))


def feature_columns(df, target):
    """Return input features, excluding targets and common identifier columns."""
    return [c for c in df.columns if c not in {target, "id", "label"}]


def signatures(df, columns):
    """
    Fast deterministic row fingerprints for duplicate and split-overlap checks.
    Missing values are handled consistently by pandas hashing.
    """
    if not columns:
        raise ValueError("Cannot create feature signatures: no feature columns.")
    return pd.util.hash_pandas_object(
        df[columns],
        index=False,
        categorize=True,
    )


def deduplicate(df, target):
    """Drop exact duplicate feature-and-target records efficiently."""
    df = clean_df(df).dropna(subset=[target]).copy()
    columns = feature_columns(df, target)
    feature_hashes = signatures(df, columns).to_numpy()
    keys = pd.DataFrame(
        {
            "features": feature_hashes,
            "target": df[target].astype(str).to_numpy(),
        }
    )
    return df.loc[~keys.duplicated().to_numpy()].reset_index(drop=True)


def grouped_split(df, target, test_size):
    """
    Deterministically split by feature fingerprint so matching feature rows
    cannot be assigned to different partitions.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if df.empty:
        raise ValueError("Cannot split an empty dataset.")

    columns = feature_columns(df, target)
    groups = signatures(df, columns).to_numpy()
    y = df[target].astype(str).to_numpy()

    n_splits = round(1 / test_size)
    n_splits = max(2, n_splits)

    unique_groups = np.unique(groups).size
    if unique_groups < n_splits:
        raise ValueError(
            f"Cannot create {n_splits} grouped folds: only "
            f"{unique_groups} unique feature groups are available."
        )

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=SEED,
    )

    try:
        train_idx, held_idx = next(
            splitter.split(np.zeros(len(df)), y, groups)
        )
    except ValueError as exc:
        raise ValueError(
            f"Grouped stratified split failed for target '{target}'. "
            "The dataset may contain classes with too few independent groups. "
            f"Original error: {exc}"
        ) from exc

    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[held_idx].reset_index(drop=True),
    )


def assert_no_group_overlap(parts, target):
    """Fail if any identical raw-feature fingerprint crosses partitions."""
    columns = feature_columns(parts[0], target)
    sets = [set(signatures(part, columns).to_numpy()) for part in parts]

    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            overlap = sets[i] & sets[j]
            if overlap:
                raise ValueError(
                    f"Feature-group leakage detected: split {i} vs split {j} "
                    f"({len(overlap)} shared feature fingerprints)."
                )


def fit_save(name, train_df, val_df, test_df, target):
    """Fit preprocessing on train only, transform all splits, and save arrays."""
    train_df, val_df, test_df = [
        clean_df(part).dropna(subset=[target]).reset_index(drop=True)
        for part in (train_df, val_df, test_df)
    ]

    if any(part.empty for part in (train_df, val_df, test_df)):
        raise ValueError(f"{name}: train, validation, and test must be non-empty.")

    assert_no_group_overlap([train_df, val_df, test_df], target)

    columns = feature_columns(train_df, target)
    for split_name, frame in (("validation", val_df), ("test", test_df)):
        missing = set(columns) - set(frame.columns)
        if missing:
            raise ValueError(
                f"{name}: {split_name} missing columns: {sorted(missing)}"
            )

    numeric, categorical = [], []
    for col in columns:
        series = train_df[col]
        if pd.api.types.is_numeric_dtype(series):
            if pd.to_numeric(series, errors="coerce").notna().any():
                numeric.append(col)
        elif series.notna().any():
            categorical.append(col)

    used = numeric + categorical
    if not used:
        raise ValueError(f"{name}: no usable features found.")

    frames = [part[used].copy() for part in (train_df, val_df, test_df)]

    for frame in frames:
        for col in numeric:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        for col in categorical:
            frame[col] = frame[col].astype("string").astype(object)
            frame[col] = frame[col].where(frame[col].notna(), np.nan)

    transformers = []

    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )

    if categorical:
        try:
            encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            )
        except TypeError:
            encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse=False,
            )

        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", encoder),
                    ]
                ),
                categorical,
            )
        )

    preprocessor = ColumnTransformer(transformers, remainder="drop")

    # Fit on train only; validation/test are transformed using the fitted train pipeline.
    X_train = preprocessor.fit_transform(frames[0])
    X_val = preprocessor.transform(frames[1])
    X_test = preprocessor.transform(frames[2])

    X_train, X_val, X_test = [
        np.asarray(x, dtype=np.float32)[:, :, None]
        for x in (X_train, X_val, X_test)
    ]

    if not all(np.isfinite(x).all() for x in (X_train, X_val, X_test)):
        raise ValueError(f"{name}: non-finite values remain after preprocessing.")

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df[target].astype(str))
    known = set(label_encoder.classes_)

    for split_name, frame in (("validation", val_df), ("test", test_df)):
        unseen = set(frame[target].astype(str)) - known
        if unseen:
            raise ValueError(
                f"{name}: unseen {split_name} labels: {sorted(unseen)}"
            )

    y_val = label_encoder.transform(val_df[target].astype(str))
    y_test = label_encoder.transform(test_df[target].astype(str))

    output = OUT / name
    output.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output / "data.npz",
        X_train=X_train,
        y_train=y_train.astype(np.int64),
        X_val=X_val,
        y_val=y_val.astype(np.int64),
        X_test=X_test,
        y_test=y_test.astype(np.int64),
        classes=label_encoder.classes_.astype(str),
    )

    print(
        f"{name}: train={len(y_train)}, val={len(y_val)}, "
        f"test={len(y_test)}, features={X_train.shape[1]}, "
        f"classes={len(label_encoder.classes_)}"
    )


def process_unsw():
    folder = ROOT / "CSV Files" / "Training and Testing Sets"
    train_path = folder / "UNSW_NB15_training-set.csv"
    test_path = folder / "UNSW_NB15_testing-set.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "UNSW-NB15 files not found. Expected:\n"
            f"  {train_path}\n  {test_path}"
        )

    train = clean_df(pd.read_csv(train_path, low_memory=False))
    test = clean_df(pd.read_csv(test_path, low_memory=False))
    target = "attack_cat"

    if target not in train.columns or target not in test.columns:
        raise ValueError("UNSW-NB15 requires the attack_cat target column.")

    train = deduplicate(train, target)
    test = deduplicate(test, target)

    columns = feature_columns(train, target)
    test_patterns = set(signatures(test, columns).to_numpy())

    # Keep the official test partition; remove matching feature patterns from train.
    train_hashes = signatures(train, columns).to_numpy()
    train = train.loc[~np.isin(train_hashes, list(test_patterns))].reset_index(drop=True)

    train, val = grouped_split(train, target, 0.20)
    fit_save("UNSW_NB15", train, val, test, target)


def process_iscx():
    folder = ROOT / "Scenario B-ARFF"

    for window in ("15s", "30s", "60s", "120s"):
        matches = list(folder.rglob(f"TimeBasedFeatures-Dataset-{window}.arff"))
        if not matches:
            raise FileNotFoundError(f"Missing ISCX {window} ARFF file in {folder}")

        df = deduplicate(load_arff(matches[0]), "class1")

        train_val, test = grouped_split(df, "class1", 0.20)
        train, val = grouped_split(train_val, "class1", 0.25)

        fit_save(f"ISCX_{window}", train, val, test, "class1")


if __name__ == "__main__":
    process_unsw()
    process_iscx()
    print(f"Preprocessing completed. Processed datasets saved to: {OUT}")
