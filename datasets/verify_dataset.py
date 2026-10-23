
import numpy as np
from pathlib import Path

root = Path(__file__).resolve().parent / "processed"
names = ["UNSW_NB15", "ISCX_15s", "ISCX_30s", "ISCX_60s", "ISCX_120s"]

for name in names:
    path = root / name / "data.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")

    with np.load(path, allow_pickle=True) as data:
        classes = data["classes"]
        arrays = {}

        for split in ["train", "val", "test"]:
            X = data[f"X_{split}"]
            y = data[f"y_{split}"]

            assert X.ndim == 3 and X.shape[2] == 1, f"{name}: invalid CNN shape"
            assert np.isfinite(X).all(), f"{name}: NaN or infinity in {split}"
            assert len(X) == len(y), f"{name}: X/y size mismatch"
            assert y.min() >= 0 and y.max() < len(classes), f"{name}: invalid labels"

            arrays[split] = X.reshape(len(X), -1)

        assert (
            arrays["train"].shape[1]
            == arrays["val"].shape[1]
            == arrays["test"].shape[1]
        ), f"{name}: feature count mismatch"

        # Count exact feature-vector overlaps between splits.
        keys = {
            split: {row.tobytes() for row in values}
            for split, values in arrays.items()
        }

        overlap = (
            len(keys["train"] & keys["val"])
            + len(keys["train"] & keys["test"])
            + len(keys["val"] & keys["test"])
        )

        print(
            f"{name}: VALID | "
            f"train={len(data['y_train'])} | "
            f"val={len(data['y_val'])} | "
            f"test={len(data['y_test'])} | "
            f"classes={len(classes)} | "
            f"exact cross-split overlaps={overlap}"
        )

print("\nBasic dataset validation completed.")
