from pathlib import Path
import json
import random
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:
    raise SystemExit(
        "PyTorch is missing from this venv. Install the correct build using "
        "the official PyTorch installation selector, then rerun. " + str(exc)
    )

ROOT = Path(__file__).resolve().parents[1]  # project root when script is in backend/
DATA_ROOT = ROOT / "datasets" / "processed"
MODEL_ROOT = ROOT / "backend" / "models"
RESULT_ROOT = ROOT / "backend" / "results"
DATASETS = ["UNSW_NB15", "ISCX_15s", "ISCX_30s", "ISCX_60s", "ISCX_120s"]
FRACTIONS = [0.10, 0.25, 0.50, 1.00]
SEED, BATCH_SIZE, MAX_EPOCHS, PATIENCE = 42, 256, 50, 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TrafficCNN(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2, ceil_mode=True),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveMaxPool1d(1), nn.Flatten(),
            nn.Dropout(0.30), nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(0.20), nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.net(x.transpose(1, 2))  # (N, features, 1) -> (N, 1, features)


def load_data(name):
    path = DATA_ROOT / name / "data.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing processed dataset: {path}")
    with np.load(path, allow_pickle=False) as d:
        required = ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test", "classes"]
        missing = set(required) - set(d.files)
        if missing:
            raise ValueError(f"{path} missing arrays: {sorted(missing)}")
        a = {k: d[k] for k in required}

    Xtr, Xv, Xte = [np.asarray(a[k], dtype=np.float32) for k in ("X_train", "X_val", "X_test")]
    ytr, yv, yte = [np.asarray(a[k], dtype=np.int64) for k in ("y_train", "y_val", "y_test")]
    classes = a["classes"].astype(str)

    for label, X, y in [("train", Xtr, ytr), ("validation", Xv, yv), ("test", Xte, yte)]:
        if X.ndim != 3 or X.shape[2] != 1 or len(X) != len(y) or not len(y):
            raise ValueError(f"{name} {label}: invalid tensor/label shapes: X={X.shape}, y={y.shape}")
        if not np.isfinite(X).all():
            raise ValueError(f"{name} {label}: NaN/Inf values found")
        if y.min() < 0 or y.max() >= len(classes):
            raise ValueError(f"{name} {label}: label outside classes array")

    expected = set(range(len(classes)))
    if set(np.unique(ytr)) != expected:
        raise ValueError(f"{name}: training split does not contain every encoded class")
    if not set(np.unique(yv)).issubset(expected) or not set(np.unique(yte)).issubset(expected):
        raise ValueError(f"{name}: validation/test has an unknown encoded class")
    return Xtr, ytr, Xv, yv, Xte, yte, classes


def labeled_indices(y, fraction):
    if fraction == 1.0:
        return np.arange(len(y))
    try:
        selected, _ = train_test_split(
            np.arange(len(y)), train_size=fraction, random_state=SEED,
            shuffle=True, stratify=y
        )
    except ValueError as exc:
        counts = pd.Series(y).value_counts().sort_index().to_dict()
        raise ValueError(
            f"Cannot make a stratified {fraction:.0%} subset; class counts={counts}. "
            f"No unstratified fallback was used. Details: {exc}"
        ) from exc
    if set(np.unique(y[selected])) != set(np.unique(y)):
        raise ValueError(f"{fraction:.0%} subset does not contain every class")
    return np.sort(selected)


def loader(X, y, shuffle=False):
    ds = TensorDataset(
        torch.from_numpy(np.ascontiguousarray(X)),
        torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64)),
    )
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0,
                      pin_memory=DEVICE.type == "cuda")


def run_experiment(name, fraction, Xtr, ytr, Xv, yv, Xte, yte, classes):
    seed_all()
    tag = f"{int(fraction * 100):03d}pct"
    run_name = f"{name}_{tag}"
    run_dir = MODEL_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)

    idx = labeled_indices(ytr, fraction)
    train_loader = loader(Xtr[idx], ytr[idx], True)
    val_loader, test_loader = loader(Xv, yv), loader(Xte, yte)
    model = TrafficCNN(Xtr.shape[1], len(classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_loss, best_epoch, stale, best_state = float("inf"), 0, 0, None
    start = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total, seen = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(yb)
            seen += len(yb)

        model.eval()
        val_total, val_seen, vt, vp = 0.0, 0, [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                val_total += criterion(logits, yb).item() * len(yb)
                val_seen += len(yb)
                vt.extend(yb.cpu().numpy())
                vp.extend(logits.argmax(1).cpu().numpy())
        val_loss = val_total / max(1, val_seen)
        scheduler.step(val_loss)
        print(f"[{run_name}] epoch {epoch:02d} train_loss={total/max(1,seen):.4f} "
              f"val_loss={val_loss:.4f} val_accuracy={accuracy_score(vt, vp):.4f}", flush=True)

        if val_loss < best_loss:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"[{run_name}] early stopping at epoch {epoch}", flush=True)
                break

    if best_state is None:
        raise RuntimeError(f"{run_name}: no model checkpoint was produced")
    model.load_state_dict(best_state)
    model.to(DEVICE).eval()

    yt, yp, test_total, test_seen = [], [], 0.0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            test_total += criterion(logits, yb).item() * len(yb)
            test_seen += len(yb)
            yt.extend(yb.cpu().numpy())
            yp.extend(logits.argmax(1).cpu().numpy())

    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        yt, yp, labels=np.arange(len(classes)), average="macro", zero_division=0
    )
    p_weighted, r_weighted, f_weighted, _ = precision_recall_fscore_support(
        yt, yp, labels=np.arange(len(classes)), average="weighted", zero_division=0
    )
    cm = confusion_matrix(yt, yp, labels=np.arange(len(classes)))
    report = classification_report(yt, yp, labels=np.arange(len(classes)),
                                   target_names=classes.tolist(), output_dict=True, zero_division=0)

    checkpoint = {
        "model_state_dict": best_state, "dataset": name, "label_fraction": fraction,
        "num_features": int(Xtr.shape[1]), "num_classes": int(len(classes)),
        "classes": classes.tolist(), "input_shape": [int(Xtr.shape[1]), 1],
        "best_epoch": int(best_epoch), "best_validation_loss": float(best_loss),
        "architecture": "TrafficCNN-1D", "seed": SEED,
    }
    torch.save(checkpoint, run_dir / "model.pt")
    np.save(run_dir / "confusion_matrix.npy", cm)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(
        run_dir / "confusion_matrix.csv", index_label="actual/predicted"
    )
    (run_dir / "classification_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    result = {
        "dataset": name, "label_fraction": fraction,
        "labeled_train_samples": int(len(idx)), "full_train_samples": int(len(ytr)),
        "validation_samples": int(len(yv)), "test_samples": int(len(yte)),
        "features": int(Xtr.shape[1]), "classes": int(len(classes)),
        "best_epoch": int(best_epoch), "best_validation_loss": float(best_loss),
        "test_loss": float(test_total / max(1, test_seen)),
        "accuracy": float(accuracy_score(yt, yp)),
        "precision_macro": float(p_macro), "recall_macro": float(r_macro), "f1_macro": float(f_macro),
        "precision_weighted": float(p_weighted), "recall_weighted": float(r_weighted),
        "f1_weighted": float(f_weighted), "training_seconds": round(time.time() - start, 2),
        "device": str(DEVICE), "model_path": str(run_dir / "model.pt"),
        "confusion_matrix_path": str(run_dir / "confusion_matrix.csv"),
    }
    print(f"[{run_name}] TEST accuracy={result['accuracy']:.4f} "
          f"macro_precision={p_macro:.4f} macro_recall={r_macro:.4f} macro_f1={f_macro:.4f}",
          flush=True)
    return result


def main():
    print(f"Device: {DEVICE}")
    print(f"Running {len(DATASETS)} datasets x {len(FRACTIONS)} label fractions.")
    print("Validation and test sets stay fixed for every label fraction.")
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    results = []

    for name in DATASETS:
        print(f"\n========== {name} ==========", flush=True)
        Xtr, ytr, Xv, yv, Xte, yte, classes = load_data(name)
        print(f"train={len(ytr)} val={len(yv)} test={len(yte)} features={Xtr.shape[1]} classes={len(classes)}")
        for fraction in FRACTIONS:
            results.append(run_experiment(name, fraction, Xtr, ytr, Xv, yv, Xte, yte, classes))
            pd.DataFrame(results).to_csv(RESULT_ROOT / "label_fraction_results.csv", index=False)
            (RESULT_ROOT / "label_fraction_results.json").write_text(
                json.dumps(results, indent=2), encoding="utf-8"
            )

    table = pd.DataFrame(results).sort_values(["dataset", "label_fraction"])
    print("\n========== SUMMARY ==========")
    print(table[["dataset", "label_fraction", "labeled_train_samples", "accuracy",
                 "precision_macro", "recall_macro", "f1_macro", "best_epoch"]].to_string(index=False))
    print(f"\nModels: {MODEL_ROOT}\nResults: {RESULT_ROOT}")


if __name__ == "__main__":
    main()
