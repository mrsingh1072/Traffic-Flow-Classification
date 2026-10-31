"""Isolated, validation-selected fine-tuning; never overwrites serving models."""
from pathlib import Path
import hashlib
import json
import random
import sys
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from train_cnn import TrafficCNN

OUT = Path(__file__).resolve().parent
NAMES = ["UNSW_NB15", "ISCX_15s", "ISCX_30s", "ISCX_60s", "ISCX_120s"]
MAX_EPOCHS, PATIENCE, SEED = 12, 4, 42
torch.set_num_threads(4)


def predict(model, features):
    model.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(features), 512):
            values.append(model(torch.from_numpy(np.ascontiguousarray(features[start:start + 512]))).argmax(1).numpy())
    return np.concatenate(values)


def scores(y, predicted, classes):
    result = {"test_samples": len(y), "accuracy": float(accuracy_score(y, predicted))}
    for average in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y, predicted, labels=np.arange(len(classes)), average=average, zero_division=0)
        result.update({f"precision_{average}": float(p), f"recall_{average}": float(r), f"f1_{average}": float(f)})
    return result


def main():
    baseline_test = pd.read_csv(ROOT / "evaluation_results/evaluation_summary.csv").set_index("dataset")
    comparison, history, class_rows = [], [], []
    for name in NAMES:
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        source = ROOT / "backend/models" / f"{name}_100pct/model.pt"
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        data_path = ROOT / "datasets/processed" / name / "data.npz"
        prep_path = data_path.parent / "preprocessor.joblib"
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        artifact = joblib.load(prep_path)
        with np.load(data_path, allow_pickle=False) as data:
            # Test tensors are intentionally not read until checkpoint selection is over.
            train_x, train_y = data["X_train"].astype(np.float32), data["y_train"].astype(np.int64)
            val_x, val_y = data["X_val"].astype(np.float32), data["y_val"].astype(np.int64)
            classes = data["classes"].astype(str).tolist()
        assert checkpoint["dataset"] == name
        assert checkpoint["classes"] == artifact["classes"] == classes
        assert train_x.shape[1:] == val_x.shape[1:] == (checkpoint["num_features"], 1)
        assert len(artifact["preprocessor"].get_feature_names_out()) == checkpoint["num_features"]
        labels = np.arange(len(classes))
        model = TrafficCNN(checkpoint["num_features"], len(classes))
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        base_pred = predict(model, val_x)
        base_f1 = float(f1_score(val_y, base_pred, labels=labels, average="macro", zero_division=0))
        base_accuracy = float(accuracy_score(val_y, base_pred))
        print(f"{name}: baseline validation accuracy={base_accuracy:.4f} macro_F1={base_f1:.4f}", flush=True)
        counts = np.bincount(train_y, minlength=len(classes))
        assert counts.min() > 0
        weights = np.sqrt(len(train_y) / (len(classes) * counts))
        weights = torch.tensor(weights / weights.mean(), dtype=torch.float32)
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=.5)
        batches = DataLoader(TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)), batch_size=256, shuffle=True, num_workers=0)
        best_score, best_eligible, stale, selected = base_f1, base_f1 + .005, 0, None
        started = time.monotonic()
        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            loss_sum = 0.
            for xb, yb in batches:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(xb), yb)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"{name}: nonfinite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                loss_sum += loss.item() * len(yb)
            predicted = predict(model, val_x)
            val_f1 = float(f1_score(val_y, predicted, labels=labels, average="macro", zero_division=0))
            val_accuracy = float(accuracy_score(val_y, predicted))
            scheduler.step(val_f1)
            history.append({"dataset": name, "epoch": epoch, "train_weighted_loss": loss_sum / len(train_y), "validation_accuracy": val_accuracy, "validation_macro_f1": val_f1})
            pd.DataFrame(history).to_csv(OUT / "training_history.csv", index=False)
            if val_f1 > best_eligible and val_accuracy >= base_accuracy - .01:
                best_eligible = val_f1
                selected = {"state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, "epoch": epoch, "f1": val_f1, "accuracy": val_accuracy}
            if val_f1 > best_score + .001:
                best_score, stale = val_f1, 0
            else:
                stale += 1
            print(f"{name} epoch {epoch:02d}: val_accuracy={val_accuracy:.4f} val_macro_F1={val_f1:.4f} selected={selected['epoch'] if selected else 'none'} elapsed={time.monotonic()-started:.0f}s", flush=True)
            if stale >= PATIENCE:
                break
        row = {"dataset": name, "baseline_validation_accuracy": base_accuracy, "baseline_validation_macro_f1": base_f1, "baseline_test_accuracy": float(baseline_test.loc[name, "accuracy"]), "baseline_test_macro_f1": float(baseline_test.loc[name, "f1_macro"]), "training_seconds": round(time.monotonic()-started, 1)}
        if selected is None:
            row.update({"recommendation": "keep_original", "reason": "No checkpoint met the validation improvement and accuracy guard."})
        else:
            destination = OUT / name
            destination.mkdir(exist_ok=True)
            candidate = dict(checkpoint)
            candidate["model_state_dict"] = selected["state"]
            candidate["fine_tuning"] = {"source_checkpoint": str(source.relative_to(ROOT)), "source_sha256": source_hash, "strategy": "inverse-sqrt-frequency weighted loss", "learning_rate": 1e-4, "seed": SEED, "best_epoch": selected["epoch"], "validation_macro_f1": selected["f1"], "validation_accuracy": selected["accuracy"], "selection": "validation macro F1 gain > 0.005; accuracy loss <= 0.01"}
            torch.save(candidate, destination / "model.pt")
            model.load_state_dict(selected["state"], strict=True)
            with np.load(data_path, allow_pickle=False) as data:
                test_x, test_y = data["X_test"].astype(np.float32), data["y_test"].astype(np.int64)
            test_predicted = predict(model, test_x)
            test_scores = scores(test_y, test_predicted, classes)
            report = classification_report(test_y, test_predicted, labels=labels, target_names=classes, output_dict=True, zero_division=0)
            for class_id, label in enumerate(classes):
                r = report[label]
                class_rows.append({"dataset": name, "class_id": class_id, "class": label, "precision": r["precision"], "recall": r["recall"], "f1_score": r["f1-score"], "support": int(r["support"])})
            (destination / "metrics.json").write_text(json.dumps(test_scores, indent=2), encoding="utf-8")
            row.update({"selected_epoch": selected["epoch"], "candidate_validation_accuracy": selected["accuracy"], "candidate_validation_macro_f1": selected["f1"], **{("candidate_test_samples" if k == "test_samples" else f"candidate_test_{k}"): v for k, v in test_scores.items()}})
            improved = test_scores["f1_macro"] > row["baseline_test_macro_f1"] and test_scores["accuracy"] >= row["baseline_test_accuracy"] - .01
            row.update({"recommendation": "candidate_improved_not_deployed" if improved else "keep_original", "reason": "Selected on validation only; test comparison is a one-time assessment, not used to tune epochs."})
            print(f"{name} FINAL: accuracy={test_scores['accuracy']:.4f} macro_F1={test_scores['f1_macro']:.4f} {row['recommendation']}", flush=True)
        assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash, "Original checkpoint changed"
        comparison.append(row)
        pd.DataFrame(comparison).to_csv(OUT / "comparison.csv", index=False)
        pd.DataFrame(class_rows).to_csv(OUT / "classification_report.csv", index=False)
    lines = ["# TrafficFlow candidate improvement", "", "Original serving checkpoints and application code are unchanged. Candidates are isolated in this directory and are not deployed.", "", "The original architecture, fitted preprocessing and existing train/validation/test splits are retained. Fine-tuning starts from the 100%-label checkpoints, uses inverse-square-root class-frequency weighted cross entropy, AdamW (initial LR 0.0001), and validation-only checkpoint selection. Maximum 12 epochs; early stopping after 4 epochs without macro-F1 improvement. No new test-driven search was performed.", "", "A candidate must improve validation macro F1 by more than 0.5 percentage points while losing no more than 1 percentage point of validation accuracy. Its selected checkpoint is then evaluated once on the existing test set. 'Keep original' means no acceptable validation checkpoint, or no corresponding test macro-F1 improvement within the accuracy guard.", "", "| Dataset | Original accuracy | Candidate accuracy | Original macro F1 | Candidate macro F1 | Recommendation |", "|---|---:|---:|---:|---:|---|"]
    for r in comparison:
        acc = f"{r['candidate_test_accuracy']*100:.2f}%" if "candidate_test_accuracy" in r else "Not evaluated"
        f1 = f"{r['candidate_test_f1_macro']*100:.2f}%" if "candidate_test_f1_macro" in r else "Not evaluated"
        lines.append(f"| {r['dataset']} | {r['baseline_test_accuracy']*100:.2f}% | {acc} | {r['baseline_test_macro_f1']*100:.2f}% | {f1} | {r['recommendation']} |")
    lines += ["", "The test set was evaluated in the preceding baseline assessment, so this is a post-development comparison, not an independent new generalization estimate. Use a new untouched test set or external validation before making deployment claims.", "", "Files: `comparison.csv`, `classification_report.csv`, `training_history.csv`, and per-dataset `model.pt` / `metrics.json` for validation-eligible candidates.", ""]
    (OUT / "improvement_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(pd.DataFrame(comparison).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
