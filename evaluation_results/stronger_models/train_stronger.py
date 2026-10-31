"""Bounded validation-led CNN training. All outputs stay outside the serving app."""
from pathlib import Path
import copy
import hashlib
import json
import random
import sys
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))
from train_cnn import TrafficCNN

NAMES = ["UNSW_NB15", "ISCX_15s", "ISCX_30s", "ISCX_60s", "ISCX_120s"]
torch.set_num_threads(4)
MAX_EPOCHS, PATIENCE = 50, 8


def logits(model, x):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.from_numpy(np.ascontiguousarray(x[i:i+512]))).numpy() for i in range(0, len(x), 512)])


class Backbone(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.layers = model.net[:12]

    def forward(self, x):
        return self.layers(x.transpose(1, 2))


def measures(y, predicted, labels):
    result = {"test_samples": len(y), "accuracy": float(accuracy_score(y, predicted))}
    for average in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y, predicted, labels=labels, average=average, zero_division=0)
        result.update({f"precision_{average}": float(p), f"recall_{average}": float(r), f"f1_{average}": float(f)})
    return result


def main():
    summary, history, validation, class_rows = [], [], [], []
    originals = pd.read_csv(ROOT / "evaluation_results/evaluation_summary.csv").set_index("dataset")
    previous = pd.read_csv(ROOT / "evaluation_results/improvement/comparison.csv").set_index("dataset")
    for name in NAMES:
        started = time.monotonic()
        source = ROOT / "backend/models" / f"{name}_100pct/model.pt"
        original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        data_path = ROOT / "datasets/processed" / name / "data.npz"
        artifact = joblib.load(data_path.parent / "preprocessor.joblib")
        with np.load(data_path, allow_pickle=False) as saved:
            xtr, ytr = saved["X_train"].astype(np.float32), saved["y_train"].astype(np.int64)
            xv, yv = saved["X_val"].astype(np.float32), saved["y_val"].astype(np.int64)
            classes = saved["classes"].astype(str).tolist()
        assert checkpoint["dataset"] == name and checkpoint["classes"] == artifact["classes"] == classes
        assert xtr.shape[1:] == xv.shape[1:] == (checkpoint["num_features"], 1)
        assert len(artifact["preprocessor"].get_feature_names_out()) == checkpoint["num_features"]
        labels = np.arange(len(classes))
        model = TrafficCNN(checkpoint["num_features"], len(classes))
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        base_logits = logits(model, xv)
        base_acc = float(accuracy_score(yv, base_logits.argmax(1)))
        base_f1 = float(f1_score(yv, base_logits.argmax(1), labels=labels, average="macro", zero_division=0))
        options = [("original", copy.deepcopy(checkpoint["model_state_dict"]), base_logits)]
        previous_path = ROOT / "evaluation_results/improvement" / name / "model.pt"
        if previous_path.is_file():
            old = torch.load(previous_path, map_location="cpu", weights_only=False)
            assert old["classes"] == classes
            model.load_state_dict(old["model_state_dict"], strict=True)
            options.append(("previous_finetune", copy.deepcopy(old["model_state_dict"]), logits(model, xv)))
        eligible_starts = [o for o in options if accuracy_score(yv, o[2].argmax(1)) >= base_acc - .01]
        start = max(eligible_starts, key=lambda o: f1_score(yv, o[2].argmax(1), labels=labels, average="macro", zero_division=0))
        counts = np.bincount(ytr, minlength=len(classes))
        assert counts.min() > 0
        log_prior = np.log(counts / counts.sum()).astype(np.float32)
        log_prior -= log_prior.mean()
        head_only = name == "UNSW_NB15"
        if head_only:
            model.load_state_dict(start[1], strict=True)
            print(f"{name}: caching frozen convolutional features...", flush=True)
            train_features = logits(Backbone(model), xtr)
            val_features = logits(Backbone(model), xv)
        print(f"{name}: original val accuracy={base_acc:.4f}, macro_F1={base_f1:.4f}; start={start[0]}", flush=True)
        for power in (0., .5):
            random.seed(43)
            np.random.seed(43)
            torch.manual_seed(43)
            model = TrafficCNN(checkpoint["num_features"], len(classes))
            model.load_state_dict(start[1], strict=True)
            learner = model.net[12:] if head_only else model
            if head_only:
                for parameter in model.net[:12].parameters():
                    parameter.requires_grad_(False)
            tx, vx = (train_features, val_features) if head_only else (xtr, xv)
            weights = (len(ytr) / (len(classes) * counts)) ** power
            weights = torch.tensor(weights / weights.mean(), dtype=torch.float32)
            criterion = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=.02)
            optimizer = torch.optim.AdamW(learner.parameters(), lr=1e-3 if head_only else 3e-4, weight_decay=1e-3)
            schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=.5)
            batches = DataLoader(TensorDataset(torch.from_numpy(tx), torch.from_numpy(ytr)), batch_size=512 if head_only else 256, shuffle=True, num_workers=0)
            best, stale = -1., 0
            best_state, best_logits, best_tag = None, None, None
            best_guarded, guarded_state, guarded_logits, guarded_tag = -1., None, None, None
            for epoch in range(1, MAX_EPOCHS + 1):
                learner.train()
                for xb, yb in batches:
                    optimizer.zero_grad(set_to_none=True)
                    loss = criterion(learner(xb), yb)
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite training loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(learner.parameters(), 5.)
                    optimizer.step()
                scores = logits(learner, vx)
                vf = float(f1_score(yv, scores.argmax(1), labels=labels, average="macro", zero_division=0))
                va = float(accuracy_score(yv, scores.argmax(1)))
                tag = f"{'head' if head_only else 'full'}_power{power}_epoch{epoch}"
                history.append({"dataset": name, "strategy": "head" if head_only else "full", "class_weight_power": power, "epoch": epoch, "validation_accuracy": va, "validation_macro_f1": vf})
                pd.DataFrame(history).to_csv(OUT / "training_history.csv", index=False)
                schedule.step(vf)
                if vf > best + .001:
                    best, stale = vf, 0
                    best_state, best_logits, best_tag = copy.deepcopy(model.state_dict()), scores, tag
                else:
                    stale += 1
                if va >= base_acc - .01 and vf > best_guarded:
                    best_guarded, guarded_state, guarded_logits, guarded_tag = vf, copy.deepcopy(model.state_dict()), scores, tag + "_guarded"
                if epoch == 1 or epoch % 5 == 0 or stale >= PATIENCE:
                    print(f"{name} power={power} epoch={epoch}: val_acc={va:.4f} val_macro_F1={vf:.4f} elapsed={time.monotonic()-started:.0f}s", flush=True)
                if stale >= PATIENCE:
                    break
            if best_state is not None:
                options.append((best_tag, best_state, best_logits))
            if guarded_state is not None:
                options.append((guarded_tag, guarded_state, guarded_logits))
        # A small global prior-correction grid uses only training priors and validation labels.
        # The chosen offset is absorbed into the final Linear bias, keeping inference compatible.
        selected = None
        for tag, state, scores in options:
            for tau in (-.25, 0., .25, .5):
                adjusted = scores - tau * log_prior[None, :]
                va = float(accuracy_score(yv, adjusted.argmax(1)))
                vf = float(f1_score(yv, adjusted.argmax(1), labels=labels, average="macro", zero_division=0))
                validation.append({"dataset": name, "checkpoint": tag, "prior_tau": tau, "validation_accuracy": va, "validation_macro_f1": vf, "passes_accuracy_guard": va >= base_acc - .01})
                if va >= base_acc - .01 and (selected is None or vf > selected["f1"]):
                    selected = {"tag": tag, "state": state, "tau": tau, "f1": vf, "accuracy": va}
        pd.DataFrame(validation).to_csv(OUT / "validation_selection.csv", index=False)
        row = {"dataset": name, "original_accuracy": float(originals.loc[name, "accuracy"]), "original_macro_f1": float(originals.loc[name, "f1_macro"]), "original_validation_accuracy": base_acc, "original_validation_macro_f1": base_f1, "training_seconds": round(time.monotonic()-started, 1)}
        if selected["f1"] <= base_f1 + .005:
            row["status"] = "no_qualifying_gain_keep_original"
        else:
            state = copy.deepcopy(selected["state"])
            state["net.16.bias"] -= torch.from_numpy(selected["tau"] * log_prior)
            model = TrafficCNN(checkpoint["num_features"], len(classes))
            model.load_state_dict(state, strict=True)
            model.eval()
            chosen_val = logits(model, xv).argmax(1)
            assert abs(f1_score(yv, chosen_val, labels=labels, average="macro", zero_division=0)-selected["f1"]) < 1e-10
            candidate = dict(checkpoint)
            candidate["model_state_dict"] = state
            candidate["improvement"] = {"strategy": selected["tag"], "prior_tau_absorbed_in_bias": selected["tau"], "source_sha256": original_hash, "validation_accuracy": selected["accuracy"], "validation_macro_f1": selected["f1"], "selection": "highest validation macro F1 subject to accuracy >= original minus 0.01", "architecture_unchanged": True}
            destination = OUT / name
            destination.mkdir(exist_ok=True)
            torch.save(candidate, destination / "model.pt")
            with np.load(data_path, allow_pickle=False) as saved:
                xt, yt = saved["X_test"].astype(np.float32), saved["y_test"].astype(np.int64)
            predicted = logits(model, xt).argmax(1)
            result = measures(yt, predicted, labels)
            report = classification_report(yt, predicted, labels=labels, target_names=classes, output_dict=True, zero_division=0)
            for i, label in enumerate(classes):
                r = report[label]
                class_rows.append({"dataset": name, "class_id": i, "class": label, "precision": r["precision"], "recall": r["recall"], "f1_score": r["f1-score"], "support": int(r["support"])})
            cm = confusion_matrix(yt, predicted, labels=labels)
            pd.DataFrame(cm, index=classes, columns=classes).to_csv(destination / "confusion_matrix.csv", index_label="actual/predicted")
            (destination / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            row.update({"selected_strategy": selected["tag"], "prior_tau": selected["tau"], "validation_accuracy": selected["accuracy"], "validation_macro_f1": selected["f1"], **result})
            row["status"] = "improved_candidate_not_deployed" if result["f1_macro"] > row["original_macro_f1"] and result["accuracy"] >= row["original_accuracy"]-.01 else "test_did_not_confirm_keep_original"
            if name in previous.index and previous.loc[name, "recommendation"] == "candidate_improved_not_deployed":
                row["previous_candidate_accuracy"] = float(previous.loc[name, "candidate_test_accuracy"])
                row["previous_candidate_macro_f1"] = float(previous.loc[name, "candidate_test_f1_macro"])
            print(f"{name} FINAL: accuracy={result['accuracy']:.4f} macro_F1={result['f1_macro']:.4f} {row['status']}", flush=True)
        assert hashlib.sha256(source.read_bytes()).hexdigest() == original_hash
        summary.append(row)
        pd.DataFrame(summary).to_csv(OUT / "comparison.csv", index=False)
        pd.DataFrame(class_rows).to_csv(OUT / "classification_report.csv", index=False)
    lines = ["# Stronger CNN training results", "", "Existing serving checkpoints, application code, datasets, preprocessing and class mappings remain unchanged. All candidates retain the original TrafficCNN-1D architecture and state-dictionary keys.", "", "Training used existing training splits only. Validation selected starts, regularized fine-tuning checkpoints, and a small global class-prior correction. UNSW-NB15 used frozen convolutional embeddings with head fine-tuning; ISCX used full-model fine-tuning. Each training arm used natural or square-root-inverse-frequency weighted cross entropy, 0.02 label smoothing, AdamW with weight decay 0.001, validation-controlled learning-rate reduction and early stopping (maximum 50 epochs, patience 8). The prior correction uses training class frequencies and is absorbed into the existing final Linear bias.", "", "All checkpoint and prior-correction selection used validation macro F1, with validation accuracy no worse than one percentage point below the original. The selected checkpoint for each dataset was assessed once on its existing test split; test results did not select epochs or correction strength. Previous development already examined these test sets, so this comparison is not independent confirmation on new data.", "", "| Dataset | Original accuracy | Candidate accuracy | Original macro F1 | Candidate macro F1 | Status |", "|---|---:|---:|---:|---:|---|"]
    for r in summary:
        a = f"{r['accuracy']*100:.2f}%" if "accuracy" in r else "Not evaluated"
        f = f"{r['f1_macro']*100:.2f}%" if "f1_macro" in r else "Not evaluated"
        lines.append(f"| {r['dataset']} | {r['original_accuracy']*100:.2f}% | {a} | {r['original_macro_f1']*100:.2f}% | {f} | {r['status']} |")
    lines += ["", "Outputs: comparison.csv, classification_report.csv, training_history.csv, validation_selection.csv, and per-dataset model.pt / metrics.json / confusion_matrix.csv when a checkpoint qualified. These checkpoints are not deployed. Original checkpoint SHA-256 values were verified unchanged.", ""]
    (OUT / "stronger_training_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(pd.DataFrame(summary).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
