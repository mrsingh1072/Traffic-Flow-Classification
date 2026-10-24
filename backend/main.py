from functools import lru_cache
from pathlib import Path
from typing import Any
import json
import os

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASETS = {
    "UNSW_NB15": {"folder": "UNSW_NB15", "model": "UNSW_NB15_100pct"},
    "ISCX_15s": {"folder": "ISCX_15s", "model": "ISCX_15s_100pct"},
    "ISCX_30s": {"folder": "ISCX_30s", "model": "ISCX_30s_100pct"},
    "ISCX_60s": {"folder": "ISCX_60s", "model": "ISCX_60s_100pct"},
    "ISCX_120s": {"folder": "ISCX_120s", "model": "ISCX_120s_100pct"},
}

DEFAULT_ORIGINS = (
    "http://localhost:5173,http://127.0.0.1:5173,"
    "http://localhost:3000,http://127.0.0.1:3000"
)
ALLOWED_ORIGINS = [
    item.strip()
    for item in os.getenv("FRONTEND_ORIGINS", DEFAULT_ORIGINS).split(",")
    if item.strip()
]


class TrafficCNN(nn.Module):
    def __init__(self, n_features: int, n_classes: int):
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
        return self.net(x.transpose(1, 2))


class PredictRequest(BaseModel):
    dataset: str = "UNSW_NB15"
    features: dict[str, Any]


class BatchPredictRequest(BaseModel):
    dataset: str = "UNSW_NB15"
    records: list[dict[str, Any]] = Field(..., min_length=1, max_length=100)


app = FastAPI(
    title="5G Traffic Flow Classification API",
    description="CNN-based network traffic classification across five datasets.",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@lru_cache(maxsize=5)
def load_bundle(dataset: str):
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset}")

    config = DATASETS[dataset]
    preprocessor_path = ROOT / "datasets" / "processed" / config["folder"] / "preprocessor.joblib"
    model_path = ROOT / "backend" / "models" / config["model"] / "model.pt"

    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessor missing for {dataset}: {preprocessor_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model missing for {dataset}: {model_path}")

    artifact = joblib.load(preprocessor_path)
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    preprocessor = artifact["preprocessor"]
    feature_columns = list(artifact["feature_columns"])
    classes = [str(c) for c in artifact["classes"]]

    if classes != [str(c) for c in checkpoint["classes"]]:
        raise RuntimeError(f"Class ordering mismatch for {dataset}")

    numeric_columns, categorical_columns = [], []
    for name, transformer, columns in preprocessor.transformers_:
        if name == "numeric":
            numeric_columns = list(columns)
        elif name == "categorical":
            categorical_columns = list(columns)

    n_features = int(checkpoint["num_features"])
    n_classes = int(checkpoint["num_classes"])
    if len(classes) != n_classes:
        raise RuntimeError(f"Class count mismatch for {dataset}")

    model = TrafficCNN(n_features, n_classes).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return {
        "preprocessor": preprocessor,
        "feature_columns": feature_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "classes": classes,
        "model": model,
        "num_features": n_features,
        "num_classes": n_classes,
    }


def prepare_features(dataset: str, supplied: dict[str, Any]) -> np.ndarray:
    bundle = load_bundle(dataset)
    if not supplied:
        raise HTTPException(status_code=422, detail="The features object cannot be empty.")

    expected = bundle["feature_columns"]
    unknown = sorted(set(supplied) - set(expected))
    missing = sorted(set(expected) - set(supplied))
    if unknown or missing:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Input feature names do not match the dataset schema.",
                "missing_features": missing,
                "unknown_features": unknown,
                "schema_endpoint": f"/schema?dataset={dataset}",
            },
        )
    if all(value is None for value in supplied.values()):
        raise HTTPException(status_code=422, detail="At least one feature must have a non-null value.")

    frame = pd.DataFrame([{c: supplied[c] for c in expected}], columns=expected)
    for column in bundle["numeric_columns"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in bundle["categorical_columns"]:
        values = frame[column].astype("string")
        frame[column] = values.astype(object).where(values.notna(), np.nan)

    try:
        transformed = np.asarray(bundle["preprocessor"].transform(frame), dtype=np.float32)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Input preprocessing failed: {exc}") from exc

    if transformed.shape != (1, bundle["num_features"]):
        raise HTTPException(status_code=500, detail="Transformed feature count does not match the model.")
    if not np.isfinite(transformed).all():
        raise HTTPException(status_code=422, detail="Preprocessing produced non-finite feature values.")
    return transformed


def predict_one(dataset: str, supplied: dict[str, Any]) -> dict[str, Any]:
    bundle = load_bundle(dataset)
    transformed = prepare_features(dataset, supplied)
    tensor = torch.from_numpy(transformed).unsqueeze(-1).to(DEVICE)
    with torch.no_grad():
        logits = bundle["model"](tensor)
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()

    predicted_id = int(np.argmax(probabilities))
    classes = bundle["classes"]
    return {
        "dataset": dataset,
        "model": "TrafficCNN-1D",
        "predicted_class": classes[predicted_id],
        "class_id": predicted_id,
        "confidence": round(float(probabilities[predicted_id]), 6),
        "probabilities": {
            classes[i]: round(float(probabilities[i]), 6)
            for i in range(len(classes))
        },
        "note": "Confidence is a softmax score, not a calibrated probability.",
    }


@app.get("/")
def home():
    return {
        "message": "5G Traffic Flow Classification API",
        "version": app.version,
        "docs": "/docs",
        "health": "/health",
        "datasets": "/datasets",
        "metrics": "/metrics",
    }


@app.get("/health")
def health():
    missing = []
    for name, config in DATASETS.items():
        pp = ROOT / "datasets" / "processed" / config["folder"] / "preprocessor.joblib"
        mp = ROOT / "backend" / "models" / config["model"] / "model.pt"
        if not pp.exists():
            missing.append(f"{name}: preprocessor")
        if not mp.exists():
            missing.append(f"{name}: model")
    return {
        "status": "healthy" if not missing else "degraded",
        "device": str(DEVICE),
        "datasets_configured": len(DATASETS),
        "missing_artifacts": missing,
    }


@app.get("/datasets")
def list_datasets():
    result = []
    for name, config in DATASETS.items():
        pp = ROOT / "datasets" / "processed" / config["folder"] / "preprocessor.joblib"
        mp = ROOT / "backend" / "models" / config["model"] / "model.pt"
        result.append({
            "dataset": name,
            "model": "TrafficCNN-1D",
            "label_fraction": "100%",
            "preprocessor_available": pp.exists(),
            "model_available": mp.exists(),
        })
    return {"datasets": result}


@app.get("/schema")
def schema(dataset: str = Query(default="UNSW_NB15")):
    if dataset not in DATASETS:
        raise HTTPException(status_code=400, detail={
            "message": "Unsupported dataset.",
            "supported_datasets": list(DATASETS),
        })
    try:
        bundle = load_bundle(dataset)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    types = {}
    for column in bundle["feature_columns"]:
        if column in bundle["numeric_columns"]:
            types[column] = "numeric"
        elif column in bundle["categorical_columns"]:
            types[column] = "categorical"
        else:
            types[column] = "unknown"
    return {
        "dataset": dataset,
        "required_feature_count": len(bundle["feature_columns"]),
        "features": types,
        "note": "Supply all listed raw feature names. Null values are accepted where the fitted imputer can handle them.",
    }


@app.post("/predict")
def predict(request: PredictRequest):
    if request.dataset not in DATASETS:
        raise HTTPException(status_code=400, detail={
            "message": "Unsupported dataset.",
            "supported_datasets": list(DATASETS),
        })
    return predict_one(request.dataset, request.features)


@app.post("/predict/batch")
def predict_batch(request: BatchPredictRequest):
    if request.dataset not in DATASETS:
        raise HTTPException(status_code=400, detail={
            "message": "Unsupported dataset.",
            "supported_datasets": list(DATASETS),
        })
    return {
        "dataset": request.dataset,
        "count": len(request.records),
        "predictions": [predict_one(request.dataset, row) for row in request.records],
    }


@app.get("/metrics")
def metrics(dataset: str | None = Query(default=None)):
    selected = [dataset] if dataset else list(DATASETS)
    unknown = sorted(set(selected) - set(DATASETS))
    if unknown:
        raise HTTPException(status_code=400, detail={
            "message": "Unsupported dataset.",
            "unknown": unknown,
            "supported_datasets": list(DATASETS),
        })

    results = []
    for name in selected:
        path = ROOT / "backend" / "results" / "all_datasets_evaluation" / name / "metrics.json"
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Metrics not found for {name}. Run evaluate_all_datasets.py first.")
        with open(path, "r", encoding="utf-8") as file:
            results.append(json.load(file))
    return {"metrics": results}


@app.get("/metrics/{dataset}/classes")
def class_metrics(dataset: str):
    if dataset not in DATASETS:
        raise HTTPException(status_code=400, detail={"supported_datasets": list(DATASETS)})
    path = ROOT / "backend" / "results" / "all_datasets_evaluation" / dataset / "classification_report.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Class report not found for {dataset}")
    report = pd.read_csv(path, index_col=0).fillna(0)
    result = []
    for index, row in report.iterrows():
        if index in {"accuracy", "macro avg", "weighted avg"}:
            continue
        result.append({
            "class_name": str(index),
            "precision": float(row.get("precision", 0)),
            "recall": float(row.get("recall", 0)),
            "f1_score": float(row.get("f1-score", 0)),
            "support": int(row.get("support", 0)),
        })
    return {"dataset": dataset, "classes": result}


@app.get("/metrics/{dataset}/confusion-matrix")
def confusion_matrix_data(dataset: str):
    if dataset not in DATASETS:
        raise HTTPException(status_code=400, detail={"supported_datasets": list(DATASETS)})
    path = ROOT / "backend" / "results" / "all_datasets_evaluation" / dataset / "confusion_matrix.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Confusion matrix not found for {dataset}")
    matrix = pd.read_csv(path, index_col=0)
    return {
        "dataset": dataset,
        "labels": [str(x) for x in matrix.index],
        "matrix": matrix.astype(int).values.tolist(),
    }
