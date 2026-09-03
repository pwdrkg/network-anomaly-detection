"""
FastAPI service for network-traffic anomaly detection.

Loads the fitted preprocessor and the selected model once at startup and
exposes:
  GET  /            -> web UI
  GET  /health      -> liveness + whether artifacts loaded
  GET  /model-info  -> model metadata + decision threshold
  POST /predict     -> score one connection (JSON) -> attack probability + label

Run locally:
    uvicorn app.main:app --reload --port 8000
"""
import os
import sys
import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# make src/ importable so we reuse the exact training-time transform
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import preprocessing as pp  # noqa: E402

from .schemas import Connection, PredictionResponse  # noqa: E402

MODEL_DIR = os.environ.get("MODEL_DIR", str(ROOT / "models"))
THRESHOLD = float(os.environ.get("THRESHOLD", "0.5"))
MODEL_NAME = os.environ.get("MODEL_NAME", "best_model")  # best_model | xgboost

app = FastAPI(title="Network Anomaly Detector",
              description="Detect anomalies (attacks) in network traffic — UNSW-NB15.",
              version="1.0.0")

STATE = {"preprocessor": None, "model": None, "features": None}


def load_artifacts():
    if STATE["model"] is None:
        STATE["preprocessor"] = joblib.load(os.path.join(MODEL_DIR, "preprocessor.joblib"))
        STATE["model"] = joblib.load(os.path.join(MODEL_DIR, f"{MODEL_NAME}.joblib"))
        STATE["features"] = STATE["preprocessor"]["final_columns"]
    return STATE


# load once at import so the model is ready under uvicorn and TestClient alike
load_artifacts()


def _score(record: dict, threshold: float):
    df = pd.DataFrame([record])
    X = pp.transform(df, STATE["preprocessor"], with_target=False)
    proba = float(STATE["model"].predict_proba(X)[0, 1])
    return proba, int(proba >= threshold)


@app.get("/health")
def health():
    ok = STATE["model"] is not None and STATE["preprocessor"] is not None
    return {"status": "ok" if ok else "loading", "model_loaded": ok,
            "model": MODEL_NAME, "threshold": THRESHOLD}


@app.get("/model-info")
def model_info():
    if STATE["model"] is None:
        raise HTTPException(503, "model not loaded")
    return {"model": MODEL_NAME, "type": type(STATE["model"]).__name__,
            "n_features": len(STATE["features"]), "threshold": THRESHOLD,
            "primary_metric": "recall (attack class) + PR-AUC",
            "reported_test_recall": 0.974, "reported_test_pr_auc": 0.989}


@app.post("/predict", response_model=PredictionResponse)
def predict(conn: Connection, threshold: float | None = None):
    if STATE["model"] is None:
        raise HTTPException(503, "model not loaded")
    t = THRESHOLD if threshold is None else float(threshold)
    try:
        proba, pred = _score(conn.model_dump(), t)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"scoring failed: {e}")
    return PredictionResponse(
        attack_probability=round(proba, 4),
        prediction=pred,
        label="attack" if pred == 1 else "normal",
        threshold=t,
    )


# ---- web UI ----
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    return (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")
