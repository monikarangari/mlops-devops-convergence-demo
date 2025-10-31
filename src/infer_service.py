import json, os, joblib, time
import numpy as np
from fastapi import FastAPI
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from pydantic import BaseModel
from src.schema import PredictRequest
from src.drift import population_stability_index

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
META_PATH  = os.getenv("META_PATH",  "models/metadata.json")

app = FastAPI(title="Iris Model Service", version="1.0")

# Prometheus metrics
PREDICTIONS_TOTAL = Counter("predictions_total", "Total prediction requests")
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Prediction latency")
MODEL_F1 = Gauge("model_f1_macro", "Model F1 macro")
DRIFT_PSI = Gauge("drift_psi_feature", "Toy PSI drift per first feature")

# Load model & metadata at startup
clf = joblib.load(MODEL_PATH)
with open(META_PATH) as f:
    meta = json.load(f)
MODEL_F1.set(meta.get("f1_macro", 0.0))
FEATURE_NAMES = meta["feature_names"]
CLASSES = meta["classes"]

# Baseline distrib for toy drift check: capture training-like baseline
# In prod, store baseline in a feature store or artifact store.
BASELINE = None  # lazy-init

@app.on_event("startup")
def init_baseline():
    global BASELINE
    # use a synthetic gaussian baseline for demo
    rng = np.random.default_rng(42)
    BASELINE = rng.normal(0, 1, size=(200, 4))

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/predict")
def predict(req: PredictRequest):
    PREDICTIONS_TOTAL.inc()
    start = time.time()
    X = np.array(req.rows, dtype=float)
    pred = clf.predict(X).tolist()
    elapsed = time.time() - start
    PREDICTION_LATENCY.observe(elapsed)

    # Toy drift check on feature 0
    psi = population_stability_index(BASELINE[:,0], X[:,0])
    DRIFT_PSI.set(psi)
    return {"classes": CLASSES, "pred": pred, "latency_s": elapsed, "psi_feature0": psi}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
