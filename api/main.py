"""Model serving API with Prometheus instrumentation.

    GET  /health                 liveness + loaded model version
    GET  /model                  champion metadata (metrics, top features)
    POST /predict                batch predictions from feature rows
    GET  /forecast               the latest published 24 h forecast
    GET  /metrics                Prometheus metrics (request latency, predictions, model version)
    POST /reload                 hot-swap to a newly exported champion without downtime
"""
from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

from gridops import __version__
from gridops.config import SITE_DATA_DIR, STATE_DIR
from gridops.registry import load_serving

SERVING_DIR = STATE_DIR / "serving"
STATE: dict = {}

REQUESTS = Counter("gridops_http_requests_total", "HTTP requests", ["path", "method", "status"])
LATENCY = Histogram("gridops_http_request_seconds", "Request latency", ["path"], buckets=(.005, .01, .025, .05, .1, .25, .5, 1, 2.5))
PREDICTIONS = Counter("gridops_predictions_total", "Rows predicted")
PRED_VALUE = Histogram("gridops_prediction_gco2_per_kwh", "Distribution of predicted carbon intensity", buckets=(25, 50, 75, 100, 150, 200, 250, 300, 400))
MODEL_VERSION = Gauge("gridops_model_version", "Champion model version currently served")
MODEL_HOLDOUT_MAE = Gauge("gridops_model_holdout_mae", "Holdout MAE of the served model")


def _load() -> None:
    bundle = load_serving(SERVING_DIR)
    meta = json.loads((SERVING_DIR / "model.json").read_text())
    STATE.update(bundle=bundle, meta=meta)
    MODEL_VERSION.set(float(bundle["version"]))
    MODEL_HOLDOUT_MAE.set(meta["metrics"]["mae"])


@asynccontextmanager
async def lifespan(_: FastAPI):
    if (SERVING_DIR / "model.joblib").exists():
        _load()
    yield


app = FastAPI(title="GridOps model server", version=__version__, lifespan=lifespan)


@app.middleware("http")
async def instrument(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    path = request.scope.get("route").path if request.scope.get("route") else "unmatched"
    LATENCY.labels(path).observe(time.perf_counter() - t0)
    REQUESTS.labels(path, request.method, str(response.status_code)).inc()
    return response


def _bundle() -> dict:
    if "bundle" not in STATE:
        raise HTTPException(503, "No champion model exported yet")
    return STATE["bundle"]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_version": STATE.get("bundle", {}).get("version"), "app_version": __version__,
            "pod": os.getenv("HOSTNAME", "local")}


@app.get("/model")
def model() -> dict:
    _bundle()
    m = STATE["meta"]
    return {"name": m["name"], "version": m["version"], "metrics": m["metrics"], "top_features": dict(list(m["importance"].items())[:8]),
            "n_features": len(STATE["bundle"]["features"])}


class PredictIn(BaseModel):
    rows: list[dict[str, float]] = Field(min_length=1, max_length=5000, description="Feature rows; see /model for feature names")


@app.post("/predict")
def predict(body: PredictIn) -> dict:
    b = _bundle()
    X = pd.DataFrame(body.rows)
    missing = [f for f in b["features"] if f not in X]
    if missing:
        raise HTTPException(422, f"Missing features: {missing[:10]}{'…' if len(missing) > 10 else ''}")
    pred = np.clip(b["model"].predict(X[b["features"]]), 0, None)
    PREDICTIONS.inc(len(pred))
    for v in pred:
        PRED_VALUE.observe(float(v))
    return {"model_version": b["version"], "predictions": [round(float(v), 2) for v in pred]}


@app.get("/forecast")
def forecast() -> dict:
    f = SITE_DATA_DIR / "console.json"
    if not f.exists():
        raise HTTPException(404, "No forecast published yet")
    c = json.loads(f.read_text())
    return {"model_version": c["champion"]["version"], "generated_at": c["generated_at"],
            "forecast": [{"time": t, "gco2_per_kwh": m, "eso": e} for t, m, e in zip(c["forecast"]["time"], c["forecast"]["model"], c["forecast"]["eso"], strict=True)]}


@app.post("/reload")
def reload() -> dict:
    old = STATE.get("bundle", {}).get("version")
    _load()
    return {"previous": old, "current": STATE["bundle"]["version"]}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
