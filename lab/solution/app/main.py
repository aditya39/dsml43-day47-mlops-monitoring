"""app/main.py — API prediksi harga rumah, versi Day 47.

Bedanya dengan Day 46:
  1. Model diambil dari MLflow Registry (models:/harga-rumah@champion), bukan file .joblib di image.
  2. Setiap request /predict menghasilkan SATU baris log JSON.
  3. Endpoint /metrics untuk Prometheus; /admin/reload untuk ganti model tanpa build ulang image.
"""
from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.model_loader import MODEL_ALIAS, MODEL_NAME, ModelStore
from app.logging_config import get_logger, setup_logging
from app.metrics import FEATURE_JARAK, LATENCY, MODEL_INFO, PREDICTED_PRICE, PREDICTIONS, REQUESTS
from app.schemas import HouseFeatures, PredictionResponse

setup_logging()
log = get_logger("api")
store = ModelStore()


def _update_model_info() -> None:
    MODEL_INFO.clear()
    if store.current:
        MODEL_INFO.labels(version=store.current.version).set(1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.try_load()  # registry masih kosong? API tetap hidup, /predict menjawab 503
    _update_model_info()
    yield


app = FastAPI(
    title="Harga Rumah API",
    description="Prediksi harga rumah Jabodetabek — Day 47: model dari MLflow Registry + monitoring",
    version="2.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def observe_requests(request: Request, call_next):
    """Hitung request & latency untuk SEMUA endpoint (kecuali /metrics itu sendiri)."""
    start = time.perf_counter()
    response = await call_next(request)
    route = request.scope.get("route")
    endpoint = route.path if route else "unmatched"  # template route -> label tidak meledak
    if endpoint != "/metrics":
        REQUESTS.labels(endpoint=endpoint, status=str(response.status_code)).inc()
        LATENCY.labels(endpoint=endpoint).observe(time.perf_counter() - start)
        if response.status_code >= 400:
            log.warning("request_failed", extra={"fields": {
                "endpoint": endpoint, "status": response.status_code}})
    return response


@app.get("/")
def root():
    return {"message": "API Prediksi Harga Rumah Jabodetabek", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    cur = store.current
    return {
        "status": "ok" if cur else "degraded",
        "model_name": MODEL_NAME,
        "model_alias": MODEL_ALIAS,
        "model_version": cur.version if cur else None,
        "run_id": cur.run_id if cur else None,
        "detail": None if cur else store.last_error,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: HouseFeatures, request: Request, response: Response):
    current = store.current
    if current is None:
        raise HTTPException(503, "Model belum tersedia: latih & daftarkan model, lalu POST /admin/reload")
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    start = time.perf_counter()
    try:
        X = pd.DataFrame([features.model_dump()])
        harga = float(current.model.predict(X)[0])
    except Exception:
        log.exception("prediction_failed", extra={"fields": {
            "request_id": request_id, "model_version": current.version}})
        raise HTTPException(500, "Prediksi gagal") from None
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    # Satu baris log JSON per request: cukup untuk audit & bahan laporan drift
    log.info("prediction", extra={"fields": {
        "request_id": request_id,
        "model_version": current.version,
        "features": features.model_dump(),
        "prediction": round(harga),
        "latency_ms": latency_ms,
    }})
    # Metrik untuk Prometheus -> Grafana
    PREDICTIONS.labels(model_version=current.version).inc()
    PREDICTED_PRICE.observe(harga)
    FEATURE_JARAK.observe(features.jarak_ke_pusat_km)

    response.headers["X-Request-ID"] = request_id
    return PredictionResponse(harga_prediksi=round(harga), model_version=current.version,
                              request_id=request_id, latency_ms=latency_ms)


@app.post("/admin/reload")
def reload_model():
    """Muat ulang @champion dari registry (dipanggil setelah promote / rollback).
    Catatan produksi: endpoint admin seperti ini WAJIB dilindungi (auth / jaringan internal)."""
    try:
        old, new = store.reload()
    except Exception as err:  # noqa: BLE001
        log.error("model_reload_failed", extra={"fields": {"error": str(err)[:300]}})
        raise HTTPException(503, f"Gagal memuat @{MODEL_ALIAS}: {err.__class__.__name__}") from None
    _update_model_info()
    log.info("model_reloaded", extra={"fields": {"old_version": old, "new_version": new}})
    return {"old_version": old, "new_version": new}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
