"""
FastAPI – Tech Challenge F4 (API de inferência LSTM)

Este módulo expõe endpoints REST para previsões de preço (Close) da AMZN
com horizontes H=1 (baseline) e H=5 (multi-saída), além de healthcheck,
exposição de métricas Prometheus e leitura de metadados.

Endpoints:
- GET /health
- GET /metadata
- GET /metrics
- GET /features-order
- POST /predict
- POST /predict-ticker
"""
from __future__ import annotations

# --- sys.path para permitir `python api/main.py` ---
import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from typing import List
import json
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from loguru import logger
from tensorflow.keras.models import load_model  # type: ignore
from joblib import load as joblib_load
from prometheus_client import (
    Counter, Histogram, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
)

# Config e Schemas
from src.utils.config import settings
from api.schemas import PredictRequest, PredictResponse, PredictTickerRequest

# -------------------------------------------------------------------------------------
# Prometheus
# -------------------------------------------------------------------------------------
REGISTRY = CollectorRegistry()
HTTP_REQUESTS = Counter(
    "http_requests_total", "Total HTTP requests",
    ["method", "endpoint", "http_status"], registry=REGISTRY
)
HTTP_LATENCY = Histogram(
    "http_request_duration_seconds", "HTTP request latency",
    ["method", "endpoint"], registry=REGISTRY
)

# -------------------------------------------------------------------------------------
# App e middleware
# -------------------------------------------------------------------------------------
app = FastAPI(title="Tech Challenge F4 – AMZN LSTM API", version="1.2.2")

@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    start = time.perf_counter()
    try:
        response = await call_next(request)
        status = response.status_code
    except Exception:
        status = 500
        HTTP_REQUESTS.labels(request.method, request.url.path, str(status)).inc()
        HTTP_LATENCY.labels(request.method, request.url.path).observe(time.perf_counter() - start)
        raise
    else:
        HTTP_REQUESTS.labels(request.method, request.url.path, str(status)).inc()
        HTTP_LATENCY.labels(request.method, request.url.path).observe(time.perf_counter() - start)
        return response

# -------------------------------------------------------------------------------------
# Utilidades
# -------------------------------------------------------------------------------------
def _load_features_order_from_npz(window: int, horizon: int) -> List[str]:
    """Lê a ordem de features de um .npz (train -> val -> test)."""
    base = Path(settings.PROCESSED_DIR)
    suffix = f"{settings.TICKER.upper()}_w{window}_h{horizon}"
    for name in (f"train_{suffix}.npz", f"val_{suffix}.npz", f"test_{suffix}.npz"):
        p = base / name
        if p.exists():
            data = np.load(p, allow_pickle=True)
            return list(data["features"])  # type: ignore
    raise FileNotFoundError(
        "Não foi possível localizar um .npz para inferir a ordem das features. "
        "Gere os arquivos com src.features antes da inferência."
    )

def _inverse_close(pred_scaled: np.ndarray, scaler, close_idx: int) -> np.ndarray:
    """Converte (H,) da escala do scaler para a escala original do Close."""
    if hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):  # StandardScaler
        return pred_scaled * scaler.scale_[close_idx] + scaler.mean_[close_idx]
    if hasattr(scaler, "data_min_") and hasattr(scaler, "data_max_"):  # MinMaxScaler
        return pred_scaled * (scaler.data_max_[close_idx] - scaler.data_min_[close_idx]) + scaler.data_min_[close_idx]
    return pred_scaled

_MODEL_CACHE: dict[int, object] = {}
_SCALER = None

def _get_model(horizon: int):
    path = Path(settings.MODELS_DIR) / ("model_h1.h5" if horizon == 1 else "model_h5.h5")
    if not path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {path}")
    if horizon not in _MODEL_CACHE:
        logger.info("Carregando modelo {}", path)
        _MODEL_CACHE[horizon] = load_model(path, compile=False)
    return _MODEL_CACHE[horizon]

def _get_scaler():
    global _SCALER
    if _SCALER is None:
        p = Path(settings.SCALER_PATH)
        if not p.exists():
            raise FileNotFoundError(f"Scaler não encontrado: {p}")
        logger.info("Carregando scaler {}", p)
        _SCALER = joblib_load(p)
    return _SCALER

# -------------------------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    return {"status": "ok", "ticker": settings.TICKER, "window_default": settings.WINDOW}

@app.get("/metadata")
def metadata() -> JSONResponse:
    p = Path(settings.MODELS_DIR) / "metadata.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="metadata.json não encontrado")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Falha ao ler metadata: {e}")
    return JSONResponse(content=data)

@app.get("/metrics")
def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest(REGISTRY).decode("utf-8"), media_type=CONTENT_TYPE_LATEST)

@app.get("/features-order")
def features_order(horizon: int, window: int | None = None) -> dict:
    if horizon not in (1, 5):
        raise HTTPException(status_code=422, detail="horizon deve ser 1 ou 5")
    w = window or settings.WINDOW
    feats = _load_features_order_from_npz(w, horizon)
    return {"horizon": horizon, "window": w, "n_features": len(feats), "features": feats}

@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    if payload.horizon not in (1, 5):
        raise HTTPException(status_code=422, detail="horizon deve ser 1 ou 5")
    window = payload.window or settings.WINDOW

    try:
        expected_feats = _load_features_order_from_npz(window, payload.horizon)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if payload.recent_features is None:
        raise HTTPException(
            status_code=422,
            detail="Envie 'recent_features' [window, n_features] na MESMA ORDEM das features de treino."
        )

    X_in = np.asarray(payload.recent_features, dtype=float)
    if X_in.ndim != 2 or X_in.shape[0] != window:
        raise HTTPException(status_code=422, detail=f"recent_features deve ter shape [window={window}, n_features]")

    n_features = X_in.shape[1]
    if payload.features_order is not None:
        if list(payload.features_order) != list(expected_feats):
            raise HTTPException(
                status_code=422,
                detail="features_order não coincide com a ordem de treino. Consulte /features-order."
            )
    else:
        if n_features != len(expected_feats):
            raise HTTPException(
                status_code=422,
                detail=f"n_features={n_features} difere do treino ({len(expected_feats)}). Informe 'features_order' se necessário."
            )

    scaler = _get_scaler()
    try:
        X_scaled = scaler.transform(X_in)
    except Exception:
        X_scaled = np.vstack([scaler.transform(X_in[i:i+1, :]) for i in range(X_in.shape[0])])

    model = _get_model(payload.horizon)
    X_model = X_scaled[np.newaxis, :, :]
    y_pred_scaled: np.ndarray = model.predict(X_model, verbose=0).squeeze()

    try:
        close_idx = list(expected_feats).index("Close")
    except ValueError:
        raise HTTPException(status_code=500, detail="Feature 'Close' ausente na lista de treino.")

    y_pred = _inverse_close(y_pred_scaled, scaler, close_idx)
    y_pred = np.atleast_1d(y_pred).astype(float).tolist()

    return PredictResponse(
        horizon=payload.horizon,
        window=window,
        n_features=n_features,
        features=list(expected_feats),
        predictions=y_pred,
        model_path=str(Path(settings.MODELS_DIR) / ("model_h1.h5" if payload.horizon == 1 else "model_h5.h5")),
        scaler_path=str(settings.SCALER_PATH),
        metadata_path=str(Path(settings.MODELS_DIR) / "metadata.json") if (Path(settings.MODELS_DIR) / "metadata.json").exists() else None,
    )

@app.post("/predict-ticker", response_model=PredictResponse)
def predict_ticker(payload: PredictTickerRequest) -> PredictResponse:
    if payload.horizon not in (1, 5):
        raise HTTPException(status_code=422, detail="horizon deve ser 1 ou 5")

    window = payload.window or settings.WINDOW
    ticker = (payload.ticker or settings.TICKER).upper()

    try:
        from api.inference import prepare_window_for_model  # type: ignore
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=("Endpoint indisponível: crie 'api/inference.py'. Detalhe: " + str(e))
        )

    try:
        X_model, features, scaler, close_idx = prepare_window_for_model(
            ticker=ticker, window=window, horizon=payload.horizon,
            lookback_days=payload.lookback_days,
            scaler_path=str(settings.SCALER_PATH),
            processed_dir=str(settings.PROCESSED_DIR),
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Falha ao preparar janela: {e}")

    model = _get_model(payload.horizon)
    y_pred_scaled: np.ndarray = model.predict(X_model, verbose=0).squeeze()
    y_pred = _inverse_close(y_pred_scaled, scaler, close_idx)
    y_pred = np.atleast_1d(y_pred).astype(float).tolist()

    return PredictResponse(
        horizon=payload.horizon,
        window=window,
        n_features=len(features),
        features=list(features),
        predictions=y_pred,
        model_path=str(Path(settings.MODELS_DIR) / ("model_h1.h5" if payload.horizon == 1 else "model_h5.h5")),
        scaler_path=str(settings.SCALER_PATH),
        metadata_path=str(Path(settings.MODELS_DIR) / "metadata.json") if (Path(settings.MODELS_DIR) / "metadata.json").exists() else None,
    )

# -------------------------------------------------------------------------------------
# Execução local
# -------------------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    host = getattr(settings, "API_HOST", "0.0.0.0")
    port = int(getattr(settings, "API_PORT", 8000))
    logger.info("Subindo API em http://{h}:{p}", h=host, p=port)
    uvicorn.run("api.main:app", host=host, port=port, reload=False)