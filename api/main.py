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

import json
import os
import sys
import time
from pathlib import Path
from typing import List
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, RedirectResponse
from joblib import load as joblib_load
from loguru import logger
import tensorflow as tf

from api.monitoring import metrics_endpoint, prometheus_middleware
from api.schemas import PredictRequest, PredictResponse, PredictTickerRequest
from src.utils.config import settings

tags_metadata = [
    {"name": "health", "description": "Liveness (/health) e readiness (/ready)."},
    {"name": "metadata", "description": "Artefatos e hiperparâmetros do treino."},
    {
        "name": "features",
        "description": "Ordem oficial de features para validação de payloads.",
    },
    {
        "name": "predict",
        "description": "Inferência com features pré-processadas ou por ticker.",
    },
    {"name": "monitoring", "description": "Métricas Prometheus para observabilidade."},
]
app = FastAPI(
    title="Tech Challenge F4 – AMZN LSTM API",
    version="1.2.3",
    openapi_tags=tags_metadata,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ajuste em produção
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.middleware("http")(prometheus_middleware)

# -------------------------------------------------------------------------------------
# Logging de acesso estruturado (JSON) – Requisito #6
# -------------------------------------------------------------------------------------
# Configura o sink do Loguru para emitir JSON no stdout quando LOG_JSON=true (default).
LOG_JSON = os.getenv("LOG_JSON", "true").lower() in ("1", "true", "yes")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
try:
    logger.remove()
except Exception:
    pass
logger.add(
    sys.stdout, level=LOG_LEVEL, serialize=LOG_JSON, backtrace=False, diagnose=False
)


@app.middleware("http")
async def access_log(request: Request, call_next):
    """Emite um log por requisição com campos estruturados.

    Campos: request_id, method, path, status, latency_ms
    Obs.: não logamos payloads para evitar PII acidental.
    """
    rid = str(uuid4())
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.bind(
        request_id=rid,
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        latency_ms=elapsed_ms,
    ).info("request")
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
        return (
            pred_scaled * (scaler.data_max_[close_idx] - scaler.data_min_[close_idx])
            + scaler.data_min_[close_idx]
        )
    return pred_scaled


_MODEL_CACHE: dict[int, object] = {}
_SCALER = None

def _get_model(horizon: int):
    path = Path(settings.MODELS_DIR) / (
        "model_h1.h5" if horizon == 1 else "model_h5.h5"
    )
    if not path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {path}")
    if horizon not in _MODEL_CACHE:
        tf_version = getattr(tf, "__version__", "unknown")
        try:
            import keras as keras_pkg  # type: ignore

            keras_version = getattr(keras_pkg, "__version__", "unknown")
        except Exception:
            keras_version = "unknown"
        logger.info(
            "Carregando modelo %s | tf=%s keras=%s",
            path,
            tf_version,
            keras_version,
        )
        try:
            _MODEL_CACHE[horizon] = tf.keras.models.load_model(path, compile=False)
        except Exception as e:
            logger.exception("Falha ao carregar modelo %s", path)
            raise HTTPException(
                status_code=500,
                detail=f"Falha ao carregar modelo {path.name}: {str(e)[:200]}",
            )
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
@app.get("/health", tags=["health"])
def health() -> dict:
    return {
        "status": "ok",
        "ticker": settings.TICKER,
        "window_default": settings.WINDOW,
    }


@app.get("/ready", tags=["health"])
def ready():
    paths = [
        Path(settings.MODELS_DIR) / "model_h1.h5",
        Path(settings.MODELS_DIR) / "model_h5.h5",
        Path(settings.SCALER_PATH),
    ]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise HTTPException(
            status_code=503, detail={"ready": False, "missing": missing}
        )
    # Tenta carregar modelos e scaler para garantir readiness real
    try:
        _ = _get_model(1)
        _ = _get_model(5)
        _ = _get_scaler()
    except Exception as e:
        logger.exception("Readiness falhou ao carregar artefatos")
        raise HTTPException(
            status_code=503,
            detail={"ready": False, "error": str(e)},
        )
    return {"ready": True}


@app.get("/metadata", tags=["metadata"])
def metadata() -> JSONResponse:
    p = Path(settings.MODELS_DIR) / "metadata.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="metadata.json não encontrado")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Falha ao ler metadata: {e}")
    return JSONResponse(content=data)


@app.get("/metrics", tags=["monitoring"])
def metrics() -> PlainTextResponse:
    return metrics_endpoint()


@app.get("/features-order", tags=["features"])
def features_order(horizon: int, window: int | None = None) -> dict:
    if horizon not in (1, 5):
        raise HTTPException(status_code=422, detail="horizon deve ser 1 ou 5")
    w = window or settings.WINDOW
    feats = _load_features_order_from_npz(w, horizon)
    return {
        "horizon": horizon,
        "window": w,
        "n_features": len(feats),
        "features": feats,
    }


@app.post("/predict", response_model=PredictResponse, tags=["predict"])
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
            detail="Envie 'recent_features' [window, n_features] na MESMA ORDEM das features de treino.",
        )

    X_in = np.asarray(payload.recent_features, dtype=float)
    if X_in.ndim != 2 or X_in.shape[0] != window:
        raise HTTPException(
            status_code=422,
            detail=f"recent_features deve ter shape [window={window}, n_features]",
        )

    n_features = X_in.shape[1]
    if payload.features_order is not None:
        if list(payload.features_order) != list(expected_feats):
            raise HTTPException(
                status_code=422,
                detail="features_order não coincide com a ordem de treino. Consulte /features-order.",
            )
    else:
        if n_features != len(expected_feats):
            raise HTTPException(
                status_code=422,
                detail=f"n_features={n_features} difere do treino ({len(expected_feats)}). Informe 'features_order' se necessário.",
            )

    scaler = _get_scaler()
    try:
        X_scaled = scaler.transform(X_in)
    except Exception:
        X_scaled = np.vstack(
            [scaler.transform(X_in[i : i + 1, :]) for i in range(X_in.shape[0])]
        )

    model = _get_model(payload.horizon)
    X_model = X_scaled[np.newaxis, :, :]
    y_pred_scaled: np.ndarray = model.predict(X_model, verbose=0).squeeze()

    try:
        close_idx = list(expected_feats).index("Close")
    except ValueError:
        raise HTTPException(
            status_code=500, detail="Feature 'Close' ausente na lista de treino."
        )

    y_pred = _inverse_close(y_pred_scaled, scaler, close_idx)
    y_pred = np.atleast_1d(y_pred).astype(float).tolist()

    return PredictResponse(
        horizon=payload.horizon,
        window=window,
        n_features=n_features,
        features=list(expected_feats),
        predictions=y_pred,
        model_path=str(
            Path(settings.MODELS_DIR)
            / ("model_h1.h5" if payload.horizon == 1 else "model_h5.h5")
        ),
        scaler_path=str(settings.SCALER_PATH),
        metadata_path=str(Path(settings.MODELS_DIR) / "metadata.json")
        if (Path(settings.MODELS_DIR) / "metadata.json").exists()
        else None,
    )


@app.post("/predict-ticker", response_model=PredictResponse, tags=["predict"])
def predict_ticker(payload: PredictTickerRequest) -> PredictResponse:
    if payload.horizon not in (1, 5):
        raise HTTPException(status_code=422, detail="horizon deve ser 1 ou 5")

    window = payload.window or settings.WINDOW
    ticker = (payload.ticker or settings.TICKER).upper().strip()

    try:
        from api.inference import prepare_window_for_model  # type: ignore
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=(
                "Endpoint indisponível: crie 'api/inference.py'. Detalhe: " + str(e)
            ),
        )

    try:
        X_model, features, scaler, close_idx = prepare_window_for_model(
            ticker=ticker,
            window=window,
            horizon=payload.horizon,
            lookback_days=payload.lookback_days,
            scaler_path=str(settings.SCALER_PATH),
            processed_dir=str(settings.PROCESSED_DIR),
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Falha ao obter dados ou preparar janela para {ticker}: {e}",
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(
            "prepare_window_for_model falhou | ticker=%s horizon=%s window=%s lookback=%s",
            ticker,
            payload.horizon,
            window,
            payload.lookback_days,
        )
        raise HTTPException(
            status_code=502,
            detail={
                "error": "predict_ticker_failed",
                "message": str(e)[:300],
                "type": type(e).__name__,
                "hint": "veja logs do Render para stacktrace completo",
            },
        )

    try:
        model = _get_model(payload.horizon)
        y_pred_scaled: np.ndarray = model.predict(X_model, verbose=0).squeeze()
        y_pred = _inverse_close(y_pred_scaled, scaler, close_idx)
        y_pred = np.atleast_1d(y_pred).astype(float).tolist()
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(
            "Falha ao inferir modelo | ticker=%s horizon=%s window=%s lookback=%s",
            ticker,
            payload.horizon,
            window,
            payload.lookback_days,
        )
        raise HTTPException(
            status_code=502,
            detail={
                "error": "model_inference_failed",
                "message": str(e)[:300],
                "type": type(e).__name__,
                "hint": "veja logs do Render para stacktrace completo",
            },
        )

    return PredictResponse(
        horizon=payload.horizon,
        window=window,
        n_features=len(features),
        features=list(features),
        predictions=y_pred,
        model_path=str(
            Path(settings.MODELS_DIR)
            / ("model_h1.h5" if payload.horizon == 1 else "model_h5.h5")
        ),
        scaler_path=str(settings.SCALER_PATH),
        metadata_path=str(Path(settings.MODELS_DIR) / "metadata.json")
        if (Path(settings.MODELS_DIR) / "metadata.json").exists()
        else None,
    )


@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/debug/yfinance", tags=["debug"])
def debug_yfinance(ticker: str, lookback_days: int = 180) -> dict:
    """Endpoint de diagnóstico para verificar download via yfinance (não carrega modelo)."""
    from api.inference import _download_ohlcv  # type: ignore

    df = _download_ohlcv(ticker, lookback_days)
    if df.empty:
        raise HTTPException(status_code=502, detail="yfinance retornou vazio")

    return {
        "ticker": ticker.upper().strip(),
        "rows": int(df.shape[0]),
        "cols": list(df.columns),
        "min_date": df.index.min().isoformat() if not df.empty else None,
        "max_date": df.index.max().isoformat() if not df.empty else None,
        "head_close": df["Close"].head(3).tolist() if "Close" in df else [],
    }


# -------------------------------------------------------------------------------------
# Execução local
# -------------------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    host = getattr(settings, "API_HOST", "0.0.0.0")
    port = int(getattr(settings, "API_PORT", 8000))
    logger.info("Subindo API em http://{h}:{p}", h=host, p=port)
    uvicorn.run("api.main:app", host=host, port=port, reload=False)
