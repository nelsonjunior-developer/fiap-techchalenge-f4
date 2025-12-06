"""
Pydantic schemas (API) — Tech Challenge F4

Modelos de request/response usados pela API FastAPI.

Observações:
- Mantemos exemplos em `json_schema_extra` para facilitar a visualização no Swagger.
- `PredictRequest` (modo features prontas) espera matriz [window, n_features] na MESMA ORDEM do treino.
- `PredictTickerRequest` abstrai a engenharia de features no servidor.
"""
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field, conlist


class PredictRequest(BaseModel):
    """Modo: features já processadas ([window, n_features]).

    Envie `recent_features` na MESMA ORDEM usada no treino; opcionalmente
    informe `features_order` para validação extra.
    """

    horizon: int = Field(..., description="H=1 ou H=5")
    window: Optional[int] = Field(None, description="Janela (default do settings)")

    # Matriz [window, n_features] na mesma ordem das features do treino
    recent_features: Optional[List[conlist(float, min_length=1)]] = Field(
        None, description="Matriz [window, n_features]"
    )
    features_order: Optional[List[str]] = Field(
        None, description="Ordem de features usada no treino"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "horizon": 5,
                "window": 60,
                "features_order": [
                    "Open", "High", "Low", "Close", "Volume",
                    "ret1", "logret1", "vol20", "rsi14", "macd",
                    "macd_signal", "macd_hist"
                ],
                "recent_features": [[0.0] * 12] * 60
            }
        }


class PredictResponse(BaseModel):
    horizon: int
    window: int
    n_features: int
    features: List[str]
    predictions: List[float]
    model_path: str
    scaler_path: str
    metadata_path: Optional[str]


class PredictTickerRequest(BaseModel):
    """Modo: por ticker (API calcula as features no servidor).

    A API baixa OHLCV recentes, calcula as 12 features do treino, aplica o scaler
    e faz a previsão para H=1 ou H=5.
    """

    horizon: int = Field(..., description="H=1 ou H=5")
    window: Optional[int] = Field(None, description="Janela (default do settings)")
    ticker: Optional[str] = Field(None, description="Ticker (default settings.TICKER)")
    lookback_days: int = Field(180, ge=60, description="Histórico para estabilizar indicadores")

    class Config:
        json_schema_extra = {
            "example": {
                "horizon": 5,
                "window": 60,
                "ticker": "AMZN",
                "lookback_days": 180
            }
        }
