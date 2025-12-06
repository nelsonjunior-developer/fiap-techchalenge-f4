
"""
Configuração centralizada do projeto (Pydantic Settings / Pydantic v2).

Este módulo lê variáveis de ambiente (arquivo `.env` + ambiente do sistema)
fornecendo **defaults**, **validações** e **conveniências** para o restante do
código. Mantém o projeto reprodutível e evita hardcode de parâmetros.

Como funciona:
- Copie `.env.example` para `.env` e ajuste.
- `Settings` carrega automaticamente `.env` e variáveis exportadas no ambiente.
- Ordem de precedência típica: CLI/código > ENV do SO > `.env` > defaults.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field
from pydantic import computed_field
from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Schema tipado para configurações do projeto.

    Os nomes dos campos coincidem com as chaves do `.env.example` para facilitar
    o onboarding. Todos possuem defaults sensatos para desenvolvimento.
    """

    # ===== Core de dados/modelo =====
    TICKER: str = "AMZN"
    START_DATE: str = "2018-01-01"  # str para manter formato ISO simples
    FREQ: str = "1D"
    WINDOW: int = 60
    H_BASELINE: int = 1
    H: int = 5

    # ===== Splits e reprodutibilidade =====
    TRAIN_SPLIT: float = 0.70
    VAL_SPLIT: float = 0.15
    TEST_SPLIT: float = 0.15
    SEED: int = 42

    # ===== Hiperparâmetros de treino =====
    EPOCHS: int = 50
    BATCH_SIZE: int = 64
    LEARNING_RATE: float = 1e-3
    LSTM_UNITS: int = 64
    DROPOUT: float = 0.2
    SCALER: str = Field("StandardScaler", description="StandardScaler|MinMaxScaler")

    # ===== Paths =====
    DATA_DIR: Path = Path("./data")
    RAW_DIR: Path = Path("./data/raw")
    PROCESSED_DIR: Path = Path("./data/processed")
    MODELS_DIR: Path = Path("./models")
    SCALER_PATH: Path = Path("./models/scaler.joblib")
    MODEL_H1_PATH: Path = Path("./models/model_h1.h5")
    MODEL_H5_PATH: Path = Path("./models/model_h5.h5")
    METADATA_PATH: Path = Path("./models/metadata.json")

    # ===== API (FastAPI / Uvicorn) =====
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_LOG_LEVEL: str = "info"
    ALLOWED_CORS_ORIGINS: str = (
        "http://localhost:8501"  # aceita vírgulas ou ponto e vírgula
    )
    ENABLE_PROMETHEUS: bool = True
    METRICS_PATH: str = "/metrics"

    # ===== Frontend (Streamlit) =====
    API_BASE_URL: str = "http://127.0.0.1:8000"

    # ===== Timezone & misc =====
    LOCAL_TIMEZONE: str = "Europe/Amsterdam"
    YF_SLEEP_SECONDS: int = 0
    LOG_FORMAT: str = Field("json", description="json|text")

    # ===== Config interna do Pydantic Settings =====
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        validate_default=True,
    )

    # -----------------
    # Validations
    # -----------------
    @field_validator("WINDOW")
    @classmethod
    def _window_min(cls, v: int) -> int:
        if v < 1:
            raise ValueError("WINDOW deve ser >= 1 (recomendado >= 30).")
        return v

    @field_validator("DROPOUT")
    @classmethod
    def _dropout_range(cls, v: float) -> float:
        if not (0.0 <= v < 1.0):
            raise ValueError("DROPOUT deve estar em [0.0, 1.0).")
        return v

    @field_validator("SCALER")
    @classmethod
    def _scaler_supported(cls, v: str) -> str:
        v_norm = v.strip()
        if v_norm not in {"StandardScaler", "MinMaxScaler"}:
            raise ValueError("SCALER deve ser 'StandardScaler' ou 'MinMaxScaler'.")
        return v_norm

    @field_validator("H_BASELINE")
    @classmethod
    def _hbaseline_is_one(cls, v: int) -> int:
        if v != 1:
            raise ValueError("H_BASELINE deve ser 1 (baseline).")
        return v

    @field_validator("H")
    @classmethod
    def _h_supported(cls, v: int) -> int:
        if v not in {1, 5}:
            raise ValueError("H deve ser 1 ou 5.")
        return v

    @field_validator("METRICS_PATH")
    @classmethod
    def _metrics_path_format(cls, v: str) -> str:
        if not v.startswith("/"):
            raise ValueError("METRICS_PATH deve iniciar com '/'.")
        return v

    @model_validator(mode="after")
    def _splits_sum_to_one(self) -> "Settings":
        total = self.TRAIN_SPLIT + self.VAL_SPLIT + self.TEST_SPLIT
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Splits devem somar 1.0 (recebido: {total:.4f})."
            )
        return self

    # -----------------
    # Convenience properties
    # -----------------
    @computed_field(return_type=List[str])
    @property
    def CORS_ORIGINS_LIST(self) -> List[str]:  # noqa: N802 (const-like)
        """Lista de origens de CORS a partir de string separada por vírgulas/;.

        Ex.: "http://localhost:8501,https://meusite.com" -> ["http://localhost:8501", ...]
        """
        raw = (self.ALLOWED_CORS_ORIGINS or "").replace(";", ",")
        return [o.strip() for o in raw.split(",") if o.strip()]

    def make_scaler(self):
        """Retorna instância do scaler configurado (scikit-learn)."""
        try:
            from sklearn.preprocessing import MinMaxScaler, StandardScaler
        except Exception as exc:  # pragma: no cover - fallback de import
            raise RuntimeError(
                "scikit-learn é necessário para criar o scaler"
            ) from exc
        return StandardScaler() if self.SCALER == "StandardScaler" else MinMaxScaler()

    @computed_field(return_type=Path)
    @property
    def PROJECT_ROOT(self) -> Path:  # noqa: N802
        """Raiz do projeto (heurística baseada neste arquivo)."""
        # src/utils/config.py -> raiz = 3 níveis acima
        return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Retorna instância única (cacheada) de Settings.

    Uso:
        from src.utils.config import get_settings
        settings = get_settings()
    """
    return Settings()


# Acesso rápido (opcional): from src.utils.config import settings
settings = get_settings()
