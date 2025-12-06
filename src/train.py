

"""
Treino dos modelos LSTM (H=1 e H=5) – Tech Challenge F4

Fluxo:
1) Carrega arrays .npz gerados por `src.features` (train/val/test).
2) Constrói o modelo LSTM conforme `horizon` (1 ou 5).
3) Treina com callbacks (ES + ReduceLROnPlateau).
4) Avalia em teste na escala original (desfaz o scaling do Close).
5) Baseline ingênuo (persistência: repete o último Close da janela).
6) Salva modelo (models/model_h{H}.h5) e registra métricas em models/metadata.json.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import tensorflow as tf
from joblib import load
from loguru import logger

from src.model import build_lstm_model, default_callbacks
from src.utils.config import settings

# -----------------------------
# Reprodutibilidade
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


@dataclass
class EvalResult:
    mae: float
    rmse: float
    mape: float


# -----------------------------
# Utilitários de dados
# -----------------------------

def _load_npz(kind: str, ticker: str, window: int, horizon: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Carrega X,y,features de um .npz em data/processed."""
    suffix = f"{ticker.upper()}_w{window}_h{horizon}"
    path = Path(settings.PROCESSED_DIR) / f"{kind}_{suffix}.npz"
    data = np.load(path, allow_pickle=True)
    X, y, feats = data["X"], data["y"], list(data["features"])
    return X, y, feats


def _inverse_close(arr_scaled: np.ndarray, scaler, close_idx: int) -> np.ndarray:
    """Converte (n,H) do espaço escalado para a escala original do Close."""
    # StandardScaler
    if hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
        return arr_scaled * scaler.scale_[close_idx] + scaler.mean_[close_idx]
    # MinMaxScaler
    if hasattr(scaler, "data_min_") and hasattr(scaler, "data_max_"):
        return arr_scaled * (scaler.data_max_[close_idx] - scaler.data_min_[close_idx]) + scaler.data_min_[close_idx]
    return arr_scaled  # fallback


def _evaluate_on_original_scale(
    y_true_s: np.ndarray, y_pred_s: np.ndarray, scaler, close_idx: int
) -> EvalResult:
    """Calcula MAE/RMSE/MAPE na escala original do preço."""
    y_true = _inverse_close(y_true_s, scaler, close_idx)
    y_pred = _inverse_close(y_pred_s, scaler, close_idx)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    denom = np.clip(np.abs(y_true), 1e-8, None)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
    return EvalResult(mae=mae, rmse=rmse, mape=mape)


def _naive_persistence_from_X(X: np.ndarray, close_idx: int, horizon: int) -> np.ndarray:
    """Baseline ingênuo (persistência): futuros (t+1..t+H) = último Close observado (escala do scaler)."""
    last_close = X[:, -1, close_idx]  # (n,)
    return np.repeat(last_close.reshape(-1, 1), horizon, axis=1)  # (n,H)


# -----------------------------
# Treino
# -----------------------------

def train_once(
    horizon: int,
    window: int,
    *,
    units: int = 64,
    dropout: float = 0.2,
    lr: float = 1e-3,
    epochs: int = 50,
    batch_size: int = 128,
) -> Dict:
    """Treina e avalia um modelo para o `horizon` informado. Retorna dict para metadata.json."""
    # Carrega dados e valida consistência das features
    X_tr, y_tr, f_tr = _load_npz("train", settings.TICKER, window, horizon)
    X_va, y_va, f_va = _load_npz("val", settings.TICKER, window, horizon)
    X_te, y_te, f_te = _load_npz("test", settings.TICKER, window, horizon)

    assert f_tr == f_va == f_te, "Lista/ordem de features divergente entre splits"
    feature_names = f_tr
    assert "Close" in feature_names, "Feature 'Close' ausente; necessária para métricas e baseline."
    close_idx = feature_names.index("Close")

    # Modelo
    n_features = X_tr.shape[-1]
    model = build_lstm_model(window, n_features, horizon, units=units, dropout=dropout, lr=lr)

    history = model.fit(
        X_tr,
        y_tr,
        validation_data=(X_va, y_va),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=list(default_callbacks()),
        verbose=1,
    )

    # Predições (teste)
    y_pred_te = model.predict(X_te, verbose=0)

    # Inversão do Close
    scaler = load(settings.SCALER_PATH)

    # Métricas no domínio original
    res_model = _evaluate_on_original_scale(y_te, y_pred_te, scaler, close_idx)

    # Baseline ingênuo (persistência a partir de X_test) — sem vazamento
    y_base_te = _naive_persistence_from_X(X_te, close_idx, horizon)
    res_base = _evaluate_on_original_scale(y_te, y_base_te, scaler, close_idx)

    # Persistência do modelo
    out_path = Path(settings.MODELS_DIR) / ("model_h1.h5" if horizon == 1 else "model_h5.h5")
    Path(settings.MODELS_DIR).mkdir(parents=True, exist_ok=True)
    model.save(out_path)
    logger.info(
        "Modelo salvo em {path} | H={H} | MAE={mae:.4f} RMSE={rmse:.4f} MAPE={mape:.2f}%",
        path=str(out_path),
        H=horizon,
        mae=res_model.mae,
        rmse=res_model.rmse,
        mape=res_model.mape,
    )

    return {
        "horizon": horizon,
        "window": window,
        "units": units,
        "dropout": dropout,
        "lr": lr,
        "epochs_trained": len(history.history.get("loss", [])),
        "metrics_test": {"model": asdict(res_model), "naive": asdict(res_base)},
        "n_features": int(n_features),
    }


def _write_metadata(metadata: Dict):
    """Atualiza (ou cria) models/metadata.json mesclando com conteúdo existente."""
    meta_path = Path(settings.MODELS_DIR) / "metadata.json"
    try:
        current = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    except Exception:
        current = {}
    current.update(metadata)
    meta_path.write_text(json.dumps(current, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("metadata.json atualizado: {p}", p=str(meta_path))


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Treina LSTM para H=1 e/ou H=5 usando arrays .npz gerados por src.features."
    )
    parser.add_argument("--horizon", choices=["1", "5", "both"], default="both")
    parser.add_argument("--window", type=int, default=settings.WINDOW)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--units", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)

    args = parser.parse_args()

    horizons = [1, 5] if args.horizon == "both" else [int(args.horizon)]

    meta = {"runs": []}
    for H in horizons:
        run = train_once(
            H,
            args.window,
            units=args.units,
            dropout=args.dropout,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        meta["runs"].append(run)

    _write_metadata(meta)