"""
Engenharia de atributos e janelamento para séries temporais – Tech Challenge F4

Este módulo cuida de:
- Construir *features* a partir de OHLCV (retornos, volatilidade, RSI, MACD, etc.).
- Normalizar/escala (fit **apenas** no treino; aplicar em val/test).
- Transformar a série em supervisado via janelas deslizantes (X[t-window:t] → y[t+1..t+H]).
- Persistir artefatos: scaler (`models/scaler.joblib`) e arrays em `data/processed/`.

Design:
- Mantemos funções puras e reutilizáveis; a CLI apenas orquestra.
- Evitamos vazamento temporal: scaler é ajustado **só** no treino.
- Compatível com H=1 (baseline) e H=5 (multi-saída) – controlado por parâmetro.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from loguru import logger

from src.utils.config import settings

# ==============================
# Conveniências / Tipos
# ==============================

FEATURE_COLUMNS_DEFAULT = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    # derivados
    "ret1",
    "logret1",
    "vol20",
    "rsi14",
    "macd",
    "macd_signal",
    "macd_hist",
]


@dataclass
class WindowedArrays:
    """Container para arrays janelados."""

    X: np.ndarray  # (n_samples, window, n_features)
    y: np.ndarray  # (n_samples, H) – quando H==1, mantém 2D (n,1)
    feature_names: List[str]


# ==============================
# Indicadores técnicos (implementações leves)
# ==============================


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona indicadores técnicos e estatísticos básicos ao DataFrame.

    Entradas: df com colunas pelo menos [Open, High, Low, Close, Volume].
    Saída: df com colunas extras (retornos, vol, RSI, MACD...).
    """
    d = df.copy()

    # Retornos (simples e log) – 1 dia
    d["ret1"] = d["Close"].pct_change(1)
    d["logret1"] = np.log(d["Close"]).diff(1)

    # Volatilidade – desvio-padrão móvel de 20 dias dos retornos simples
    d["vol20"] = d["ret1"].rolling(20, min_periods=5).std()

    # RSI(14)
    delta = d["Close"].diff()
    up = delta.clip(lower=0).rolling(14, min_periods=5).mean()
    down = (-delta.clip(upper=0)).rolling(14, min_periods=5).mean()
    rs = up / (down.replace(0, np.nan))
    d["rsi14"] = 100 - (100 / (1 + rs))

    # MACD(12,26,9)
    ema12 = _ema(d["Close"], 12)
    ema26 = _ema(d["Close"], 26)
    macd = ema12 - ema26
    signal = _ema(macd, 9)
    d["macd"] = macd
    d["macd_signal"] = signal
    d["macd_hist"] = macd - signal

    # Remove primeiras linhas com NaN por conta dos indicadores
    d = d.dropna()
    return d


# ==============================
# Escalonamento (fit no treino apenas)
# ==============================


def scale_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Ajusta scaler no treino e aplica em val/test. Salva o scaler em disco.

    Retorna (train_scaled, val_scaled, test_scaled).
    """
    scaler = settings.make_scaler()

    # Ajuste no treino – usando apenas as colunas de *features*
    scaler.fit(train_df[feature_cols].values)

    # Aplicação
    train_scaled = train_df.copy()
    val_scaled = val_df.copy()
    test_scaled = test_df.copy()

    train_scaled[feature_cols] = scaler.transform(train_df[feature_cols].values)
    val_scaled[feature_cols] = scaler.transform(val_df[feature_cols].values)
    test_scaled[feature_cols] = scaler.transform(test_df[feature_cols].values)

    # Persistência
    Path(settings.MODELS_DIR).mkdir(parents=True, exist_ok=True)
    dump(scaler, settings.SCALER_PATH)
    logger.info(f"Scaler salvo em: {settings.SCALER_PATH}")

    return train_scaled, val_scaled, test_scaled


# ==============================
# Janelamento (supervisionado)
# ==============================


def _make_future_targets(y: pd.Series, horizon: int) -> pd.DataFrame:
    """Cria DataFrame com colunas y_{t+1}..y_{t+H} a partir de uma série y (ex.: Close)."""
    cols = {}
    for k in range(1, horizon + 1):
        cols[f"t+{k}"] = y.shift(-k)
    out = pd.concat(cols, axis=1)
    return out


def make_windows(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    window: int,
    horizon: int,
) -> WindowedArrays:
    """Transforma DataFrame em arrays supervisionados via janelamento.

    - X: pilha de janelas (n amostras, window, n_features)
    - y: futuros (n amostras, H) – quando H==1, y tem shape (n,1)
    """
    if window < 1:
        raise ValueError("window deve ser >= 1")
    if horizon not in (1, 5):
        logger.warning("Horizon não usual; suportado 1 ou 5. Continuando mesmo assim.")

    df = df.copy()
    # Alinha targets futuros
    y_future = _make_future_targets(df[target_col], horizon)
    df_all = pd.concat([df[feature_cols], y_future], axis=1).dropna()

    feats = df_all[feature_cols].values
    ymat = df_all[[c for c in y_future.columns]].values

    n = len(df_all)
    n_feats = len(feature_cols)
    samples = n - window + 1
    if samples <= 0:
        logger.warning(
            f"Janelamento resultou em zero amostras (n={n}, window={window}). Retornando arrays vazios."
        )
        return WindowedArrays(X=np.empty((0, window, n_feats)), y=np.empty((0, horizon)), feature_names=list(feature_cols))

    X = np.zeros((samples, window, n_feats), dtype=float)
    for i in range(samples):
        X[i] = feats[i : i + window, :]

    # Alinha y ao fim de cada janela: a linha referente ao índice (i+window-1)
    y = ymat[window - 1 : window - 1 + samples]

    # Garante 2D para H=1
    if horizon == 1 and y.ndim == 1:
        y = y.reshape(-1, 1)

    return WindowedArrays(X=X, y=y, feature_names=list(feature_cols))


# ==============================
# Pipeline de alto nível
# ==============================


def build_features_and_windows(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    *,
    feature_cols: Sequence[str] | None = None,
    window: int | None = None,
    horizon: int | None = None,
    target_col: str = "Close",
) -> Tuple[WindowedArrays, WindowedArrays, WindowedArrays]:
    """Pipeline completo: indicadores → escala → janelas.

    Retorna tupla com (train_arrays, val_arrays, test_arrays).
    """
    window = window or settings.WINDOW
    horizon = horizon or settings.H

    # 1) Indicadores
    train_f = compute_indicators(train)
    val_f = compute_indicators(val)
    test_f = compute_indicators(test)

    # 2) Colunas de features
    if feature_cols is None:
        feature_cols = FEATURE_COLUMNS_DEFAULT

    # 3) Escalonamento (fit no treino)
    train_s, val_s, test_s = scale_datasets(train_f, val_f, test_f, feature_cols)

    # 4) Janelamento
    tr_arr = make_windows(train_s, feature_cols, target_col, window, horizon)
    va_arr = make_windows(val_s, feature_cols, target_col, window, horizon)
    te_arr = make_windows(test_s, feature_cols, target_col, window, horizon)

    return tr_arr, va_arr, te_arr


# ==============================
# CLI – converte CSVs processados em arrays .npz
# ==============================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gera features + janelas a partir de CSVs processados.")
    parser.add_argument("--ticker", default=settings.TICKER, help="Ticker (ex.: AMZN)")
    parser.add_argument("--window", type=int, default=settings.WINDOW, help="Tamanho da janela (ex.: 60)")
    parser.add_argument("--horizon", type=int, default=settings.H, help="Horizonte (1 ou 5)")
    parser.add_argument(
        "--outdir",
        default=str(settings.PROCESSED_DIR),
        help="Diretório de saída para .npz (default: data/processed)",
    )

    args = parser.parse_args()

    # Localiza CSVs criados por src/data.py
    base = Path(settings.PROCESSED_DIR)
    train_path = sorted(base.glob(f"{args.ticker.upper()}_train_*.csv"))
    val_path = sorted(base.glob(f"{args.ticker.upper()}_val_*.csv"))
    test_path = sorted(base.glob(f"{args.ticker.upper()}_test_*.csv"))
    if not (train_path and val_path and test_path):
        raise SystemExit("Não foram encontrados CSVs processados. Rode primeiro: python -m src.data ...")
    train_csv, val_csv, test_csv = train_path[-1], val_path[-1], test_path[-1]

    # Carrega
    def _load(p: Path) -> pd.DataFrame:
        df = pd.read_csv(p, parse_dates=["Date"]).set_index("Date").sort_index()
        return df

    train_df = _load(train_csv)
    val_df = _load(val_csv)
    test_df = _load(test_csv)

    # Constrói pipeline completo
    tr, va, te = build_features_and_windows(
        train_df,
        val_df,
        test_df,
        window=args.window,
        horizon=args.horizon,
        feature_cols=FEATURE_COLUMNS_DEFAULT,
    )

    # Persistência dos arrays
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.ticker.upper()}_w{args.window}_h{args.horizon}"

    np.savez_compressed(outdir / f"train_{suffix}.npz", X=tr.X, y=tr.y, features=np.array(tr.feature_names, dtype=object))
    np.savez_compressed(outdir / f"val_{suffix}.npz", X=va.X, y=va.y, features=np.array(va.feature_names, dtype=object))
    np.savez_compressed(outdir / f"test_{suffix}.npz", X=te.X, y=te.y, features=np.array(te.feature_names, dtype=object))

    logger.info(
        "Arrays salvos em {outdir} com sufixo {suffix}",
        outdir=str(outdir),
        suffix=suffix,
    )
