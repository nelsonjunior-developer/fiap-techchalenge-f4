"""
api/inference.py — pipeline de inferência por ticker (AMZN por padrão)

Responsável por:
- baixar OHLCV recentes via yfinance;
- construir as MESMAS 12 features usadas no treino:
  ["Open","High","Low","Close","Volume",
   "ret1","logret1","vol20","rsi14","macd","macd_signal","macd_hist"]
- alinhar a ORDEM das colunas com a ordem oficial dos .npz (train/val/test);
- aplicar o scaler salvo (models/scaler.joblib);
- montar o tensor (1, window, n_features) para o modelo.

Obs.: Mantemos a engenharia de features aqui (lado servidor) para o endpoint
`POST /predict-ticker`, evitando divergências de ordem/escala no cliente.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from joblib import load as joblib_load
from loguru import logger


# =============================
# Indicadores técnicos auxiliares
# =============================

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI (Wilder). Implementação estável e sucinta.

    - Usa suavização exponencial (Wilder) com alpha = 1/period
    - Retorna 50.0 quando não há dados suficientes (evita NaN residuais)
    """
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD clássico (EMAs 12/26, sinal 9)."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


# ==================================
# Leitura da ordem oficial de features
# ==================================

def _load_features_order_from_npz(processed_dir: str, ticker: str, window: int, horizon: int) -> List[str]:
    """Carrega a ordem oficial das features a partir de `data/processed/*_TICKER_w{w}_h{h}.npz`.

    Estratégia: tenta train -> val -> test. Lança erros informativos se não encontrar.
    """
    base = Path(processed_dir)
    suffix = f"{ticker.upper()}_w{window}_h{horizon}"
    for kind in ("train", "val", "test"):
        p = base / f"{kind}_{suffix}.npz"
        if p.exists():
            data = np.load(p, allow_pickle=True)
            feats = list(data["features"])  # type: ignore
            if not feats:
                raise ValueError(f"Arquivo {p} não contém 'features'.")
            return feats
    raise FileNotFoundError(
        "Não foi possível localizar .npz para inferir a ordem das features. "
        f"Gere os arquivos com: python -m src.features --ticker {ticker} --window {window} --horizon {horizon}"
    )


# ===============================
# Ingestão yfinance + engenharia
# ===============================

def _download_ohlcv(ticker: str, lookback_days: int) -> pd.DataFrame:
    """Baixa OHLCV diário recente com yfinance (auto_adjust=True).

    Retorna DataFrame com colunas [Open, High, Low, Close, Volume] ordenado por data.
    """
    try:
        tkr = yf.Ticker(ticker)
        df = tkr.history(period=f"{lookback_days}d", interval="1d", auto_adjust=True)
    except Exception as e:
        raise RuntimeError(f"Falha no yfinance para {ticker}: {e}")

    if df is None or df.empty:
        raise RuntimeError(f"yfinance retornou vazio para {ticker} (period={lookback_days}d).")

    cols = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Colunas ausentes do yfinance: {missing}")

    out = df[cols].copy().dropna().sort_index()
    return out


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Constrói as 12 features usadas no treino.

    Base OHLCV +:
    - ret1      : retorno simples 1d (Close_t / Close_{t-1} - 1)
    - logret1   : log-return 1d (ln(Close_t) - ln(Close_{t-1}))
    - vol20     : desvio-padrão móvel de 20d do logret1
    - rsi14     : RSI(14)
    - macd trio : macd, macd_signal, macd_hist

    Mantém NaNs iniciais (de janelas) para posterior recorte.
    """
    out = df.copy()

    out["ret1"] = out["Close"].pct_change(1)
    out["logret1"] = np.log(out["Close"]).diff(1)
    out["vol20"] = out["logret1"].rolling(20).std()
    out["rsi14"] = _rsi(out["Close"], period=14)

    macd, macd_signal, macd_hist = _macd(out["Close"], fast=12, slow=26, signal=9)
    out["macd"] = macd
    out["macd_signal"] = macd_signal
    out["macd_hist"] = macd_hist

    return out


def _align_and_cut_window(df_feat: pd.DataFrame, features_order: List[str], window: int) -> pd.DataFrame:
    """Reordena colunas para `features_order`, remove NaNs e recorta as últimas `window` linhas."""
    df_feat = df_feat.copy()
    df_feat = df_feat[features_order]
    df_feat = df_feat.dropna()
    if len(df_feat) < window:
        raise ValueError(
            "Histórico útil insuficiente após estabilização de indicadores: "
            f"precisa de {window}, tem {len(df_feat)}."
        )
    return df_feat.iloc[-window:]


# ======================================
# Função principal usada pelo endpoint
# ======================================

def prepare_window_for_model(
    ticker: str,
    window: int,
    horizon: int,
    lookback_days: int,
    scaler_path: str,
    processed_dir: str,
) -> Tuple[np.ndarray, List[str], object, int]:
    """Prepara a janela (1, window, n_features) alinhada com o treino e pronta para inferência.

    Retorna:
      - X_model  : np.ndarray shape (1, window, n_features) **já escalado**
      - feats    : lista de nomes das features na ordem do treino
      - scaler   : objeto scaler carregado
      - close_idx: índice da coluna 'Close' dentro de `feats` (para inversão do scaling)

    Levanta erros informativos se faltar histórico, .npz/ordem de features ou scaler.
    """
    feats = _load_features_order_from_npz(processed_dir, ticker, window, horizon)

    # 1) baixa OHLCV bruto
    df = _download_ohlcv(ticker, lookback_days)

    # 2) constrói features derivadas
    df_feat = _build_features(df)

    # 3) alinha a ordem e recorta a última janela estável
    df_win = _align_and_cut_window(df_feat, feats, window)

    # 4) carrega scaler e transforma
    scaler = joblib_load(Path(scaler_path))
    X_in = df_win.values.astype(float)  # (window, n_features)

    try:
        X_scaled = scaler.transform(X_in)
    except Exception:
        # Alguns scalers esperam (n_samples, n_features). Aplicamos por linha.
        X_scaled = np.vstack([scaler.transform(X_in[i : i + 1, :]) for i in range(X_in.shape[0])])

    X_model = X_scaled[np.newaxis, :, :]  # (1, window, n_features)

    try:
        close_idx = feats.index("Close")
    except ValueError:
        raise RuntimeError("Feature 'Close' não consta na lista de treino.")

    logger.debug(
        "Janela pronta: shape=%s | n_features=%d | close_idx=%d",
        X_model.shape,
        len(feats),
        close_idx,
    )
    return X_model, feats, scaler, close_idx
