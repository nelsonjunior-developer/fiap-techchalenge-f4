"""
Módulo de ingestão e preparo de dados (yfinance) – Tech Challenge F4.

Responsabilidades (simples e bem documentadas):
- Baixar OHLCV do Yahoo Finance via `yfinance`.
- Normalizar índice temporal (timezone -> datas diárias), limpar faltantes e duplicatas.
- Remover outliers de forma **simples** (winsorizar Close e Volume por IQR – opcional).
- Salvar CSV bruto em `data/raw/` e splits processados em `data/processed/`.
- Realizar split temporal (train/val/test) **sem vazamento**.

Observações:
- Este módulo **não** faz janelamento (fica em `features.py`).
- Valores padrão vêm de `src.utils.config.settings` e podem ser sobrescritos via `.env`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import time

import pandas as pd
import yfinance as yf
from loguru import logger

from src.utils.config import settings

# ==============================
# Utilidades de logging simples
# ==============================
if settings.LOG_FORMAT == "json":
    # Loguru já formata como texto; para JSON real, integrar com json-logger.
    # Mantemos formato compacto e consistente para o projeto.
    logger.remove()
    logger.add(lambda msg: print(msg, end=""))  # stdout


@dataclass
class SplitResult:
    """Container tipado para retornos de splits temporais."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


# ==============================
# Ingestão e limpeza
# ==============================


def _normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Garante colunas simples ['Open','High','Low','Close','Volume'].

    Em yfinance>=0.2.66 é comum vir **MultiIndex** (ex.: nível 0='Price', nível 1='AMZN'),
    ou nível 0 já conter 'Open/High/Low/Close/Volume'. Este helper achata nesses casos
    de forma determinística, para que *df['Close']* seja uma **Series** e não um DataFrame.
    """
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)
        expected = {"Open", "High", "Low", "Close", "Volume"}
        if expected.issubset(set(lvl0)):
            # Mantém apenas o nível-0: vira colunas simples Open/High/...
            df.columns = lvl0
        elif "Price" in set(lvl0):
            # Recorta o bloco 'Price' que contém Open/High/Low/Close/Volume
            try:
                df = df.xs("Price", axis=1, level=0)
            except Exception:
                # Fallback: usa o último nível
                df.columns = df.columns.get_level_values(-1)
        else:
            # Fallback genérico: último nível
            df.columns = df.columns.get_level_values(-1)

    # Normaliza capitalização (open->Open etc.)
    rename_map = {c: (str(c).replace("_", " ").title()) for c in df.columns}
    df = df.rename(columns=rename_map)
    return df


def _download_with_retries(
    ticker: str,
    start: str,
    end: Optional[str],
    interval: str,
    auto_adjust: bool,
    attempts: int = 3,
    sleep_seconds: float = 1.0,
) -> pd.DataFrame:
    """Tenta baixar dados com algumas tentativas e faz fallbacks progressivos.

    Estratégia:
      1) `yf.download` com retries e `repair=True` (se suportado pela versão instalada).
      2) `yf.Ticker(...).history(start/end)`
      3) `yf.Ticker(...).history(period='max')` + recorte por [start, end]
    """
    interval = (interval or "1d").lower()

    # 1) Tentativas com yf.download
    for i in range(1, attempts + 1):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=auto_adjust,
                progress=False,
                threads=False,
                group_by="column",
                repair=True,  # ignorado se não suportado
                timeout=30,
            )
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
            logger.warning(f"Tentativa {i}/{attempts}: download vazio para {ticker}.")
        except Exception as e:
            logger.warning(f"Tentativa {i}/{attempts} falhou no yf.download: {e}")
        time.sleep(max(settings.YF_SLEEP_SECONDS, sleep_seconds) * i)

    # 2) Fallback: Ticker.history com start/end
    try:
        t = yf.Ticker(ticker)
        df = t.history(
            start=start,
            end=end,
            interval=interval,
            auto_adjust=auto_adjust,
            actions=False,
            repair=True,
        )
        if isinstance(df, pd.DataFrame) and not df.empty:
            cols = [
                c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns
            ]
            if cols:
                return df[cols].copy()
    except Exception as e:
        logger.error(f"Fallback history(start/end) falhou: {e}")

    # 3) Fallback: Ticker.history(period='max') e recorte de datas
    try:
        t = yf.Ticker(ticker)
        df = t.history(
            period="max",
            interval=interval,
            auto_adjust=auto_adjust,
            actions=False,
            repair=True,
        )
        if isinstance(df, pd.DataFrame) and not df.empty:
            df.index = pd.to_datetime(df.index)
            if start:
                df = df[df.index >= pd.to_datetime(start)]
            if end:
                df = df[df.index <= pd.to_datetime(end)]
            cols = [
                c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns
            ]
            if cols and not df.empty:
                return df[cols].copy()
            logger.error(
                "Fallback period='max' retornou vazio após recorte ou sem colunas OHLCV."
            )
    except Exception as e:
        logger.error(f"Fallback history(period='max') falhou: {e}")

    return pd.DataFrame()


def fetch_ohlcv_yf(
    ticker: str,
    start: str,
    end: Optional[str] = None,
    interval: str = "1d",
    auto_adjust: bool = False,
) -> pd.DataFrame:
    """Baixa OHLCV do Yahoo Finance.

    Args:
        ticker: Código do ativo (ex.: "AMZN").
        start: Data inicial (YYYY-MM-DD).
        end: Data final (YYYY-MM-DD) – se None, usa hoje.
        interval: Intervalo (usaremos 1d por padrão).
        auto_adjust: Se True, ajusta por proventos/splits (mantemos False p/ manter OHLCV bruto).

    Returns:
        DataFrame com colunas [Open, High, Low, Close, Volume] e índice de datas normalizado.
    """
    interval = (interval or "1d").lower()  # yfinance espera lowercase
    if interval not in {"1d", "1wk", "1mo"}:
        logger.warning(
            f"Intervalo '{interval}' não suportado para fallback robusto; forçando '1d'."
        )
        interval = "1d"
    logger.info(
        f"Baixando dados: ticker={ticker}, start={start}, end={end}, interval={interval}"
    )

    df = _download_with_retries(
        ticker=ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
        attempts=3,
        sleep_seconds=1.0,
    )

    if not isinstance(df, pd.DataFrame) or df.empty:
        raise RuntimeError("yfinance retornou DataFrame vazio.")

    # NOVO: achata/normaliza colunas antes de qualquer seleção
    df = _normalize_ohlcv_columns(df)

    # Garantir colunas esperadas e normalização do índice
    expected = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes no dataset: {missing}")

    df = df[expected].copy()
    df.index = pd.to_datetime(df.index)

    # Se vier com timezone, normalizamos para datas (sem tz, dia civil)
    try:
        if getattr(df.index, "tz", None) is not None:
            # Converte para timezone local configurado e remove tz
            df.index = df.index.tz_convert(settings.LOCAL_TIMEZONE).tz_localize(None)
    except Exception:
        # Se não der para converter, apenas remova tz
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)

    # Normaliza para dias (00:00), remove duplicatas e ordena
    df.index = df.index.normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()

    # Faltantes: forward-fill simples, depois drop de remanescentes
    df = df.ffill().dropna()

    return df


def winsorize_iqr(
    df: pd.DataFrame,
    columns: Tuple[str, str] = ("Close", "Volume"),
    whisker: float = 1.5,
) -> pd.DataFrame:
    """Aplica winsorização por IQR em colunas numéricas (limpeza simples de outliers).

    Estratégia: capping nos limites [Q1-1.5*IQR, Q3+1.5*IQR].
    Mantemos a série contínua (não removemos linhas) para não quebrar janelas.
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lo = q1 - whisker * iqr
        hi = q3 + whisker * iqr
        # aqui df[col] será **Series** (graças ao _normalize_ohlcv_columns)
        df[col] = df[col].clip(lower=lo, upper=hi)
    return df


# ==============================
# Persistência de arquivos
# ==============================


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_raw_csv(df: pd.DataFrame, out_dir: Path, ticker: str) -> Path:
    """Salva CSV bruto em `out_dir` com nome padronizado.

    Ex.: data/raw/AMZN_2018-01-01_2025-10-30.csv
    """
    _ensure_dir(out_dir)
    start = df.index.min().strftime("%Y-%m-%d")
    end = df.index.max().strftime("%Y-%m-%d")
    out_path = out_dir / f"{ticker.upper()}_{start}_{end}.csv"
    df.to_csv(out_path, index_label="Date")
    logger.info(f"CSV bruto salvo em: {out_path}")
    return out_path


def temporal_split(
    df: pd.DataFrame,
    train_split: float = settings.TRAIN_SPLIT,
    val_split: float = settings.VAL_SPLIT,
    test_split: float = settings.TEST_SPLIT,
) -> SplitResult:
    """Realiza split temporal (70/15/15 por padrão) sem embaralhar.

    Garantimos `train+val+test=1` no `config.py`; aqui só aplicamos os cortes.
    """
    if not abs((train_split + val_split + test_split) - 1.0) < 1e-6:
        raise ValueError("Os splits devem somar 1.0.")

    n = len(df)
    n_train = int(n * train_split)
    n_val = int(n * val_split)
    # resto vai para teste
    train = df.iloc[:n_train]
    val = df.iloc[n_train : n_train + n_val]
    test = df.iloc[n_train + n_val :]

    # Sanidade: índices em ordem e não vazios
    if any(len(x) == 0 for x in (train, val, test)):
        logger.warning(
            f"Split gerou partições vazias: n={n}, train={len(train)}, val={len(val)}, test={len(test)}"
        )

    return SplitResult(train=train, val=val, test=test)


def save_processed_splits(
    splits: SplitResult, out_dir: Path, ticker: str
) -> Tuple[Path, Path, Path]:
    """Salva splits processados em CSVs nomeados por partição.

    Retorna as paths (train, val, test).
    """
    _ensure_dir(out_dir)

    def _name(part: str, df: pd.DataFrame) -> Path:
        start = df.index.min().strftime("%Y-%m-%d") if not df.empty else "NA"
        end = df.index.max().strftime("%Y-%m-%d") if not df.empty else "NA"
        return out_dir / f"{ticker.upper()}_{part}_{start}_{end}.csv"

    p_train = _name("train", splits.train)
    p_val = _name("val", splits.val)
    p_test = _name("test", splits.test)

    splits.train.to_csv(p_train, index_label="Date")
    splits.val.to_csv(p_val, index_label="Date")
    splits.test.to_csv(p_test, index_label="Date")

    logger.info(f"Splits salvos em: {p_train}, {p_val}, {p_test}")
    return p_train, p_val, p_test


# ==============================
# Orquestração: download + limpeza + split + salvamento
# ==============================


def download_prepare_and_save(
    ticker: str = settings.TICKER,
    start: str = settings.START_DATE,
    end: Optional[str] = None,
    *,
    winsorize: bool = True,
) -> Tuple[pd.DataFrame, SplitResult]:
    """Pipeline compacto usado por scripts/CLI.

    1) Baixa OHLCV
    2) Limpa faltantes/duplicatas e normaliza index
    3) (Opcional) winsoriza Close/Volume
    4) Salva CSV bruto
    5) Split temporal e salvamento dos splits
    """
    df = fetch_ohlcv_yf(
        ticker=ticker,
        start=start,
        end=end,
        interval=settings.FREQ.lower(),  # robusto a FREQ="1D" no .env
    )
    if winsorize:
        df = winsorize_iqr(df, columns=("Close", "Volume"))

    # Persistência
    _ = save_raw_csv(df, Path(settings.RAW_DIR), ticker)

    # Splits
    splits = temporal_split(df)
    save_processed_splits(splits, Path(settings.PROCESSED_DIR), ticker)

    logger.info("Ingestão concluída.")
    return df, splits


# ==============================
# Modo CLI (útil para uso rápido e CI)
# ==============================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Baixa e prepara OHLCV via yfinance (salva raw e splits)."
    )
    parser.add_argument("--ticker", default=settings.TICKER, help="Ticker (ex.: AMZN)")
    parser.add_argument(
        "--start", default=settings.START_DATE, help="Data inicial YYYY-MM-DD"
    )
    parser.add_argument("--end", default=None, help="Data final YYYY-MM-DD (opcional)")
    parser.add_argument(
        "--no-winsorize",
        action="store_true",
        help="Desabilita winsorização IQR de Close/Volume",
    )

    args = parser.parse_args()

    _ = download_prepare_and_save(
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        winsorize=not args.no_winsorize,
    )
