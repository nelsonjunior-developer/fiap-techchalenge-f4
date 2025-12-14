"""
Streamlit frontend para o Tech Challenge Fase 4.

Objetivo: prover uma UI simples para consumir a API FastAPI do projeto e
visualizar previsões H=1 ou H=5 para o ticker AMZN (ou outro, se suportado).

Notas importantes:
- Este app **não** treina modelos; ele consome a API (/health, /metadata, /predict).
- Requer `streamlit` instalado no ambiente (não está no runtime mínimo da API).
  Para usar: `pip install streamlit` (ou adicione ao requirements se desejar incluir no runtime).
- Por padrão, a URL da API é lida de `API_BASE_URL` ou `http://127.0.0.1:8000`.

Execução local:
    streamlit run app.py
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
import yfinance as yf

def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    """Log simples para stdout (aparece no terminal do Streamlit)."""
    print(f"[app.py] {_now()} {msg}")


def resolve_api_base_url() -> str:
    """Resolve a URL base da API (env > secrets > fallback local)."""
    env_url = os.getenv("API_BASE_URL")
    secret_url: Optional[str] = None
    try:
        secret_url = st.secrets.get("API_BASE_URL")  # type: ignore[attr-defined]
    except Exception:
        secret_url = None

    url = env_url or secret_url or "http://127.0.0.1:8000"
    url = url.rstrip("/")
    if not url:
        url = "http://127.0.0.1:8000"
    return url


# ============================
# Configuração básica do app
# ============================
API_BASE_URL = resolve_api_base_url()
log(f"Using API_BASE_URL={API_BASE_URL}")
st.set_page_config(page_title="Tech Challenge F4 – LSTM Forecast", layout="wide")

# Mostrar detalhes de erros no cliente Streamlit (útil para debug)
st.set_option("client.showErrorDetails", True)


# ============================
# Helpers de requisição HTTP
# ============================
def _request(
    method: str,
    url: str,
    *,
    json_payload: Optional[Dict[str, Any]] = None,
    timeout: int = 15,
) -> Tuple[Optional[Dict[str, Any]], float, Optional[str]]:
    """Envolve `requests` e retorna (json, latência_em_segundos, erro_str).

    Mantemos a assinatura simples para instrumentação/erros na UI.
    """
    start = datetime.now()
    try:
        resp = requests.request(method, url, json=json_payload, timeout=timeout)
        latency = (datetime.now() - start).total_seconds()
        resp.raise_for_status()
        # Tenta JSON; se falhar, devolve texto bruto
        try:
            log(f"HTTP {method} {url} -> {resp.status_code} in {latency:.3f}s")
            return resp.json(), latency, None
        except Exception:
            return {"raw": resp.text}, latency, None
    except Exception as exc:  # noqa: BLE001 – exibimos erro detalhado na UI
        latency = (datetime.now() - start).total_seconds()
        log(f"HTTP {method} {url} FAILED in {latency:.3f}s: {exc}")
        return None, latency, str(exc)


def api_health(api_url: str) -> Tuple[Optional[Dict[str, Any]], float, Optional[str]]:
    url = f"{api_url.rstrip('/')}/health"
    log(f"Calling {url}")
    return _request("GET", url)


def api_metadata(api_url: str) -> Tuple[Optional[Dict[str, Any]], float, Optional[str]]:
    url = f"{api_url.rstrip('/')}/metadata"
    log(f"Calling {url}")
    return _request("GET", url)


def api_predict(
    api_url: str, payload: Dict[str, Any]
) -> Tuple[Optional[Dict[str, Any]], float, Optional[str]]:
    url = f"{api_url.rstrip('/')}/predict"
    log(f"Calling {url} with payload keys={list(payload.keys())}")
    return _request("POST", url, json_payload=payload)


def api_features_order(
    api_url: str, horizon: int, window: int
) -> Tuple[Optional[Dict[str, Any]], float, Optional[str]]:
    url = f"{api_url.rstrip('/')}/features-order?horizon={horizon}&window={window}"
    log(f"Calling {url}")
    return _request("GET", url)


def api_predict_ticker(
    api_url: str, payload: Dict[str, Any]
) -> Tuple[Optional[Dict[str, Any]], float, Optional[str]]:
    url = f"{api_url.rstrip('/')}/predict-ticker"
    log(f"Calling {url} with payload keys={list(payload.keys())}")
    return _request("POST", url, json_payload=payload)


# ============================
# Dados auxiliares via yfinance (opcional)
# ============================
@st.cache_data(show_spinner=False)
def fetch_history_yf(ticker: str, days_back: int = 400) -> pd.DataFrame:
    """Busca OHLCV recente pelo yfinance para visualização e/ou envio ao backend.

    days_back: janela de histórico para exibir/usar (aprox.).
    Retorna DataFrame com colunas padrão do Yahoo (Open, High, Low, Close, Volume).
    """
    try:
        end = datetime.now()
        start = end - timedelta(days=days_back)
        log(f"yfinance.download({ticker}, {start.date()} -> {end.date()}, interval=1d)")
        df = yf.download(
            ticker,
            start=start.date().isoformat(),
            end=end.date().isoformat(),
            interval="1d",
            auto_adjust=False,
        )
        if not isinstance(df, pd.DataFrame) or df.empty:
            log("yfinance returned empty DataFrame")
            return pd.DataFrame()
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        log(f"yfinance dataframe shape={df.shape}")
        return df
    except Exception as exc:
        log(f"yfinance failed: {exc}")
        return pd.DataFrame()


def build_payload_from_df(
    df: pd.DataFrame, window: int, horizon: int, ticker: Optional[str] = None
) -> Dict[str, Any]:
    """Prepara um payload de previsão com base em um DataFrame OHLCV.

    Formato pensado para alinhar com o `schemas.py` do backend:
        {
          "horizon": 5,
          "window": 60,
          "ticker": "AMZN",            # opcional
          "history": [                  # últimos `window` registros
            {"date": "YYYY-MM-DD", "open": float, "high": float, "low": float,
             "close": float, "volume": float},
            ...
          ]
        }
    """
    tail = df.tail(window)
    records = [
        {
            "date": idx.strftime("%Y-%m-%d"),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": float(row["Volume"]),
        }
        for idx, row in tail.iterrows()
    ]
    payload: Dict[str, Any] = {
        "horizon": int(horizon),
        "window": int(window),
        "history": records,
    }
    if ticker:
        payload["ticker"] = ticker
    return payload


# ============================
# UI
# ============================


def sidebar_ui() -> Dict[str, Any]:
    st.sidebar.header("Configurações")
    api_url = st.sidebar.text_input(
        "API Base URL", value=API_BASE_URL, help="Ex.: http://127.0.0.1:8000"
    )

    st.sidebar.markdown("---")
    input_mode = st.sidebar.radio(
        "Entrada de dados",
        options=(
            "Ticker (API busca)",
            "Ticker (app busca via yfinance)",
            "Vetor de Features (JSON)",
        ),
        index=0,
        help=(
            "Formas de obter o histórico recente: \n"
            "• API busca: o backend coleta os dados do ticker. \n"
            "• App busca: este app usa yfinance e envia parâmetros equivalentes. \n"
            "• Vetor JSON: você fornece `recent_features` (lista de listas)."
        ),
    )

    ticker = st.sidebar.text_input("Ticker", value="AMZN")
    horizon = st.sidebar.select_slider(
        "Horizon (passos à frente)", options=[1, 5], value=5
    )
    window = st.sidebar.slider(
        "Window (tamanho da janela)", min_value=30, max_value=180, value=60, step=5
    )
    lookback = st.sidebar.slider(
        "Lookback (dias)",
        min_value=60,
        max_value=365,
        value=180,
        step=30,
        help="Usado pelos modos de ticker (/predict-ticker).",
    )

    st.sidebar.markdown("---")
    health_btn = st.sidebar.button("Testar /health")
    meta_btn = st.sidebar.button("Ver /metadata")

    log(
        f"sidebar: api_url={api_url} mode={input_mode} ticker={ticker} H={horizon} W={window} lookback={lookback}"
    )

    return {
        "api_url": api_url,
        "input_mode": input_mode,
        "ticker": ticker,
        "horizon": horizon,
        "window": window,
        "lookback": lookback,
        "health_btn": health_btn,
        "meta_btn": meta_btn,
    }


def show_health_and_metadata(api_url: str, do_health: bool, do_meta: bool) -> None:
    cols = st.columns(2)
    if do_health:
        with cols[0]:
            st.subheader("/health")
            data, lat, err = api_health(api_url)
            if err:
                st.error(f"Falha no /health ({lat:.3f}s): {err}")
            else:
                st.success(f"OK ({lat:.3f}s)")
                st.json(data)
    if do_meta:
        with cols[1]:
            st.subheader("/metadata")
            data, lat, err = api_metadata(api_url)
            if err:
                st.error(f"Falha no /metadata ({lat:.3f}s): {err}")
            else:
                st.info(f"Carregado em {lat:.3f}s")
                st.json(data)


def main() -> None:
    st.title("📈 Tech Challenge F4 – LSTM Forecast UI")
    st.caption(
        "Frontend simples em Streamlit para consumir a API (FastAPI) deste projeto. "
        "Use como apoio didático para explorar o comportamento do modelo."
    )

    # Checa /health da API base antes de renderizar a UI completa
    base_url = API_BASE_URL
    health, lat, err = api_health(base_url)
    if err:
        st.error(
            f"API indisponível ({err}). Configure API_BASE_URL nos Secrets do Streamlit ou nas variáveis de ambiente."
        )
    else:
        st.info(f"API OK ({lat:.3f}s) em {base_url}")

    cfg = sidebar_ui()
    log("main: loaded sidebar config")
    api_url = cfg["api_url"]

    # Bloco opcional: ações rápidas /health e /metadata
    show_health_and_metadata(api_url, cfg["health_btn"], cfg["meta_btn"])

    st.markdown("---")
    st.header("Previsão")

    # Seção de entrada e preparação de payload
    payload: Optional[Dict[str, Any]] = None
    history_df: Optional[pd.DataFrame] = None

    if cfg["input_mode"] == "Ticker (API busca)":
        st.write(
            "O backend fará a coleta do histórico. Informe somente os parâmetros de janela e horizonte."
        )
        payload = {
            "ticker": cfg["ticker"],
            "window": int(cfg["window"]),
            "horizon": int(cfg["horizon"]),
            "lookback_days": int(cfg["lookback"]),
        }
        log(f"prepared payload for /predict-ticker: {json.dumps(payload)[:200]}...")

    elif cfg["input_mode"] == "Ticker (app busca via yfinance)":
        st.write(
            "Este app coletará o histórico via yfinance e enviará os últimos `window` pontos para a API."
        )
        with st.spinner("Baixando dados..."):
            history_df = fetch_history_yf(
                cfg["ticker"], days_back=max(cfg["window"] * 3, int(cfg["lookback"]))
            )
        if history_df is None or history_df.empty:
            st.warning(
                "Não foi possível obter dados via yfinance. Tente novamente ou altere o ticker."
            )
        else:
            st.success(f"Histórico carregado: {len(history_df)} linhas")
            st.line_chart(history_df["Close"], height=220)
            log(f"history_df loaded via yfinance: rows={len(history_df)}")
            # Observação: para manter compatibilidade com o backend, enviaremos para /predict-ticker,
            # que calcula as features no servidor. O histórico baixado aqui é apenas para visualização.
            payload = {
                "ticker": cfg["ticker"],
                "window": int(cfg["window"]),
                "horizon": int(cfg["horizon"]),
                "lookback_days": int(cfg["lookback"]),
            }

    else:  # Vetor de Features (JSON)
        st.write(
            "Forneça um JSON com `recent_features` no formato lista de listas (shape `window × n_features`). "
            "A ordem exata das features pode ser consultada em `/features-order`."
        )
        meta, _, _ = api_features_order(api_url, cfg["horizon"], cfg["window"])
        if meta:
            log(f"/features-order returned n_features={meta.get('n_features')}")
            st.caption(
                f"Ordem de treino (n={meta.get('n_features')}): {meta.get('features')}"
            )
        txt = st.text_area("Cole aqui o JSON de `recent_features` (lista de listas).")
        if txt.strip():
            try:
                rf = json.loads(txt)
                if isinstance(rf, list):
                    payload = {
                        "horizon": int(cfg["horizon"]),
                        "window": int(cfg["window"]),
                        "recent_features": rf,
                    }
                    log(
                        f"prepared payload for /predict (JSON mode) with window={cfg['window']}"
                    )
                else:
                    st.error(
                        "JSON inválido: esperado uma lista de listas (recent_features)."
                    )
            except Exception as exc:
                st.error(f"Falha ao interpretar JSON: {exc}")

    # Botão de previsão
    predict_col, payload_col = st.columns([1, 1])
    with payload_col:
        st.subheader("Payload que será enviado")
        if payload is not None:
            st.code(
                json.dumps(payload, indent=2)[:2000], language="json"
            )  # limita tamanho na UI
        else:
            st.info("Aguardando dados para montar o payload…")

    with predict_col:
        st.subheader("Executar previsão")
        btn_label = (
            "/predict-ticker" if cfg["input_mode"].startswith("Ticker") else "/predict"
        )
        run = st.button(btn_label, type="primary", use_container_width=True)
        if run:
            if payload is None:
                st.warning(f"Necessário montar o payload antes de chamar {btn_label}.")
            else:
                use_ticker = cfg["input_mode"].startswith("Ticker")
                with st.spinner(f"Chamando API {btn_label}…"):
                    if use_ticker:
                        data, lat, err = api_predict_ticker(api_url, payload)
                    else:
                        data, lat, err = api_predict(api_url, payload)
                log(f"response from {btn_label}: error={bool(err)} latency={lat:.3f}s")
                if err:
                    st.error(f"Falha no {btn_label} ({lat:.3f}s): {err}")
                elif not data:
                    st.warning(f"Resposta vazia do backend ({lat:.3f}s)")
                else:
                    st.success(f"Previsão recebida em {lat:.3f}s")
                    st.json(data)

                    # Exibição amigável: tentamos detectar um formato comum
                    # Esperado (sugestão de schemas no backend):
                    # {
                    #   "predictions": [float, float, ...],
                    #   "horizon": 5,
                    #   "last_date": "YYYY-MM-DD"  # opcional
                    # }
                    preds = data.get("predictions") if isinstance(data, dict) else None
                    if isinstance(preds, list) and preds:
                        last_date_str = data.get("last_date")
                        if (
                            last_date_str is None
                            and history_df is not None
                            and not history_df.empty
                        ):
                            last_date_str = history_df.index.max().strftime("%Y-%m-%d")
                        # Cria índice de datas futuras (útil para visualização)
                        try:
                            base_date = (
                                pd.to_datetime(last_date_str)
                                if last_date_str
                                else pd.Timestamp.today()
                            )
                        except Exception:
                            base_date = pd.Timestamp.today()
                        future_idx = pd.date_range(
                            base_date + pd.Timedelta(days=1),
                            periods=len(preds),
                            freq="D",
                        )
                        df_pred = pd.DataFrame(
                            {"PredictedClose": preds}, index=future_idx
                        )

                        st.subheader("Tabela de Previsões")
                        st.dataframe(df_pred, use_container_width=True)

                        if history_df is not None and not history_df.empty:
                            st.subheader("Histórico (Close) + Previsões")
                            # Concatenamos para um chart único
                            plot_df = pd.concat(
                                [
                                    history_df[["Close"]]
                                    .rename(columns={"Close": "Close"})
                                    .tail(200),
                                    df_pred.rename(columns={"PredictedClose": "Close"}),
                                ]
                            )
                            st.line_chart(plot_df["Close"], height=300)


if __name__ == "__main__":
    # Permite executar como script padrão (sem streamlit) para checar disponibilidade da API
    # Ex.: python app.py
    api_url = API_BASE_URL
    health, lat, err = api_health(api_url)
    if err:
        log(f"/health erro ({lat:.3f}s): {err}")
    else:
        log(f"/health ok ({lat:.3f}s): {health}")

    # Renderiza a UI quando executado via `streamlit run app.py`
    main()
