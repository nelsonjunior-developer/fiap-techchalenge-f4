"""
Avaliação dos modelos – Tech Challenge F4

- Carrega arrays .npz (train/val/test) gerados por src.features.
- Avalia o modelo treinado (H=1 ou H=5) em escala ORIGINAL de preços.
- Compara com baseline ingênuo (persistência).
- Gera gráficos e um relatório JSON com as métricas.
- (Novo) Gera gráficos de resíduos e um backtesting simples (walk-forward com modelo fixo).

Uso:
  python -m src.evaluate --horizon 1 --window 60
  python -m src.evaluate --horizon 5 --window 60
  # flags opcionais:
  python -m src.evaluate --horizon 5 --window 60 --no-residuals --no-walkforward
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")  # garante backend não interativo para salvar figuras
import matplotlib.pyplot as plt
import numpy as np
from joblib import load
from loguru import logger
from tensorflow.keras.models import load_model  # type: ignore

from src.utils.config import settings


@dataclass
class EvalResult:
    mae: float
    rmse: float
    mape: float


# ---------- Utilidades de dados ----------


def _load_npz(
    kind: str, ticker: str, window: int, horizon: int
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    suffix = f"{ticker.upper()}_w{window}_h{horizon}"
    path = Path(settings.PROCESSED_DIR) / f"{kind}_{suffix}.npz"
    data = np.load(path, allow_pickle=True)
    X, y, feats = data["X"], data["y"], list(data["features"])
    return X, y, feats


def _inverse_close(arr_scaled: np.ndarray, scaler, close_idx: int) -> np.ndarray:
    """Converte (n,H) do espaço escalado para a escala original do Close."""
    if hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):  # StandardScaler
        return arr_scaled * scaler.scale_[close_idx] + scaler.mean_[close_idx]
    if hasattr(scaler, "data_min_") and hasattr(scaler, "data_max_"):  # MinMaxScaler
        return (
            arr_scaled * (scaler.data_max_[close_idx] - scaler.data_min_[close_idx])
            + scaler.data_min_[close_idx]
        )
    return arr_scaled


def _evaluate_on_original_scale(
    y_true_s: np.ndarray, y_pred_s: np.ndarray, scaler, close_idx: int
) -> EvalResult:
    y_true = _inverse_close(y_true_s, scaler, close_idx)
    y_pred = _inverse_close(y_pred_s, scaler, close_idx)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    denom = np.clip(np.abs(y_true), 1e-8, None)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
    return EvalResult(mae=mae, rmse=rmse, mape=mape)


def _per_horizon_metrics(
    y_true_s: np.ndarray, y_pred_s: np.ndarray, scaler, close_idx: int
) -> List[Dict[str, float]]:
    """Métricas por passo futuro (t+1..t+H). Retorna lista de dicts."""
    y_true = _inverse_close(y_true_s, scaler, close_idx)
    y_pred = _inverse_close(y_pred_s, scaler, close_idx)
    H = y_true.shape[1]
    out = []
    for k in range(H):
        yt = y_true[:, k]
        yp = y_pred[:, k]
        mae = float(np.mean(np.abs(yt - yp)))
        rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
        denom = np.clip(np.abs(yt), 1e-8, None)
        mape = float(np.mean(np.abs((yt - yp) / denom)) * 100.0)
        out.append({"step": k + 1, "mae": mae, "rmse": rmse, "mape": mape})
    return out


def _naive_persistence_from_X(
    X: np.ndarray, close_idx: int, horizon: int
) -> np.ndarray:
    """Baseline ingênuo (persistência) no espaço escalado: repete último Close da janela."""
    last_close = X[:, -1, close_idx]  # (n,)
    return np.repeat(last_close.reshape(-1, 1), horizon, axis=1)  # (n,H)


# ---------- Resíduos ----------


def _compute_residuals_original(
    y_true_s: np.ndarray, y_pred_s: np.ndarray, scaler, close_idx: int
) -> np.ndarray:
    """Retorna matriz (n,H) de resíduos no domínio original: y_true - y_pred."""
    y_true = _inverse_close(y_true_s, scaler, close_idx)
    y_pred = _inverse_close(y_pred_s, scaler, close_idx)
    return y_true - y_pred


def _plot_residuals_time(resid: np.ndarray, out_path: Path, title: str):
    """Plota resíduos ao longo do tempo. Para H>1, usa o passo t+1."""
    r = resid[:, 0] if resid.shape[1] > 1 else resid[:, 0]
    plt.figure(figsize=(10, 4))
    plt.plot(r)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_residuals_hist(resid: np.ndarray, out_path: Path, title: str):
    r = resid[:, 0] if resid.shape[1] > 1 else resid[:, 0]
    plt.figure(figsize=(6, 4))
    plt.hist(r, bins=40)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ---------- Walk-forward simples (modelo fixo) ----------


def _walkforward_fixed_model(
    y_true_s: np.ndarray, y_pred_s: np.ndarray, scaler, close_idx: int
) -> Dict:
    """Backtesting simples com modelo fixo: erro por amostra ao longo do tempo (t+1).

    Observação: não há re-treino entre janelas; objetivo é inspecionar estabilidade temporal
    do erro do modelo já treinado.
    """
    y_true = _inverse_close(y_true_s, scaler, close_idx)
    y_pred = _inverse_close(y_pred_s, scaler, close_idx)
    # Série de erro absoluto para t+1
    err_abs_t1 = (
        np.abs(y_true[:, 0] - y_pred[:, 0])
        if y_true.shape[1] >= 1
        else np.abs(y_true.squeeze() - y_pred.squeeze())
    )
    summary = {
        "mae_t1_mean": float(np.mean(err_abs_t1)),
        "mae_t1_median": float(np.median(err_abs_t1)),
        "n": int(err_abs_t1.shape[0]),
    }
    return {"err_abs_t1": err_abs_t1.tolist(), "summary": summary}


def _plot_walkforward(err_abs_t1: np.ndarray, out_path: Path, title: str):
    plt.figure(figsize=(10, 4))
    plt.plot(err_abs_t1)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ---------- Execução principal ----------


def _plot_pred_vs_true(
    y_true_s: np.ndarray,
    y_pred_s: np.ndarray,
    scaler,
    close_idx: int,
    out_path: Path,
    title: str,
):
    """Gráfico simples (índice de amostra no eixo x)."""
    y_true = _inverse_close(y_true_s, scaler, close_idx)
    y_pred = _inverse_close(y_pred_s, scaler, close_idx)

    # Para H=1: série única; para H>1: plota apenas t+1 para visual rápido.
    if y_true.shape[1] == 1:
        yt = y_true[:, 0]
        yp = y_pred[:, 0]
        plt.figure(figsize=(10, 4))
        plt.plot(yt, label="True")
        plt.plot(yp, label="Pred")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
    else:
        yt = y_true[:, 0]
        yp = y_pred[:, 0]
        plt.figure(figsize=(10, 4))
        plt.plot(yt, label="True (t+1)")
        plt.plot(yp, label="Pred (t+1)")
        plt.title(title + " — passo t+1")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()


def evaluate(
    horizon: int, window: int, *, do_residuals: bool = True, do_walkforward: bool = True
) -> Dict:
    # Carrega dados
    X_te, y_te, feats = _load_npz("test", settings.TICKER, window, horizon)
    assert "Close" in feats, "Feature 'Close' ausente."
    close_idx = feats.index("Close")

    # Carrega modelo (.h5) com compile=False para evitar desserializar losses/metrics legadas
    model_path = Path(settings.MODELS_DIR) / (
        "model_h1.h5" if horizon == 1 else "model_h5.h5"
    )
    assert model_path.exists(), f"Modelo não encontrado: {model_path}"
    model = load_model(model_path, compile=False)

    # Predições
    y_pred_te = model.predict(X_te, verbose=0)

    # Scaler
    scaler = load(settings.SCALER_PATH)

    # Métricas agregadas
    res_model = _evaluate_on_original_scale(y_te, y_pred_te, scaler, close_idx)

    # Baseline ingênuo
    y_base_te = _naive_persistence_from_X(X_te, close_idx, horizon)
    res_base = _evaluate_on_original_scale(y_te, y_base_te, scaler, close_idx)

    # Métricas por horizonte (apenas se H>1)
    per_h = (
        _per_horizon_metrics(y_te, y_pred_te, scaler, close_idx) if horizon > 1 else []
    )

    # Diretório de plots
    plots_dir = Path(settings.MODELS_DIR) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Plot principal (pred vs true)
    out_plot = plots_dir / f"pred_vs_true_{settings.TICKER}_w{window}_h{horizon}.png"
    _plot_pred_vs_true(
        y_te, y_pred_te, scaler, close_idx, out_plot, f"Pred vs True – H={horizon}"
    )

    # Resíduos
    resid_report = None
    if do_residuals:
        resid = _compute_residuals_original(y_te, y_pred_te, scaler, close_idx)
        out_res_time = (
            plots_dir / f"residuals_time_{settings.TICKER}_w{window}_h{horizon}.png"
        )
        out_res_hist = (
            plots_dir / f"residuals_hist_{settings.TICKER}_w{window}_h{horizon}.png"
        )
        _plot_residuals_time(
            resid, out_res_time, f"Resíduos no tempo – H={horizon} (t+1)"
        )
        _plot_residuals_hist(
            resid, out_res_hist, f"Histograma de resíduos – H={horizon} (t+1)"
        )
        resid_report = {
            "residuals_time_plot": str(out_res_time),
            "residuals_hist_plot": str(out_res_hist),
            "residuals_stats_t1": {
                "mean": float(np.mean(resid[:, 0])),
                "std": float(np.std(resid[:, 0])),
                "p95_abs": float(np.percentile(np.abs(resid[:, 0]), 95)),
            },
        }

    # Walk-forward simples (modelo fixo)
    wf_report = None
    if do_walkforward:
        wf = _walkforward_fixed_model(y_te, y_pred_te, scaler, close_idx)
        err_abs_t1 = np.array(wf["err_abs_t1"])
        out_wf = (
            plots_dir / f"walkforward_mae_t1_{settings.TICKER}_w{window}_h{horizon}.png"
        )
        _plot_walkforward(
            err_abs_t1, out_wf, f"Walk-forward (MAE t+1 por amostra) – H={horizon}"
        )
        wf_report = {
            "walkforward_plot": str(out_wf),
            "summary": wf["summary"],
        }

    # Relatório
    report = {
        "ticker": settings.TICKER,
        "window": window,
        "horizon": horizon,
        "metrics_test_model": asdict(res_model),
        "metrics_test_naive": asdict(res_base),
        "per_horizon": per_h,
        "plot_pred_vs_true": str(out_plot),
        "model_path": str(model_path),
        "scaler_path": str(settings.SCALER_PATH),
        "residuals": resid_report,
        "walkforward": wf_report,
    }

    # Salva JSON
    out_json = (
        Path(settings.MODELS_DIR)
        / f"evaluation_report_{settings.TICKER}_w{window}_h{horizon}.json"
    )
    out_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Relatório salvo em: {p}", p=str(out_json))
    logger.info("Plot salvo em: {p}", p=str(out_plot))

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Avalia modelo LSTM para H=1 ou H=5.")
    parser.add_argument("--horizon", type=int, choices=[1, 5], required=True)
    parser.add_argument("--window", type=int, default=settings.WINDOW)
    parser.add_argument(
        "--no-residuals", action="store_true", help="não gerar gráficos de resíduos"
    )
    parser.add_argument(
        "--no-walkforward", action="store_true", help="não rodar backtesting simples"
    )
    args = parser.parse_args()

    _ = evaluate(
        args.horizon,
        args.window,
        do_residuals=not args.no_residuals,
        do_walkforward=not args.no_walkforward,
    )
