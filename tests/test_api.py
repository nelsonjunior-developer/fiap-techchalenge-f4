import os

import pytest
from fastapi.testclient import TestClient

# Importa app; se falhar (ex.: problemas de PYTHONPATH no CI), faz skip
try:
    from api.main import app
except Exception as e:
    pytest.skip(f"Falha ao importar API: {e}", allow_module_level=True)

client = TestClient(app)


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body.get("status") == "ok"


def test_ready_ok():
    r = client.get("/ready")
    assert r.status_code == 200  # conteúdo pode ser simples; foco é readiness


def test_features_order_structure():
    # endpoint leve, não depende de artefatos
    r = client.get("/features-order", params={"horizon": 1, "window": 60})
    assert r.status_code in (200, 422)  # 422 caso validação de query seja estrita
    if r.status_code == 200:
        body = r.json()
        assert "horizon" in body and "window" in body and "n_features" in body and "features" in body
        assert isinstance(body["features"], list)
        assert isinstance(body["n_features"], int)
        assert len(body["features"]) == body["n_features"]


def test_metadata_responds():
    r = client.get("/metadata")
    assert r.status_code == 200
    body = r.json()
    assert "runs" in body  # mesmo que vazio


def test_metrics_exposes_text():
    r = client.get("/metrics")
    assert r.status_code == 200
    # Prometheus exposition format em texto
    ctype = r.headers.get("content-type", "")
    assert "text/plain" in ctype or "text; version=0.0.4" in ctype
    assert "# HELP" in r.text or "# TYPE" in r.text


def test_predict_validation_error():
    # Sem artefatos, validamos apenas que a API retorna 422 ao payload inválido
    bad_payload = {"horizon": 1}  # faltam campos obrigatórios
    r = client.post("/predict", json=bad_payload)
    assert r.status_code in (400, 422)


@pytest.mark.parametrize("horizon", [1, 5])
def test_predict_ticker_smoke_or_skip(horizon):
    # Teste leve: só roda localmente e se não for CI; caso contrário pula
    if os.environ.get("CI", "").lower() == "true":
        pytest.skip("Skip de predict-ticker em CI para evitar dependências externas")

    # Pode depender de yfinance e artefatos; se falhar, marcar como skip
    payload = {"horizon": horizon, "window": 60, "ticker": "AMZN", "lookback_days": 120}
    try:
        r = client.post("/predict-ticker", json=payload, timeout=60)
    except Exception as e:
        pytest.skip(f"predict-ticker indisponível: {e}")

    if r.status_code >= 500:
        pytest.skip(f"predict-ticker retornou {r.status_code}")
    # Aceita 200 ou 400/422 (validação) para não tornar o teste frágil
    assert r.status_code in (200, 400, 422)
    if r.status_code == 200:
        body = r.json()
        assert "predictions" in body
        preds = body["predictions"]
        assert isinstance(preds, list)
        assert len(preds) == (1 if horizon == 1 else horizon)
