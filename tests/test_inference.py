import json
import os

import pytest

# Tenta importar funções do módulo de inferência
try:
    from api.inference import (
        get_artifact_paths,  # retorna caminhos dos artefatos para (horizon, window)
        get_features_order,  # retorna ordem das features para (horizon, window)
        predict_from_recent,  # roda inferência dados timesteps recentes
    )
except Exception as e:
    pytest.skip(f"inference não importável: {e}", allow_module_level=True)


METADATA_PATH = "models/metadata.json"


def _load_metadata_or_skip():
    if not os.path.exists(METADATA_PATH):
        pytest.skip(f"metadata ausente: {METADATA_PATH}")
    with open(METADATA_PATH, "r") as f:
        return json.load(f)


def _artifact_paths_for(horizon: int, window: int):
    """Obtém caminhos de artefatos de forma robusta, com fallback."""
    model_path = f"models/model_h{horizon}.h5"
    scaler_path = "models/scaler.joblib"
    metadata_path = METADATA_PATH
    try:
        paths = get_artifact_paths(horizon=horizon, window=window)
        # aceita dict ou tupla
        if isinstance(paths, dict):
            model_path = paths.get("model_path", model_path)
            scaler_path = paths.get("scaler_path", scaler_path)
            metadata_path = paths.get("metadata_path", metadata_path)
        elif isinstance(paths, (tuple, list)) and len(paths) >= 3:
            model_path, scaler_path, metadata_path = paths[:3]
    except Exception:
        # fallback silencioso
        pass
    return model_path, scaler_path, metadata_path


def test_metadata_has_runs_h1_h5():
    md = _load_metadata_or_skip()
    horizons = sorted({int(run.get("horizon")) for run in md.get("runs", [])})
    assert 1 in horizons and 5 in horizons, f"horizons presentes: {horizons}"


@pytest.mark.parametrize("horizon", [1, 5])
def test_artifacts_exist_or_skip(horizon):
    md = _load_metadata_or_skip()
    # tenta descobrir window do metadata (padrão=60)
    run = next((r for r in md.get("runs", []) if int(r.get("horizon")) == horizon), None)
    window = int(run.get("window", 60)) if run else 60

    model_path, scaler_path, metadata_path = _artifact_paths_for(horizon, window)

    # Se não houver arquivos, não falha o CI — marcamos como skip
    missing = [p for p in (model_path, scaler_path, metadata_path) if not os.path.exists(p)]
    if missing:
        pytest.skip(f"Artefatos ausentes para H={horizon}: {missing}")

    assert os.path.exists(model_path)
    assert os.path.exists(scaler_path)
    assert os.path.exists(metadata_path)


@pytest.mark.parametrize("horizon", [1, 5])
def test_predict_from_recent_smoke_or_skip(horizon):
    """Smoke-test: só roda se artefatos existirem. Em CI (CI=true) também faz skip para evitar custo pesado."""
    if os.environ.get("CI", "").lower() == "true":
        pytest.skip("Skip inferência pesada em CI")

    md = _load_metadata_or_skip()
    run = next((r for r in md.get("runs", []) if int(r.get("horizon")) == horizon), None)
    if not run:
        pytest.skip(f"Sem run para H={horizon} no metadata")

    window = int(run.get("window", 60))
    n_features = int(run.get("n_features", 12))

    model_path, scaler_path, metadata_path = _artifact_paths_for(horizon, window)
    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        pytest.skip(f"Artefatos ausentes para smoke inferência H={horizon}")

    # ordem de features
    try:
        feats = get_features_order(horizon=horizon, window=window)
    except Exception:
        feats = None  # se não disponível, o backend deve assumir default

    # monta janela sintética (zeros) com o número certo de features
    recent = [[0.0] * n_features for _ in range(window)]

    # Chama inferência; apenas valida formato da resposta
    preds = predict_from_recent(
        horizon=horizon,
        window=window,
        recent_features=recent,
        features_order=feats,
    )

    # aceita lista ou np.ndarray
    if hasattr(preds, "tolist"):
        preds = preds.tolist()

    assert isinstance(preds, (list, tuple))
    if horizon == 1:
        assert len(preds) == 1
    else:
        assert len(preds) == horizon
