import glob
import os
import re

import numpy as np
import pytest

DATA_DIR = "data/processed"


def _pick_any_npz(pattern: str):
    files = sorted(glob.glob(os.path.join(DATA_DIR, pattern)))
    return files[0] if files else None


def _load_npz(path: str):
    data = np.load(path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    features = list(data["features"])
    return X, y, features


@pytest.mark.parametrize("horizon", [1, 5])
def test_npz_shapes_match_features(horizon):
    """
    Garante que n_features em X coincide com len(features) e y tem dimensão esperada.
    Busca qualquer split (train/val/test) para o horizonte informado.
    """
    # procura qq arquivo *h{horizon}.npz (train/val/test)
    candidate = (
        _pick_any_npz(f"train_*_h{horizon}.npz")
        or _pick_any_npz(f"val_*_h{horizon}.npz")
        or _pick_any_npz(f"test_*_h{horizon}.npz")
    )
    if not candidate:
        pytest.skip(f"Sem npz para H={horizon} em {DATA_DIR}")
    X, y, feats = _load_npz(candidate)

    # X: (n amostras, window, n_features)
    assert X.ndim == 3 and X.shape[0] > 0
    n_features = X.shape[2]
    assert n_features == len(feats), "n_features de X difere de len(features)"

    # y: (n amostras, H) para H>1; (n amostras,) ou (n amostras,1) para H=1
    if horizon == 1:
        assert y.ndim in (1, 2)
        if y.ndim == 2:
            assert y.shape[1] == 1
    else:
        assert y.ndim == 2 and y.shape[1] == horizon


@pytest.mark.parametrize("horizon", [1, 5])
def test_window_length_is_constant(horizon):
    """
    Verifica se o comprimento da janela (X.shape[1]) é consistente com o sufixo _wXX_ do arquivo.
    """
    candidate = (
        _pick_any_npz(f"train_*_h{horizon}.npz")
        or _pick_any_npz(f"val_*_h{horizon}.npz")
        or _pick_any_npz(f"test_*_h{horizon}.npz")
    )
    if not candidate:
        pytest.skip(f"Sem npz para H={horizon} em {DATA_DIR}")

    # extrai window do nome do arquivo: *_w{window}_h{horizon}.npz
    m = re.search(r"_w(\d+)_h{}".format(horizon), os.path.basename(candidate))
    if not m:
        pytest.skip(f"Não foi possível inferir window a partir do nome: {candidate}")
    window_from_name = int(m.group(1))

    X, y, feats = _load_npz(candidate)
    assert X.shape[1] == window_from_name, "Window nos dados difere do nome do arquivo"


def test_features_non_empty_and_finite():
    """
    Garante que há pelo menos um .npz e que X/Y não são vazios nem contêm NaN/Inf.
    """
    candidate = (
        _pick_any_npz("train_*_h*.npz")
        or _pick_any_npz("val_*_h*.npz")
        or _pick_any_npz("test_*_h*.npz")
    )
    if not candidate:
        pytest.skip(f"Sem npz em {DATA_DIR}")

    X, y, feats = _load_npz(candidate)
    assert X.size > 0 and y.size > 0 and len(feats) > 0
    assert np.isfinite(X).all(), "X contém NaN/Inf"
    assert np.isfinite(y).all(), "y contém NaN/Inf"
