"""
Arquiteturas LSTM (Keras) para previsão de série temporal – Tech Challenge F4

- Duas variantes via `horizon`: H=1 (baseline) e H=5 (multi-saída).
- Entrada esperada: (window, n_features).
- Perda: MSE. Métricas: MAE, RMSE, MAPE.

Decisões de design (sucintas):
- Mantemos uma arquitetura simples e clara para fins acadêmicos: LSTM -> Dropout -> Dense(horizon).
- Hiperparâmetros (units, dropout, lr) são configuráveis pelo `src/train.py`.
- `default_callbacks()` provê EarlyStopping e ReduceLROnPlateau para estabilidade.
"""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf
from tensorflow.keras import callbacks, layers, metrics, models, optimizers

# Hiperparâmetros padrão (podem ser sobrescritos no train.py)
DEFAULT_UNITS = 64
DEFAULT_DROPOUT = 0.2
DEFAULT_LR = 1e-3


def build_lstm_model(
    window: int,
    n_features: int,
    horizon: int,
    *,
    units: int = DEFAULT_UNITS,
    dropout: float = DEFAULT_DROPOUT,
    lr: float = DEFAULT_LR,
) -> tf.keras.Model:
    """Cria um modelo LSTM simples para H=1 ou H=5.

    Args:
        window: tamanho da janela temporal (timesteps).
        n_features: número de colunas/atributos por timestep.
        horizon: passos futuros a prever (1 ou 5 neste projeto).
        units: neurônios na LSTM.
        dropout: taxa de dropout após a LSTM (0–1).
        lr: taxa de aprendizado do Adam.

    Returns:
        tf.keras.Model compilado e pronto para treino.
    """
    # Entrada com shape (window, n_features)
    inputs = layers.Input(shape=(window, n_features), name="window_input")

    # Camada recorrente principal
    x = layers.LSTM(units, name="lstm_1")(inputs)

    # Regularização
    if dropout and dropout > 0:
        x = layers.Dropout(dropout, name="dropout")(x)

    # Cabeça de regressão multi-saída (horizon neurônios)
    outputs = layers.Dense(horizon, name="dense_out")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=f"lstm_h{horizon}")

    # Compilação: MSE como perda; métricas para relatório
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=[
            metrics.MeanAbsoluteError(name="mae"),
            metrics.RootMeanSquaredError(name="rmse"),
            metrics.MeanAbsolutePercentageError(name="mape"),
        ],
    )
    return model


def default_callbacks(
    patience_es: int = 10, patience_rlr: int = 5
) -> Tuple[callbacks.Callback, ...]:
    """Callbacks padrão para treino estável.

    - EarlyStopping: para quando `val_loss` não melhora, restaurando os melhores pesos.
    - ReduceLROnPlateau: reduz `lr` quando `val_loss` estagna, evitando ficar preso em platôs.
    """
    es = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience_es,
        mode="min",
        restore_best_weights=True,
        verbose=1,
    )
    rlr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=patience_rlr,
        min_lr=1e-6,
        verbose=1,
    )
    return (es, rlr)
