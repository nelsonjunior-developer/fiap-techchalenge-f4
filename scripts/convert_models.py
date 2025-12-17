"""
Utilitário para converter modelos .h5 para o formato .keras compatível com TF 2.15.

Uso:
    python scripts/convert_models.py

Pré-requisitos:
    - models/model_h1.h5 e models/model_h5.h5 existentes
    - tensorflow==2.15.1 instalado
"""

from __future__ import annotations

from pathlib import Path

import tensorflow as tf


def convert_model(src: Path, dst: Path) -> None:
    if not src.exists():
        print(f"[warn] arquivo não encontrado: {src}")
        return
    print(f"[info] carregando {src} (tf={tf.__version__})")
    model = tf.keras.models.load_model(src, compile=False)
    print(f"[info] salvando em {dst}")
    model.save(dst, save_format="keras")


def main() -> None:
    models_dir = Path("models")
    pairs = [
        (models_dir / "model_h1.h5", models_dir / "model_h1.keras"),
        (models_dir / "model_h5.h5", models_dir / "model_h5.keras"),
    ]
    for src, dst in pairs:
        convert_model(src, dst)
    print("[done] conversão finalizada")


if __name__ == "__main__":
    main()
