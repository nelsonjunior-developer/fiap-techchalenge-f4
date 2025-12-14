#!/usr/bin/env python3

"""
scripts/preprocess.py — Wrapper CLI para geração de janelas/npz (features)

Este script é um *atalho* para o pipeline de features definido em `src/features.py`.
Ele chama `python -m src.features` a partir da **raiz do repositório**, garantindo
`PYTHONPATH` e a existência das pastas `data/processed`.

Exemplos:
  python scripts/preprocess.py --ticker AMZN --window 60 --horizon 1
  python scripts/preprocess.py --ticker AMZN --window 60 --horizon 5
  python scripts/preprocess.py --ticker AMZN --window 60 --horizon both  # roda 1 e 5

Dica: você também pode usar o Makefile:
  make features-h1   # horizon=1
  make features-h5   # horizon=5
  make features      # both
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable


def _repo_root() -> Path:
    """Retorna a raiz do repositório assumindo este arquivo em `scripts/`."""
    return Path(__file__).resolve().parents[1]


def _ensure_dirs(root: Path) -> None:
    (root / "data" / "processed").mkdir(parents=True, exist_ok=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Wrapper para executar o módulo src.features e gerar janelas/npz para treino."
        )
    )
    p.add_argument("--ticker", default="AMZN", help="Ticker (default: AMZN)")
    p.add_argument(
        "--window",
        type=int,
        default=60,
        help="Tamanho da janela temporal (default: 60)",
    )
    p.add_argument(
        "--horizon",
        default="both",
        choices=["1", "5", "both"],
        help="Horizonte de previsão: 1, 5 ou both (default: both)",
    )
    return p.parse_args(argv)


def _run_features_once(root: Path, ticker: str, window: int, horizon: int) -> int:
    cmd = [
        sys.executable,
        "-m",
        "src.features",
        "--ticker",
        ticker,
        "--window",
        str(window),
        "--horizon",
        str(horizon),
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(root))  # garante imports de src.*

    print(f"[preprocess] Executando: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(root), env=env, check=False)
    return int(proc.returncode)


def _iter_horizons(opt: str) -> Iterable[int]:
    if opt == "both":
        return (1, 5)
    return (int(opt),)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = _repo_root()
    _ensure_dirs(root)

    rc_total = 0
    for h in _iter_horizons(args.horizon):
        rc = _run_features_once(root, args.ticker, args.window, h)
        rc_total = rc_total or rc  # mantém primeiro código não-zero
        if rc != 0:
            print(f"[preprocess] Aviso: retorno != 0 para horizon={h} (rc={rc}).")
    return rc_total


if __name__ == "__main__":
    raise SystemExit(main())
