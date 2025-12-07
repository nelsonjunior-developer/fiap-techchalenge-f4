#!/usr/bin/env python3

"""
scripts/fetch_data.py — Wrapper CLI para ingestão de dados (yfinance)

Atalho para executar `python -m src.data` a partir da **raiz do repositório**,
configurando `PYTHONPATH` e garantindo as pastas de dados.

Uso:
  python scripts/fetch_data.py --ticker AMZN --start 2018-01-01 [--end AAAA-MM-DD]

Observações:
- Padrões seguem o enunciado (AMZN, início 2018-01-01).
- Cria `data/raw` e `data/processed` se faltarem.
- Flag opcional `--no-winsorize` é repassada ao src.data.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_data_dirs(root: Path) -> None:
    (root / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (root / "data" / "processed").mkdir(parents=True, exist_ok=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Wrapper para executar o módulo src.data com parâmetros de ingestão de dados.",
    )
    p.add_argument("--ticker", default="AMZN", help="Ticker (default: AMZN)")
    p.add_argument("--start", default="2018-01-01", help="AAAA-MM-DD (default: 2018-01-01)")
    p.add_argument("--end", default=None, help="AAAA-MM-DD (opcional)")
    p.add_argument(
        "--no-winsorize",
        action="store_true",
        help="Desativa winsorização de outliers no preparo (repassa ao src.data)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = _repo_root()
    _ensure_data_dirs(root)

    cmd = [
        sys.executable,
        "-m",
        "src.data",
        "--ticker",
        args.ticker,
        "--start",
        args.start,
    ]
    if args.end:
        cmd.extend(["--end", args.end])
    if args.no_winsorize:
        cmd.append("--no-winsorize")

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(root))

    print(f"[fetch_data] Script: {__file__}")
    print("[fetch_data] Executando:", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, cwd=str(root), env=env, check=False)
        return int(proc.returncode)
    except KeyboardInterrupt:
        print("[fetch_data] Interrompido pelo usuário.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
