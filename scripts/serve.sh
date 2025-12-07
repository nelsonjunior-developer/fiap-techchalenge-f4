#!/usr/bin/env bash
# scripts/serve.sh — Inicializa a API FastAPI (Uvicorn) com conveniências de ambiente
# Objetivos:
#  - Rodar a partir da raiz do repo (independente de onde você chame o script)
#  - Carregar variáveis do .env (se existir)
#  - Definir defaults sensatos e permitir flags como --reload/--host/--port
#  - Exportar PYTHONPATH para que src.* funcione
#  - Usar .venv/bin/uvicorn se existir; senão, procurar no PATH
#
# Uso:
#   bash scripts/serve.sh [--reload] [--host 0.0.0.0] [--port 8000] [--log-level INFO]
# Exemplos:
#   bash scripts/serve.sh --reload
#   bash scripts/serve.sh --host 127.0.0.1 --port 9000 --log-level DEBUG
set -euo pipefail

# Descobre a raiz do repositório a partir da pasta scripts/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

# Carrega .env (se existir) exportando variáveis
if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# Defaults (podem ser sobrescritos via .env ou flags)
API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8000}"
LOG_JSON="${LOG_JSON:-true}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
APP_IMPORT_PATH="${APP_IMPORT_PATH:-api.main:app}"
declare -a UVICORN_EXTRA_OPTS=()

usage() {
  echo "Usage: $(basename "$0") [--reload] [--host HOST] [--port PORT] [--log-level LEVEL]"
  exit 0
}

# Parse de argumentos simples
while [[ $# -gt 0 ]]; do
  case "$1" in
    --reload)
      UVICORN_EXTRA_OPTS+=("--reload"); shift ;;
    --host)
      [[ $# -ge 2 ]] || { echo "--host requer um valor" >&2; exit 2; }
      API_HOST="$2"; shift 2 ;;
    --port)
      [[ $# -ge 2 ]] || { echo "--port requer um valor" >&2; exit 2; }
      API_PORT="$2"; shift 2 ;;
    --log-level)
      [[ $# -ge 2 ]] || { echo "--log-level requer um valor" >&2; exit 2; }
      LOG_LEVEL="$2"; shift 2 ;;
    -h|--help)
      usage ;;
    *)
      echo "Arg desconhecido: $1" >&2; usage ;;
  esac
done

# Garante que os imports relativos funcionem
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

# Escolhe o binário do uvicorn (.venv se existir, senão PATH)
UVICORN_BIN="${ROOT}/.venv/bin/uvicorn"
if [[ ! -x "${UVICORN_BIN}" ]]; then
  UVICORN_BIN="$(command -v uvicorn || true)"
fi
if [[ -z "${UVICORN_BIN}" ]]; then
  echo "ERRO: uvicorn não encontrado. Rode 'make install' ou 'pip install -r requirements.txt'." >&2
  exit 1
fi

# Pré-checagem: avisa sobre artefatos ausentes (não bloqueia)
missing=()
[[ -f "models/model_h1.h5" ]] || missing+=("models/model_h1.h5")
[[ -f "models/model_h5.h5" ]] || missing+=("models/model_h5.h5")
[[ -f "models/scaler.joblib" ]] || missing+=("models/scaler.joblib")
if [[ ${#missing[@]} -gt 0 ]]; then
  echo "WARN: artefatos ausentes: ${missing[*]} — /ready falhará até treinar (make features && make train)." >&2
fi

echo "Iniciando API: ${APP_IMPORT_PATH} em ${API_HOST}:${API_PORT}"
echo "LOG_JSON=${LOG_JSON} LOG_LEVEL=${LOG_LEVEL} PYTHONPATH=${PYTHONPATH}"

# Monta comando final respeitando --reload quando definido
CMD=( "${UVICORN_BIN}" "${APP_IMPORT_PATH}" --host "${API_HOST}" --port "${API_PORT}" )
if [[ ${#UVICORN_EXTRA_OPTS[@]} -gt 0 ]]; then
  CMD+=( "${UVICORN_EXTRA_OPTS[@]}" )
fi

exec env LOG_JSON="${LOG_JSON}" LOG_LEVEL="${LOG_LEVEL}" "${CMD[@]}"
