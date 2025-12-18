# =============================
# Makefile — Tech Challenge F4
# =============================

# Binaries / defaults
PY ?= .venv/bin/python
API_HOST ?= 0.0.0.0
API_PORT ?= 8000
LOG_JSON ?= true
LOG_LEVEL ?= INFO
API_BASE_URL ?= http://127.0.0.1:8000

# Data / training defaults
TICKER ?= AMZN
START ?= 2018-01-01
WINDOW ?= 60
EPOCHS ?= 50
BATCH ?= 128

# Colors (pretty help)
BLUE := \033[36m
NC := \033[0m

.PHONY: help install env data features-h1 features-h5 features train eval-h1 eval-h5 api api-dev smoke ready docker-build docker-run compose-up compose-down clean lint format test eda-run streamlit

help: ## Lista alvos disponíveis
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sed -e 's/:.*## /\t- /' | sort

install: ## Cria .venv e instala requirements
	python3 -m venv .venv
	. .venv/bin/activate; pip install --upgrade pip; pip install -r requirements.txt

env: ## Cria .env a partir de .env.example, se não existir
	@test -f .env || cp .env.example .env

# ---- Data pipeline (via scripts) ----
DATA_FLAGS :=
ifdef NO_WINSORIZE
DATA_FLAGS += --no-winsorize
endif

data: ## Baixa/prepara dados (yfinance) — usa scripts/fetch_data.py
	$(PY) scripts/fetch_data.py --ticker $(TICKER) --start $(START) $(DATA_FLAGS)

features-h1: ## Gera features (npz) para H=1 — via scripts/preprocess.py
	$(PY) scripts/preprocess.py --ticker $(TICKER) --window $(WINDOW) --horizon 1

features-h5: ## Gera features (npz) para H=5 — via scripts/preprocess.py
	$(PY) scripts/preprocess.py --ticker $(TICKER) --window $(WINDOW) --horizon 5

features: ## Gera features para H=1 e H=5 (ambos)
	$(PY) scripts/preprocess.py --ticker $(TICKER) --window $(WINDOW) --horizon both

# ---- Train / Evaluate ----
train: ## Treina H=1 e H=5 (salva modelos em models/)
	$(PY) -m src.train --horizon both --window $(WINDOW) --epochs $(EPOCHS) --batch_size $(BATCH)

eval-h1: ## Avalia H=1 e salva plots
	$(PY) -m src.evaluate --horizon 1 --window $(WINDOW)

eval-h5: ## Avalia H=5 e salva plots
	$(PY) -m src.evaluate --horizon 5 --window $(WINDOW)

# ---- EDA ----
eda-run: ## Executa notebooks/eda.ipynb e atualiza no lugar
	$(PY) -m jupyter nbconvert --to notebook --inplace --execute notebooks/eda.ipynb

# ---- Streamlit ----
streamlit: ## Sobe o frontend Streamlit
	API_BASE_URL=$(API_BASE_URL) . .venv/bin/activate; streamlit run app.py --server.port 8501

# ---- API (fonte única via scripts/serve.sh) ----
api: ## Sobe a API via scripts/serve.sh (fonte única de execução)
	@echo ">> Iniciando API com scripts/serve.sh (LOG_JSON=$(LOG_JSON), LOG_LEVEL=$(LOG_LEVEL), PORT=$(API_PORT))"
	LOG_JSON=$(LOG_JSON) LOG_LEVEL=$(LOG_LEVEL) API_HOST=$(API_HOST) API_PORT=$(API_PORT) \
	  bash scripts/serve.sh --host $(API_HOST) --port $(API_PORT) --log-level $(LOG_LEVEL)

api-dev: ## API (dev) via scripts/serve.sh --reload
	@echo ">> Iniciando API (dev) com scripts/serve.sh (reload, LOG_JSON=false, LOG_LEVEL=DEBUG, PORT=$(API_PORT))"
	LOG_JSON=false LOG_LEVEL=DEBUG API_HOST=$(API_HOST) API_PORT=$(API_PORT) \
	  bash scripts/serve.sh --reload --host $(API_HOST) --port $(API_PORT) --log-level DEBUG

convert-models: ## (Opcional) Converte modelos .h5 para .keras
	. .venv/bin/activate; python scripts/convert_models.py

# ---- Smoke / Ready ----
smoke: ## Smoke test: /health e /features-order
	@echo "${BLUE}GET /health${NC}"; curl -s http://127.0.0.1:$(API_PORT)/health | jq . || true
	@echo "${BLUE}GET /features-order?horizon=1&window=$(WINDOW)${NC}"; curl -s "http://127.0.0.1:$(API_PORT)/features-order?horizon=1&window=$(WINDOW)" | jq . || true

ready: ## Readiness: verifica artefatos e API
	@echo "${BLUE}GET /ready${NC}"; curl -s http://127.0.0.1:$(API_PORT)/ready | jq . || true

# ---- Docker ----
docker-build: ## Build da imagem da API
	docker build -t tech-f4-api -f docker/Dockerfile .

docker-run: ## Executa a imagem mapeando porta 8000
	docker run --rm -p $(API_PORT):8000 --env-file .env tech-f4-api

compose-up: ## Sobe API + Prometheus/Grafana + Streamlit via Compose
	docker compose -f docker/docker-compose.yml up --build

compose-down: ## Derruba o Compose
	docker compose -f docker/docker-compose.yml down -v --remove-orphans

# ---- Qualidade / Outros ----
lint: ## Ruff check (se instalado)
	@if [ -x .venv/bin/ruff ]; then \
	  .venv/bin/ruff check . ; \
	elif command -v ruff >/dev/null 2>&1; then \
	  ruff check . ; \
	else \
	  echo "ruff não instalado; skip" ; \
	fi

format: ## Ruff format (se instalado)
	@command -v ruff >/dev/null 2>&1 && ruff format . || echo "ruff não instalado; skip"

test: ## Pytest (se instalado)
	@command -v pytest >/dev/null 2>&1 && pytest -q || echo "pytest não instalado; skip"

clean: ## Remove npz, modelos e plots
	rm -f data/processed/*.npz
	rm -f models/model_h*.h5 models/*.keras models/scaler.joblib models/metadata.json
	rm -rf models/plots
