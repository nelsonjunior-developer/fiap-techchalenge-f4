# fiap-techchalenge-f4


## Estrutura do Projeto

```text
tech-challenge/
│
├── data/
│   ├── raw/                     # Dados brutos (yfinance)
│   └── processed/               # Dados após limpeza, splits e janelas
│
├── notebooks/
│   └── eda.ipynb                # EDA: estatísticas, sazonalidade, ACF/PACF
│
├── src/
│   ├── data.py                  # Ingestão, limpeza, split temporal (train/val/test)
│   ├── features.py              # Janelamento, escalonamento, indicadores técnicos
│   ├── model.py                 # Definição das LSTMs (H=1 e H=5) + callbacks
│   ├── train.py                 # Pipeline de treino e salvamento de artefatos
│   ├── evaluate.py              # Backtesting, métricas (MAE/RMSE/MAPE) e gráficos
│   └── utils/                   # Helpers (logging, paths, seed, config, validações)
│
├── api/
│   ├── main.py                  # FastAPI: /health, /predict, /metadata, /metrics
│   ├── inference.py             # Carrega artefatos e executa previsão (horizon=1|5)
│   ├── schemas.py               # Pydantic: validação dos payloads
│   └── monitoring.py            # Métricas Prometheus e middlewares de latência
│
├── models/
│   ├── model_h1.h5              # Modelo Keras para H=1 (baseline)
│   ├── model_h5.h5              # Modelo Keras para H=5 (multi-saída)
│   ├── scaler.joblib            # Escalonador salvo (fit no treino)
│   └── metadata.json            # Datas, métricas, hiperparâmetros, versões
│
├── monitoring/
│   ├── prometheus.yml           # Exemplo de scrape config (opcional)
│   └── dashboards/              # Dashboards (JSON) para observabilidade (opcional)
│
├── tests/
│   ├── test_features.py         # Shapes/índices e janelas
│   ├── test_inference.py        # Carregamento e seleção H=1/H=5
│   └── test_api.py              # /health e /predict
│
├── docker/
│   ├── Dockerfile               # Imagem da API (python:3.11-slim + uvicorn)
│   └── docker-compose.yml       # API + Prometheus/Grafana (opcional)
│
├── scripts/
│   ├── fetch_data.py            # CLI: baixa dados (yfinance)
│   ├── preprocess.py            # CLI: processa e gera janelas
│   └── serve.sh                 # Sobe a API localmente
│
├── app.py                       # Streamlit consumindo a API
├── requirements.txt             # Dependências (pinned)
├── README.md                    # Guia completo (setup, execução, decisões)
├── .env.example                 # Ex.: WINDOW=60, H=5 etc.
├── .gitignore
└── .github/workflows/ci.yml     # CI: lint + testes
```

## Ambiente virtual e execução local

> Pré-requisitos: **Python 3.11+**, **pip**, **git**.

### 1) Criar e ativar o ambiente virtual

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Instalar dependências
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3) (Opcional) Configurar variáveis de ambiente
Crie o arquivo `.env` a partir do exemplo e ajuste valores conforme necessidade:
```bash
cp .env.example .env
# edite .env (ex.):
# WINDOW=60
# H=5
# TICKER=AMZN
# START_DATE=2018-01-01
```

### 4) Subir a API localmente
Com o ambiente ativo e dependências instaladas, rode:
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```
A documentação interativa (OpenAPI/Swagger) estará disponível em: `http://127.0.0.1:8000/docs`.

Teste rápido de saúde:
```bash
curl http://127.0.0.1:8000/health
```

### 5) Desativar o ambiente virtual
```bash
deactivate
```

> Dica: se preferir, crie um `Makefile` com alvos como `make venv`, `make install` e `make api` para simplificar os comandos (opcional).
```