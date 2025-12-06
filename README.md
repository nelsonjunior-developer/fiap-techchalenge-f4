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


## Treinamento (LSTM H=1 e H=5)

A sequência abaixo treina os dois modelos exigidos (baseline H=1 e solução principal H=5) de forma reprodutível.

> Pré-requisitos: ambiente virtual ativo e dependências instaladas; diretórios de dados criados pelo passo de ingestão.

1) **Ingestão/Pré-processamento (se ainda não fez):**
```bash
python -m src.data --ticker AMZN --start 2018-01-01
```

2) **Geração de janelas/arrays (.npz):**
```bash
# baseline H=1
python -m src.features --ticker AMZN --window 60 --horizon 1
# solução principal H=5 (multi-saída)
python -m src.features --ticker AMZN --window 60 --horizon 5
```

3) **Treino dos modelos (H=1 e H=5):**
```bash
# Observação: instale TensorFlow conforme seu ambiente, se ainda não o fez.
# Intel/Linux/Windows
pip install "tensorflow>=2.15,<3"
# Apple Silicon (opcional)
# pip install tensorflow-macos tensorflow-metal

# Treinar ambos (H=1 e H=5)
python -m src.train --horizon both --window 60 --epochs 50 --batch_size 128
```

**Saídas esperadas:**
- `models/model_h1.h5` e `models/model_h5.h5` (pesos dos modelos)
- `models/metadata.json` (métricas, hiperparâmetros, versões)
- `models/scaler.joblib` (gerado no `features.py`)

---

## Avaliação (métricas, resíduos e backtesting)

Rode a avaliação para cada horizonte desejado. As métricas são calculadas na **escala original** do preço (Close), há comparação com o baseline ingênuo (persistência) e são gerados gráficos de diagnóstico.

```bash
# Avaliar H=1
python -m src.evaluate --horizon 1 --window 60

# Avaliar H=5
python -m src.evaluate --horizon 5 --window 60

# (opcionais) desabilitar gráficos de resíduos e/ou backtesting simples
python -m src.evaluate --horizon 5 --window 60 --no-residuals --no-walkforward
```

**Saídas esperadas:**
- Relatório: `models/evaluation_report_AMZN_w60_h{H}.json`
- Gráficos em `models/plots/`:
  - `pred_vs_true_AMZN_w60_h{H}.png`
  - `residuals_time_AMZN_w60_h{H}.png`
  - `residuals_hist_AMZN_w60_h{H}.png`
  - `walkforward_mae_t1_AMZN_w60_h{H}.png`

---

## O que cada gráfico mostra

**1) `pred_vs_true_*.png`** — *Série real vs. prevista*  
Para H=1 plota a série completa; para H=5 mostra o passo **t+1**. Ajuda a ver alinhamento, atrasos e períodos de maior erro.

**2) `residuals_time_*.png`** — *Resíduos no tempo*  
Resíduo = `y_true − y_pred` no domínio original. O ideal é média próxima de 0 e sem padrão visível. Tendências, blocos de erro alto ou heterocedasticidade sinalizam oportunidades de melhoria (novas features, transformações, tuning).

**3) `residuals_hist_*.png`** — *Distribuição dos resíduos*  
Mostra simetria e caudas. Um histograma centrado em 0 e relativamente estreito é desejável. Caudas pesadas/outliers indicam choques/regimes; considerar robustez extra.

**4) `walkforward_mae_t1_*.png`** — *Backtesting simples (modelo fixo)*  
Série do erro absoluto no passo **t+1** ao longo das amostras de teste (sem re-treino). Útil para checar **estabilidade temporal**: picos localizados sugerem regimes específicos onde o modelo perde desempenho.

---

## Dicas & Solução de Problemas
- **Arquivos .npz não encontrados:** gere as janelas com `src.features` para o `--horizon` correspondente.  
- **Aviso do Keras sobre HDF5:** ignorável; o `evaluate.py` usa `compile=False`. Se quiser, podemos salvar também no formato `.keras`.  
- **Erro de backend gráfico:** o `evaluate.py` já força `matplotlib` no backend `Agg`; apenas garanta `matplotlib` instalado.
```
 
---

## API REST (FastAPI) — Endpoints e Exemplos

### Como subir a API
Com o ambiente ativo e dependências instaladas:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
# (opcional) recarregamento automático em dev
# uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

- Documentação interativa (Swagger/OpenAPI): `http://127.0.0.1:8000/docs`
- As previsões são sempre devolvidas na **escala original** do preço **Close**.
- A API usa os artefatos em `models/` (ex.: `model_h1.h5`, `model_h5.h5`, `scaler.joblib`, `metadata.json`).

> Dica: se você não tiver os `.npz` (ordem de features) e o `scaler.joblib`, gere-os com `python -m src.features` (para cada `--horizon`).

---

### 1) `GET /health`
**Intenção:** Healthcheck para orquestradores e monitoramento.

```bash
curl -s http://127.0.0.1:8000/health | jq .
```
**Resposta (exemplo):**
```json
{
  "status": "ok",
  "ticker": "AMZN",
  "window_default": 60
}
```

---

### 2) `GET /metadata`
**Intenção:** Retorna o conteúdo de `models/metadata.json` (métricas, hiperparâmetros, versões).

```bash
curl -s http://127.0.0.1:8000/metadata | jq .
```

---

### 3) `GET /features-order`
**Intenção:** Descobrir a **ordem oficial** de features (e `n_features`) usada no treino, para um `horizon` e `window`.

```bash
curl -s "http://127.0.0.1:8000/features-order?horizon=1&window=60" | jq .
```
**Resposta (exemplo):**
```json
{
  "horizon": 1,
  "window": 60,
  "n_features": 12,
  "features": [
    "Open", "High", "Low", "Close", "Volume",
    "ret1", "logret1", "vol20", "rsi14", "macd", "macd_signal", "macd_hist"
  ]
}
```

---

### 4) `POST /predict`
**Intenção:** Fazer inferência **enviando as features já processadas** na mesma ordem do treino.

- **Payload mínimo**: `horizon`, `window` (opcional; usa default do settings), `recent_features` como matriz `[window, n_features]`.
- **Validação opcional**: inclua `features_order` com a lista de nomes **exatamente** na ordem do treino.

**Exemplo (teste rápido, sem `features_order`)** — matriz de zeros só para validar o endpoint:
```bash
python - <<'PY' > payload.json
import json
row=[0.0]*12            # n_features (ver em /features-order)
payload={
  "horizon": 1,
  "window": 60,
  "recent_features": [row]*60   # 60 timesteps
}
print(json.dumps(payload))
PY

curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  --data-binary @payload.json | jq .
```

**Exemplo (com `features_order`)** — protege contra ordem incorreta:
```bash
python - <<'PY' > payload.json
import json
feats=["Open","High","Low","Close","Volume","ret1","logret1","vol20","rsi14","macd","macd_signal","macd_hist"]
row=[0.0]*len(feats)
payload={
  "horizon": 5,
  "window": 60,
  "features_order": feats,
  "recent_features": [row]*60
}
print(json.dumps(payload))
PY

curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  --data-binary @payload.json | jq .
```

> Erros comuns: `features_order não coincide` (use `/features-order`) ou `n_features` incorreto (ajuste a largura das linhas em `recent_features`).

---

### 5) `POST /predict-ticker`
**Intenção:** Fazer inferência **apenas informando o ticker**; a API baixa OHLCV, calcula as **mesmas 12 features** do treino, aplica o `scaler` e prevê.

**Payload:**
- `horizon`: 1 ou 5
- `window` (opcional; default do settings)
- `ticker` (opcional; default `AMZN`)
- `lookback_days` (opcional; default `180`) — histórico para estabilizar os indicadores

**Exemplos:**
```bash
# H=1
tmp='{"horizon":1,"window":60,"ticker":"AMZN","lookback_days":180}'
curl -s -X POST http://127.0.0.1:8000/predict-ticker \
  -H "Content-Type: application/json" -d "$tmp" | jq .

# H=5
tmp='{"horizon":5,"window":60,"ticker":"AMZN","lookback_days":180}'
curl -s -X POST http://127.0.0.1:8000/predict-ticker \
  -H "Content-Type: application/json" -d "$tmp" | jq .
```

> A ordem de features usada internamente é a mesma dos `.npz`. Se os `.npz`/`scaler.joblib` não existirem, gere-os com `python -m src.features`.

---

### 6) `GET /metrics`
**Intenção:** Expor métricas Prometheus (contagem por método/rota/status e latência por rota).

```bash
curl -s http://127.0.0.1:8000/metrics | head -n 20
```
**Uso típico no Prometheus (exemplo):**
```yaml
scrape_configs:
  - job_name: tech-challenge-api
    static_configs:
      - targets: ["localhost:8000"]
```

---

### Códigos de erro (mais comuns)
- **400**: inconsistências de artefatos/ordem de features (ex.: `.npz` ausente)
- **422**: payload inválido (ex.: `horizon` fora de {1,5}, shapes incorretos)
- **503**: endpoint indisponível (ex.: módulo de inferência por ticker ausente)

``` 