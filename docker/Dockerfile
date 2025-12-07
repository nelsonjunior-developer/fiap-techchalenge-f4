# docker/Dockerfile – Tech Challenge F4
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Dependências mínimas do sistema (mantidas enxutas)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instala dependências primeiro (melhora cache em rebuilds)
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copia o projeto (código, api, src, etc.)
COPY . .

# Usuário não-root por segurança
RUN useradd -ms /bin/bash appuser
USER appuser

# Python encontra os pacotes do projeto
ENV PYTHONPATH=/app
# Logs estruturados (podem ser sobrescritos via ambiente/.env)
ENV LOG_JSON=true \
    LOG_LEVEL=INFO

EXPOSE 8000

# Entrypoint da API (FastAPI/Uvicorn)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
