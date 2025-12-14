"""
Monitoramento e métricas Prometheus para a API FastAPI.

Este módulo isola a instrumentação (counters/histogram) e o middleware de
latência, deixando `api/main.py` mais enxuto e alinhado com a estrutura do projeto.

Como usar em `api/main.py`:

    from api.monitoring import prometheus_middleware, metrics_endpoint

    app.middleware("http")(prometheus_middleware)

    @app.get("/metrics", tags=["monitoring"])
    def metrics():
        return metrics_endpoint()

Observação: usamos um `CollectorRegistry` próprio para exportar somente métricas da aplicação.
"""

from __future__ import annotations

import time
from typing import Awaitable, Callable

from fastapi import Request, Response
from fastapi.responses import PlainTextResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)

# -----------------------------------------------------------------------------
# Registry e métricas
# -----------------------------------------------------------------------------

# Registry dedicado (evita exportar métricas de outros libs/processos)
REGISTRY: CollectorRegistry = CollectorRegistry()

# Contador de requisições HTTP por método, endpoint e status
HTTP_REQUESTS: Counter = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "http_status"],
    registry=REGISTRY,
)

# Histograma de latência por método e endpoint (segundos)
HTTP_LATENCY: Histogram = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint"],
    registry=REGISTRY,
)

# -----------------------------------------------------------------------------
# Middleware de latência/contagem
# -----------------------------------------------------------------------------


async def prometheus_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """
    Middleware assíncrono para instrumentar todas as requisições:
    - mede latência (Histogram) por método/endpoint;
    - incrementa contadores por método/endpoint/status.
    """
    start = time.perf_counter()
    try:
        response = await call_next(request)
        status = response.status_code
    except Exception:
        status = 500
        HTTP_REQUESTS.labels(request.method, request.url.path, str(status)).inc()
        HTTP_LATENCY.labels(request.method, request.url.path).observe(
            time.perf_counter() - start
        )
        raise
    else:
        HTTP_REQUESTS.labels(request.method, request.url.path, str(status)).inc()
        HTTP_LATENCY.labels(request.method, request.url.path).observe(
            time.perf_counter() - start
        )
        return response


# -----------------------------------------------------------------------------
# Endpoint /metrics
# -----------------------------------------------------------------------------


def metrics_endpoint() -> PlainTextResponse:
    """
    Retorna as métricas no formato do Prometheus exposition format.
    Registrar em `main.py` como handler para GET /metrics.
    """
    return PlainTextResponse(
        generate_latest(REGISTRY).decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )
