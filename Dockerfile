# =============================================================================
# Dockerfile — build multi-stage para a API de inferência (FastAPI).
# Usa uv para instalação rápida e reprodutível; roda como usuário não-root.
# =============================================================================

# ---- Stage 1: builder -------------------------------------------------------
FROM python:3.12-slim AS builder

# Instala o uv (gerenciador de pacotes).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Copia manifestos primeiro para aproveitar o cache de camadas.
COPY pyproject.toml uv.lock* ./

# Instala apenas as dependências de runtime + extra 'api' (sem dev).
RUN uv sync --no-dev --extra api --frozen --no-install-project || \
    uv sync --no-dev --extra api --no-install-project

# ---- Stage 2: runtime -------------------------------------------------------
FROM python:3.12-slim AS runtime

# Usuário não-root.
RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

# Copia o ambiente virtual pronto do builder.
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copia o código-fonte e artefatos necessários ao serving.
COPY src ./src
COPY app ./app
COPY configs ./configs
COPY models ./models

USER appuser

EXPOSE 8000

# Healthcheck simples no endpoint /health.
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
