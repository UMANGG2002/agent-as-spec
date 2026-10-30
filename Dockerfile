# One image, three process roles (api | runner | relay), selected by the
# command. Multi-stage: build wheels with uv, ship a slim runtime as non-root.
FROM python:3.12-slim AS build
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
RUN pip install --no-cache-dir "uv>=0.5"
WORKDIR /app
COPY pyproject.toml uv.lock* ./
# Uses uv.lock when it matches pyproject.toml, re-resolves otherwise.
RUN uv sync --no-install-project --no-dev --extra litellm --extra openapi-mcp --extra memory
COPY app ./app
COPY agents ./agents
COPY migrations ./migrations
COPY scripts ./scripts

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PATH="/app/.venv/bin:$PATH" \
    ENVIRONMENT=production MEM0_TELEMETRY=False
RUN useradd --uid 10001 --create-home app
WORKDIR /app
COPY --from=build --chown=app:app /app /app
USER app
EXPOSE 8000
# API by default. Runner: `python -m app.sessions.runner`. Relay: `python -m app.relay_main`.
# Migrations (owner role, once per deploy): `python -m app.core.migrate`.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--loop", "uvloop", \
     "--http", "httptools", "--workers", "2", "--no-access-log", "--timeout-graceful-shutdown", "25"]
