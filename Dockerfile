# Multi-stage build: dependencies are locked by uv.lock and the runtime image has no build tools.
FROM python:3.13-slim AS build
ENV UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv UV_PYTHON_DOWNLOADS=never
RUN pip install --no-cache-dir uv==0.13.7
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev
ENV FASTEMBED_CACHE_PATH=/opt/fastembed
RUN /opt/venv/bin/python -c "from fastembed import TextEmbedding as T; T('BAAI/bge-small-en-v1.5')"

FROM python:3.13-slim
RUN useradd --create-home --uid 10001 modelmesh
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
COPY --from=build /opt/venv /opt/venv
COPY --from=build /opt/fastembed /opt/fastembed
ENV FASTEMBED_CACHE_PATH=/opt/fastembed HF_HUB_OFFLINE=1
WORKDIR /app
COPY --from=build /app/src ./src
COPY config ./config
COPY docs/contract/openapi-v1.yaml ./docs/contract/openapi-v1.yaml
COPY alembic.ini ./
COPY migrations ./migrations
USER modelmesh
EXPOSE 8000
CMD ["uvicorn", "modelmesh.api.app:app_factory", "--factory", "--host", "0.0.0.0", "--port", "8000"]
