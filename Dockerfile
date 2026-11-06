# Multi-stage, non-root, slim serving image (no MLflow needed at serve time)
FROM python:3.12-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir --prefix=/install ".[api]"

FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl && rm -rf /var/lib/apt/lists/* && useradd --create-home app
WORKDIR /app
COPY --from=build /install /usr/local
COPY src ./src
COPY api ./api
# CI copies the exported champion into state/serving before building
COPY state ./state
RUN chown -R app /app
USER app
ENV PYTHONPATH=/app/src:/app PORT=8000 GRIDOPS_STATE=/app/state
EXPOSE 8000
HEALTHCHECK --interval=20s --timeout=3s CMD curl -fs http://localhost:${PORT}/health || exit 1
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --workers 2"]
