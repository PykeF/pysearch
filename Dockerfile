# Multi-stage: dependencies are resolved once in a builder, and the runtime
# image carries only the virtual environment and the application.
FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.3 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /srv

# Dependency layer first, so editing application code does not re-resolve.
COPY pyproject.toml uv.lock README.md ./
COPY app ./app
# The semantic extra brings numpy, tokenizers and the model2vec loader. It adds
# tens of megabytes rather than the gigabytes a torch-based embedding stack
# would, which is a large part of why that model was chosen. Drop --extra
# semantic for a lexical-only image.
RUN uv sync --frozen --no-dev --extra semantic


FROM python:3.13-slim AS runtime

# Runs unprivileged: the process only needs to read its code and write its own
# database, and nothing about a search node needs root.
RUN useradd --create-home --uid 10001 pysearch

WORKDIR /srv

COPY --from=builder --chown=pysearch:pysearch /srv/.venv /srv/.venv
COPY --from=builder --chown=pysearch:pysearch /srv/app /srv/app

# HF_HOME points at the data volume so the embedding model is downloaded once on
# first start and survives restarts. Bake the model into the image instead if
# startup must not depend on the network.
ENV PATH="/srv/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYSEARCH_STORAGE_PATH=/data/pysearch.db \
    HF_HOME=/data/models

# Each node writes its own database here; Compose mounts a separate volume per
# shard, because two shards sharing one file would not be sharding at all.
RUN mkdir -p /data && chown pysearch:pysearch /data
VOLUME ["/data"]

USER pysearch
EXPOSE 8000

# No --reload: that is a development convenience, configured in Compose.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
