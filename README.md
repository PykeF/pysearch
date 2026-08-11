# PySearch

An educational distributed search engine, written from scratch in Python.

> **Current status: Phase 0 — engineering foundation.**
> This repository currently contains the service skeleton only: application
> bootstrap, configuration, structured logging, and testing infrastructure.
> **No search functionality exists yet.** Indexing, ranking, storage and
> distribution are described below as the planned roadmap, not as things that
> are implemented.

## Motivation

Search engines sit at the intersection of most of the interesting problems in
backend engineering: data structures, ranking algorithms, storage, concurrency,
networking, and distributed-systems failure modes. Reaching for an off-the-shelf
engine hides all of it behind an API call.

PySearch exists to work through those problems directly. The core mechanics —
tokenization, the inverted index, BM25 scoring, document routing, shard fan-out,
top-k merging — are implemented here rather than delegated to a library,
because implementing them is the entire point. Commodity concerns (the HTTP
layer, config parsing) use well-understood libraries so the effort stays on the
parts worth understanding.

## Goals

PySearch is built to evolve, one phase at a time, from a single-node lexical
search engine into a distributed hybrid retrieval system:

```text
Lexical Retrieval
        |
        v
Distributed Lexical Search
        |
        v
Semantic / Vector Search
        |
        v
Hybrid Retrieval
```

Each phase is expected to be small enough to explain end to end, and the commit
history is meant to show the architecture actually evolving rather than being
designed up front.

## Architecture

### Current architecture (Phase 0)

A single FastAPI process. That is genuinely all there is today.

```text
HTTP client
    |
    v
FastAPI application  (app/main.py — create_app factory)
    |
    +--> app/api/      request handling            (currently: GET /health)
    +--> app/core/     configuration, logging
```

Composition happens in one place: `create_app()` reads settings, configures
logging, and registers routers. There are no abstract interfaces, no service
layer and no storage layer, because nothing in Phase 0 needs them.

### Planned architecture

Later phases are expected to introduce, roughly in this order: an indexing and
ranking core, a persistence layer with recovery, a coordinator that fans queries
out to sharded search nodes and merges the results, replication and failure
detection, and finally a vector index alongside the lexical one with rank fusion
over both. Containerization arrives with the distributed phases, where running
several nodes reproducibly becomes a real requirement.

None of that exists yet, and the shape above is a plan, not a commitment.

## Technology stack

**In use now**

| Technology | Role |
| --- | --- |
| Python 3.12+ | Implementation language |
| FastAPI | HTTP layer, request validation, OpenAPI schema |
| Pydantic / pydantic-settings | Typed models and environment-driven configuration |
| Uvicorn | ASGI server |
| pytest | Test runner |
| ruff | Linting and formatting |
| mypy | Static type checking |
| uv | Dependency management and reproducible environments |

**To be evaluated in later phases** — not dependencies of this project today:
Docker and Docker Compose (multi-node local deployment), a persistence
technology such as PostgreSQL or a custom on-disk format, Redis for caching,
gRPC for inter-node communication, NumPy, embedding models, and an ANN
library such as FAISS or an alternative. Each will be chosen when a phase
creates a concrete need for it, not before.

## Local development

### Install uv

```bash
brew install uv
```

Or, without Homebrew:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install dependencies

`uv` reads `pyproject.toml` and `uv.lock`, creating a virtual environment in
`.venv` with the correct Python version. It will download Python 3.12+ if the
machine does not already have it.

```bash
uv sync
```

### Run the application

```bash
uv run uvicorn app.main:app --reload
```

Then:

```bash
curl http://127.0.0.1:8000/health
```

which returns:

```json
{"status": "ok"}
```

Interactive API documentation is served at `http://127.0.0.1:8000/docs`.

### Run the quality gates

```bash
uv run pytest
```

```bash
uv run ruff check .
```

```bash
uv run ruff format --check .
```

```bash
uv run mypy app
```

## Configuration

All settings are optional and read from `PYSEARCH_`-prefixed environment
variables, or from a local `.env` file. Copy `.env.example` to `.env` to
override them locally; `.env` is git-ignored.

| Variable | Default | Values |
| --- | --- | --- |
| `PYSEARCH_APP_NAME` | `pysearch` | any string |
| `PYSEARCH_ENVIRONMENT` | `local` | `local`, `test`, `production` |
| `PYSEARCH_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` (case-insensitive) |

Invalid values fail loudly at startup rather than being silently ignored.

## Logging

Logs are emitted as one JSON object per line on stdout, using only the standard
library:

```json
{"timestamp": "2026-01-01T12:00:00+00:00", "level": "INFO", "logger": "app.main", "message": "application configured", "environment": "local", "log_level": "INFO"}
```

Fields passed via the standard `extra={...}` argument are merged into the
payload, so later phases can attach node and request identifiers without
changing the formatter. Uvicorn's own loggers are routed through the same
handler so server and application logs share one format.

## Roadmap

### Phase 0 — Engineering foundation ✅ *current*

FastAPI application bootstrap, environment-driven configuration, structured
logging, testing infrastructure, and repository engineering standards.

### Phase 1 — Core information retrieval

Document model, text normalization, tokenization, inverted index, posting lists,
term- and document-frequency statistics, BM25 ranking, indexing and search APIs,
and unit tests covering ranking behaviour. Implemented directly rather than
delegated to a search library.

### Phase 2 — Storage and index persistence

Persistent document and index storage, write-ahead logging where appropriate,
snapshots, recovery, document updates and deletion, and a storage abstraction.
The storage technology will be chosen from the requirements that emerge here.

### Phase 3 — Distributed search

Search nodes, deterministic document routing, sharding, a coordinator node,
inter-node communication, parallel query fan-out, distributed top-k merging, and
a cluster-aware search API. Containerization (Docker and Docker Compose) is
expected to land in this phase, where reproducible multi-node local deployment
becomes a genuine requirement.

### Phase 4 — Reliability and scalability

Replication, health checks, heartbeats, failure detection, replica selection,
failover, rebalancing, query caching, and backpressure — with the emphasis on
understanding failure modes and trade-offs.

### Phase 5 — Semantic and vector search

Document and query embeddings, vector indexing, similarity search, approximate
nearest-neighbour retrieval, a semantic search API, and recall/latency
evaluation. The ANN approach will be selected on merit; FAISS is a candidate,
not a foregone conclusion.

### Phase 6 — Hybrid search

BM25 and vector candidate retrieval, score normalization, reciprocal rank fusion
or an equivalent, configurable retrieval strategies, and measured comparison of
BM25-only versus vector-only versus hybrid retrieval.

### Phase 7 — Production engineering

Metrics, observability, benchmarking, load testing, profiling, CI/CD,
architecture documentation, failure testing, and scalability experiments.

## Repository layout

```text
app/
├── api/          HTTP routes
│   └── health.py
├── core/         configuration and logging
│   ├── config.py
│   └── logging.py
└── main.py       application factory and entry point
tests/
├── unit/
└── integration/
```

Modules for indexing, search, storage and distribution will be created when the
phase that needs them begins — not in advance.

## Independence and attribution

PySearch is an independent educational project inspired by general distributed
search-engine architecture. It is not affiliated with Elasticsearch or Elastic,
and does not contain Elasticsearch source code.
