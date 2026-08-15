# Development

Setup, workflows, and every command this project supports.

- [Setup](#setup)
- [Running a node](#running-a-node)
- [Quality gates](#quality-gates)
- [Tests](#tests)
- [Semantic and hybrid search](#semantic-and-hybrid-search)
- [The demo](#the-demo)
- [Running a real cluster](#running-a-real-cluster)
- [Benchmarks and evaluation](#benchmarks-and-evaluation)
- [Docker](#docker)
- [Configuration reference](#configuration-reference)
- [Logging](#logging)
- [Repository conventions](#repository-conventions)

## Setup

The project is managed with [uv](https://docs.astral.sh/uv/).

```bash
brew install uv
```

Or, without Homebrew:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then:

```bash
uv sync
```

Python 3.12 or newer is required. `.python-version` pins **3.13** as the
recommended local interpreter — the same version the Dockerfile builds on. CI
runs the full gates on both 3.12 and 3.13, so the declared lower bound is
verified rather than assumed.

`uv sync` installs runtime and dev dependencies but **not** the semantic extra,
so a fresh clone needs no model download. See
[Semantic and hybrid search](#semantic-and-hybrid-search) for that.

## Running a node

```bash
uv run uvicorn app.main:app --reload
```

Interactive API documentation is at `http://127.0.0.1:8000/docs`.

A default node is `single`: a complete standalone search engine that needs to
know nothing about clustering.

```bash
curl -X PUT localhost:8000/documents/doc-1 \
  -H 'content-type: application/json' \
  -d '{"text": "Distributed systems make search scalable."}'
```

```bash
curl 'localhost:8000/search?q=distributed+search'
```

## Quality gates

All four must pass. CI runs exactly these commands, on Python 3.12 and 3.13:

```bash
uv run ruff check .
```

```bash
uv run ruff format --check .
```

```bash
uv run mypy app
```

```bash
uv run pytest
```

`mypy` runs in `strict` mode with `warn_unreachable`. Ruff enforces pycodestyle,
pyflakes, import sorting, naming, pyupgrade, bugbear, simplification and
ruff-specific rules at a 100-character line length.

## Tests

```text
tests/unit/          no I/O, no network, no model
tests/integration/   real HTTP via TestClient, real SQLite, in-process clusters
```

The full suite is **475 tests**; `uv run pytest` runs 468 of them and deselects
7. It never touches the network: the suite runs against a deterministic fake
embedder, which is why semantic and hybrid behaviour can be tested exhaustively
without a model.

The 7 deselected tests load the real pinned model and are behind a marker:

```bash
uv run --extra semantic pytest -m semantic_model
```

These are **not** run in CI — that would make every build depend on a Hugging
Face download. They are a local pre-release check.

Useful subsets:

```bash
uv run pytest tests/unit
```

```bash
uv run pytest -k hybrid
```

## Semantic and hybrid search

Semantic retrieval needs the optional extra and is off by default:

```bash
uv sync --extra semantic
```

```bash
PYSEARCH_SEMANTIC_ENABLED=true uv run --extra semantic uvicorn app.main:app
```

The pinned model (~30 MB) downloads once on first start. Then all three modes
are reachable on the same corpus:

```bash
curl 'localhost:8000/search?q=car+maintenance'
```

```bash
curl 'localhost:8000/search/semantic?q=car+maintenance'
```

```bash
curl 'localhost:8000/search/hybrid?q=car+maintenance&explain=true'
```

`explain=true` adds the underlying BM25 and cosine scores to each hit without
changing the fusion score or the ordering.

## The demo

One script walks through the whole engine without starting a web server:

```bash
uv run python scripts/demo.py
```

It covers the inverted index internals, the engine, a restart proving durability,
and — when the semantic extra is installed — semantic and hybrid search with
fusion ranks shown. Without the extra it prints the lexical sections and tells
you how to see the rest, rather than failing.

```bash
uv run --extra semantic python scripts/demo.py
```

Everything happens in a temporary directory, so the demo leaves nothing behind.

## Running a real cluster

**This is the canonical distributed demo.** No containers are involved — these
are ordinary OS processes talking real HTTP over real sockets.

```bash
uv run python scripts/run_cluster.py --replication-factor 2
```

That starts **1 coordinator + 3 logical shards × 2 copies = 7 processes**, each
with its own database, waits for the cluster to report ready, and prints the
coordinator's address. Ctrl-C stops everything.

```bash
uv run python scripts/run_cluster.py --help
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--shards` | 3 | logical shards |
| `--replication-factor` | 1 | copies per logical shard (1 or 2) |
| `--base-port` | 9000 | first shard-node port |
| `--coordinator-port` | 8000 | coordinator port |
| `--data-dir` | temporary | where the databases go |

### Watching failover

With the cluster running:

```bash
curl -X PUT localhost:8000/documents/doc-1 \
  -H 'content-type: application/json' \
  -d '{"text": "distributed search"}'
```

The response names the shard that took the write.

```bash
curl localhost:8000/cluster/status
```

Each logical shard reports its copies with role, state and generation, plus
`search_available` and `write_available` separately.

Now **kill one primary process** and search again. The results are identical —
served from that shard's replica — while writes to that logical shard start
returning 503, because there is no automatic promotion. Restart the process and
it resynchronizes from its own database and rejoins.

To run the cluster with semantic and hybrid search enabled:

```bash
PYSEARCH_SEMANTIC_ENABLED=true uv run --extra semantic python scripts/run_cluster.py --replication-factor 2
```

## Benchmarks and evaluation

Full results and their caveats are in [evaluation.md](evaluation.md). The
commands:

```bash
uv run --extra semantic python scripts/evaluate_retrieval.py
```

```bash
uv run --extra semantic python scripts/evaluate_retrieval.py --develop
```

```bash
uv run --extra semantic python scripts/semantic_benchmark.py
```

```bash
uv run python scripts/rebuild_benchmark.py
```

None of these require editing source code, and none of them modify the labelled
evaluation data.

## Docker

> **Container runtime verification has not been executed.** Docker is not
> installed on the development machine, so the image has never been built or run
> here. The configuration below is provided and statically validated, not
> runtime-verified. The distributed system does not depend on Docker and was
> verified with real multi-process clusters instead — see
> [Running a real cluster](#running-a-real-cluster).

```bash
docker compose up --build
```

The coordinator is published on port 8000. Seven services come up: three
primaries, three replicas and the coordinator.

**Every physical node keeps its own named volume** — a primary and its replica
sharing a database would be two views of one copy, which is not replication. The
coordinator has no volume: it owns no documents.

### Static validation performed

| Check | Result |
| --- | --- |
| Compose YAML parses | pass |
| Service count | pass — 7 |
| Named volumes | pass — 6, one per physical node |
| Coordinator has no document volume | pass |
| Every environment variable maps to a `Settings` field | pass — all 14 |
| Replica URL separator semantics (`;` across shards, `,` within) | pass |
| Topology satisfies the startup validator | pass |
| Dockerfile `COPY` sources exist | pass |
| `.dockerignore` does not exclude a file the build needs | pass |

### Known Compose limitations

- **The coordinator re-downloads the embedding model on every restart.**
  `HF_HOME` points at `/data/models`, but the coordinator is the one service
  with no `/data` volume — correctly, since it owns no documents. It does embed
  queries, so it does load the model, and that cache does not survive the
  container. Shard nodes cache the model in their own volumes and are unaffected.
- **Every node downloads the model independently on first start**, since each
  volume is separate. That is the same duplicate-inference trade-off replication
  makes elsewhere, applied to the download.
- **The image bundles the semantic extra.** Drop `--extra semantic` from the
  Dockerfile's `uv sync` for a smaller lexical-only image. No image size is
  quoted here because none has been measured.

## Configuration reference

All settings are optional and read from `PYSEARCH_`-prefixed environment
variables, or from a local `.env` file. Copy `.env.example` to `.env` to
override them locally; `.env` is git-ignored.

| Variable | Default | Values |
| --- | --- | --- |
| `PYSEARCH_APP_NAME` | `pysearch` | any string |
| `PYSEARCH_ENVIRONMENT` | `local` | `local`, `test`, `production` |
| `PYSEARCH_LOG_LEVEL` | `INFO` | `DEBUG`…`CRITICAL` (case-insensitive) |
| `PYSEARCH_STORAGE_PATH` | `pysearch.db` | path to the SQLite corpus; parents created |
| `PYSEARCH_NODE_ROLE` | `single` | `single`, `shard`, `coordinator` |
| `PYSEARCH_SHARD_COUNT` | `1` | shards in the cluster; fixed for its lifetime |
| `PYSEARCH_SHARD_ID` | — | required on a shard; in `[0, shard_count)` |
| `PYSEARCH_SHARD_URLS` | — | required on a coordinator; comma-separated, indexed by shard id |
| `PYSEARCH_REPLICA_ROLE` | — | `primary` or `replica`; required on a shard node |
| `PYSEARCH_REPLICA_URLS` | — | replicas: `;` between logical shards, `,` within one |
| `PYSEARCH_PRIMARY_URL` | — | required on a replica, so it can verify and resynchronize |
| `PYSEARCH_NODE_ID` | — | stable name for logs and cluster status |
| `PYSEARCH_CONNECT_TIMEOUT` | `1.0` | seconds |
| `PYSEARCH_REQUEST_TIMEOUT` | `2.0` | seconds, coordinator to node |
| `PYSEARCH_REPLICATION_TIMEOUT` | `2.0` | seconds, primary to replica |
| `PYSEARCH_SEMANTIC_ENABLED` | `false` | load an embedding model and maintain vectors |
| `PYSEARCH_EMBEDDING_MODEL` | `minishlab/potion-base-8M` | embedding model |
| `PYSEARCH_EMBEDDING_MODEL_REVISION` | `bf8b0566…` | pinned revision, not a branch |

There is deliberately **no vector-dimension setting**: it is a property of the
model, discovered when it loads. Configuring it would only create a way for it
to be wrong.

Inconsistent topologies fail at startup rather than in flight: a shard without
an id, a shard id outside the shard count, a coordinator without URLs, a URL
count that disagrees with the shard count, or duplicate shard URLs.

**Run one process per database.** Two processes over the same file would each
hold their own in-memory index and would not see each other's writes.

## Logging

Logs are emitted as one JSON object per line on stdout, using only the standard
library:

```json
{"timestamp": "2026-01-01T12:00:00+00:00", "level": "INFO", "logger": "app.main", "message": "application configured", "environment": "local", "log_level": "INFO"}
```

Fields passed via the standard `extra={...}` argument are merged into the
payload, so node and request identifiers attach without changing the formatter.
Uvicorn's own loggers are routed through the same handler so server and
application logs share one format.

## Repository conventions

```text
app/api/        HTTP layer only — parsing, validation, response models
app/search/     retrieval core — imports no web framework
app/storage/    durable corpus — imports no web framework
app/semantic/   embeddings and vectors — imports no web framework
app/hybrid/     pure rank fusion — imports neither FastAPI nor retrieval machinery
app/cluster/    distributed logic — imports no web framework
app/core/       configuration and logging
```

The rule that keeps this honest: **nothing outside `app/api/` and `app/main.py`
may import FastAPI.** The engine is usable from plain Python, and the test suite
depends on that.

Other conventions:

- Errors are raised as transport-agnostic exceptions in the core and mapped to
  status codes in the API layer. No core module knows what a 503 is.
- Test expectations are written as literals where practical. A test that
  recomputes the implementation cannot detect an error in it.
- Comments explain *why*, not *what*. The code says what it does.
- Commits follow [Conventional Commits](https://www.conventionalcommits.org/)
  with a body explaining the reasoning — see [CONTRIBUTING.md](../CONTRIBUTING.md).
