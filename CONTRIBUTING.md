# Contributing

PySearch is an educational portfolio project. Issues and pull requests are
welcome, but the scope is deliberately narrow: it exists to work through search
and distributed-systems problems directly, so changes that replace hand-written
retrieval code with a library are usually not the right fit.

## Setup

The project uses [uv](https://docs.astral.sh/uv/). Python 3.12 or newer is
required; `.python-version` pins 3.13 as the recommended local interpreter.

```bash
uv sync
```

Semantic and hybrid search need an optional extra, which downloads a pinned
~30 MB embedding model on first use:

```bash
uv sync --extra semantic
```

## Quality gates

All four must pass. CI runs exactly these commands on Python 3.12 and 3.13:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy app
uv run pytest
```

`uv run pytest` never touches the network — the suite runs against a
deterministic fake embedder. The tests that load the real model are behind a
marker and are not run in CI:

```bash
uv run --extra semantic pytest -m semantic_model
```

`mypy` runs in strict mode. New code is expected to type-check without
`# type: ignore`, or to explain the exception in a comment.

## Tests

Tests live in `tests/unit/` (no I/O, no network) and `tests/integration/`
(real HTTP through FastAPI's `TestClient`, real SQLite, multi-node clusters
assembled in-process). New behaviour needs a test; new failure modes need a test
that exercises the failure, not just the happy path.

Expected values are written as literals where practical. A test that recomputes
the implementation cannot detect an error in it.

## Commits

The history follows [Conventional Commits](https://www.conventionalcommits.org/)
with a body explaining *why*, not just what:

```text
feat: add hybrid search with reciprocal rank fusion
fix: reject a replica generation gap instead of applying it
docs: document the write-all replication trade-off
```

The commit history is part of the project's documentation — each commit
corresponds to a coherent milestone. Please keep changes focused rather than
bundling unrelated work.

## Scope

Please open an issue before starting work on anything large. Some things are
absent by design rather than by oversight — approximate nearest-neighbour
search, leader election, automatic promotion, and stop-word filtering are all
discussed in [docs/architecture.md](docs/architecture.md) and the README's
limitations section, with the reasoning for leaving them out.
