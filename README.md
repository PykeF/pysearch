# PySearch

An educational distributed search engine, written from scratch in Python.

> **Current status: Phase 1 — core information retrieval.**
> PySearch is a working single-node lexical search engine: it analyses text,
> builds an inverted index in memory, and ranks results with BM25 over an HTTP
> API. Everything is **in memory and lost when the process exits**. There is no
> persistence, no second node, and no semantic search — those are the roadmap
> below, not features of this code.

## Motivation

Search engines sit at the intersection of most of the interesting problems in
backend engineering: data structures, ranking algorithms, storage, concurrency,
networking, and distributed-systems failure modes. Reaching for an off-the-shelf
engine hides all of it behind an API call.

PySearch exists to work through those problems directly. The tokenizer, the
inverted index, the corpus statistics and the BM25 implementation here are all
written from scratch — no search or IR library is used at any point. Commodity
concerns (the HTTP layer, config parsing) use well-understood libraries so the
effort stays on the parts worth understanding.

## What exists today

**Implemented (Phase 1)**

- Unicode-aware text normalization and tokenization, shared by documents and queries
- An in-memory inverted index with posting lists and term frequencies
- Incrementally maintained corpus statistics (`N`, `df`, `tf`, `dl`, `avgdl`)
- BM25 ranking with deterministic ordering and tie-breaking
- Indexing, replacement and deletion of documents, with statistics kept correct
- HTTP APIs for indexing, deletion, search and index statistics
- Structured JSON logging, environment-driven configuration
- 120 tests, strict type checking, linting and formatting gates

**Not implemented** — persistence, recovery, replication, multiple nodes,
sharding, caching, phrase or fuzzy search, stemming, embeddings, vector search,
hybrid retrieval, authentication. See the roadmap.

## Architecture

```text
                  +----------------+
                  |    FastAPI     |   app/api/
                  +-------+--------+
                          |
                  +-------v--------+
                  | SearchEngine   |   app/search/engine.py
                  +-------+--------+
                          |
             +------------+-------------+
             |                          |
      +------v-------+           +------v------+
      | Text Analyzer|           | BM25 Ranker |
      +------+-------+           +------+------+
             |                          |
             +------------+-------------+
                          |
                  +-------v--------+
                  | Inverted Index |
                  +----------------+
```

The boundary that matters is the one under FastAPI. Nothing in `app/search/`
imports FastAPI, so the whole engine is usable from plain Python:

```python
from app.search.document import Document
from app.search.engine import SearchEngine

engine = SearchEngine()
engine.index_document(Document(document_id="doc-1", text="distributed search"))
engine.search("search", limit=10)
```

`scripts/demo.py` is a runnable version of this. The HTTP layer only parses
requests, validates them, calls the engine, and turns the engine's errors into
status codes.

## How search works

### Normalization and tokenization

One pipeline, `analyze()`, runs over document text at index time and over query
text at search time. Sharing a single entry point is what guarantees the two
cannot drift apart — a query term normalised differently from the document term
it should match would simply never match it.

1. **Unicode NFKC normalization**, so compatibility variants of a character
   (full-width forms, ligatures) compare equal.
2. **Case folding** via `str.casefold`, which is the Unicode-aware operation
   `str.lower` is not: it maps `ß` to `ss`.
3. **Tokenization** into maximal runs of alphanumeric characters. Punctuation,
   symbols and whitespace separate tokens and are discarded. Repeated terms are
   preserved, because term frequency is computed from this sequence.

```text
"Distributed, SEARCH!"  ->  ["distributed", "search"]
"search search index"   ->  ["search", "search", "index"]
"!!! ???"               ->  []
```

Two consequences are limitations rather than decisions, and are tested as such:
an apostrophe separates tokens, so `don't` becomes `don` and `t`; and there is
no segmentation for scripts that do not delimit words with spaces, so a run of
CJK characters becomes one token. Stemming, lemmatization and stop-word removal
are deliberately absent — they change what "the same word" means, and deserve
to be introduced with evaluation behind them.

### Inverted index and posting lists

The index maps terms to the documents containing them, which is what lets a
query touch only relevant documents instead of scanning the corpus:

```text
"search" -> {"doc-1": 2, "doc-3": 1}
"index"  -> {"doc-2": 1}
```

Four structures are maintained, all updated incrementally:

| Structure | Shape | Purpose |
| --- | --- | --- |
| `_postings` | `term -> document_id -> tf` | posting lists and term frequencies |
| `_document_terms` | `document_id -> unique terms` | makes deletion cheap |
| `_document_lengths` | `document_id -> tokens` | `dl` for length normalization |
| `_total_length` | `int` | running sum, so `avgdl` is O(1) |

A posting list is a mapping rather than a sorted list. That makes term-frequency
lookup, insertion and single-document removal all O(1), and makes document
frequency `len(postings[term])`. The price is giving up what an ordered posting
list buys — skip pointers, delta compression, merge joins — none of which
matters while the index lives in a dict in RAM, and all of which belongs to the
phase that designs an on-disk format. A `Posting` dataclass provides the
readable external view; ranking iterates the mapping directly rather than
allocating one object per posting per query.

`_document_terms` duplicates information derivable from `_postings`. That is
deliberate: without it, deleting one document means scanning the entire
vocabulary. With it, deletion costs O(unique terms in that document).

### BM25

```text
                              tf(q,D) * (k1 + 1)
    score(D,Q) = sum  idf(q) * ---------------------------------------
                 q in Q        tf(q,D) + k1 * (1 - b + b * dl(D)/avgdl)

    idf(q) = ln(1 + (N - df(q) + 0.5) / (df(q) + 0.5))
```

Defaults are **`k1 = 1.2`** and **`b = 0.75`**, held in a `BM25Params`
dataclass rather than written into the scoring expression.

The `tf` fraction saturates: the tenth occurrence of a term adds far less than
the second, and `k1` controls how fast that happens. The `dl/avgdl` factor
penalises long documents — a term appearing once in a 20-word document is
stronger evidence than the same term once in a 2000-word one — and `b` controls
how much, from `b=0` (ignore length) to `b=1` (fully normalise).

The IDF is the Lucene variant rather than the classic
`ln((N - df + 0.5) / (df + 0.5))`. The classic form goes negative once a term
appears in more than half the corpus, which lets a common term push a document
*below* one that does not contain it at all. Adding one inside the logarithm
keeps IDF positive for every `df`.

**Ordering is deterministic**: by descending score, then ascending
`document_id`. The second key is what makes ties reproducible rather than
dependent on dictionary iteration order. A term repeated in a query is scored
once per occurrence, so repeating it emphasises it.

## API

### Index or replace a document

```bash
curl -X PUT localhost:8000/documents/doc-1 -H 'content-type: application/json' -d '{"text": "Distributed systems make search scalable."}'
```

`201 Created` for a new document, `200 OK` when an existing one is replaced,
`400 Bad Request` if the identifier is blank. Replacement is a full reindex:
terms no longer present disappear from the index, and corpus statistics are
updated. Empty text is valid — the document is indexed with zero tokens, counts
towards `N`, and can never match a query.

### Search

```bash
curl 'localhost:8000/search?q=distributed%20search&limit=5'
```

```json
{
  "query": "distributed search",
  "total": 2,
  "results": [
    {"document_id": "doc-1", "score": 1.2345, "text": "Distributed systems make search scalable."}
  ]
}
```

`total` is the number of matching documents before `limit` is applied. `limit`
defaults to 10 and must be between 1 and 100; anything else is `422`. A query
that analyses to no terms — empty, or only punctuation — returns an empty
result set rather than an error, because matching nothing is not a client
mistake.

### Delete a document

```bash
curl -X DELETE localhost:8000/documents/doc-1
```

`204 No Content` on success, `404 Not Found` if no such document is indexed.
Deletion removes the document's postings, drops any term whose posting list
becomes empty, and updates `df`, document lengths and `avgdl`.

### Index statistics

```bash
curl localhost:8000/index/stats
```

```json
{"document_count": 2, "unique_term_count": 7, "average_document_length": 4.0}
```

These are exactly the quantities BM25 scores with, which is why they are worth
exposing: they make ranking behaviour and the effect of updates and deletions
observable. Posting lists and other internal structures are not exposed.

### Health

`GET /health` returns `{"status": "ok"}`. It is a liveness check and reports
nothing about the index.

## Complexity and memory

Let `T` be the tokens in a document, `U` its unique terms, `V` the vocabulary,
`N` the corpus size, and `P(q)` the number of documents containing term `q`.

| Operation | Cost | Why |
| --- | --- | --- |
| Analysis | O(characters) | one pass over the text |
| Indexing a document | O(T) | counting terms is one pass; each of `U` posting insertions is an amortised O(1) dict write |
| Replacing a document | O(T + T′) | a removal followed by an insertion |
| Deleting a document | O(U) | the forward term set gives the exact terms to touch, avoiding an O(V) scan |
| Query | O(Σ P(q) + M log M) | only the query terms' posting lists are visited; `M` matching documents are then sorted |
| Corpus statistics | O(1) | maintained incrementally, never recomputed |

The query cost is the whole point of an inverted index: it is proportional to
how many documents contain the query terms, not to `N`. A term absent from the
corpus costs a single failed dict lookup.

Sorting all `M` matches is O(M log M) where a bounded heap would be
O(M log limit). At Phase 1 scale the difference is not worth the loss of
clarity, and Phase 3 will need a different top-k path anyway to merge results
across nodes.

Memory is dominated by the posting entries: one `document_id` key and one
integer per (term, document) pair, plus the unique-term set and length per
document, plus the original text of every document in the engine's store. All
of it is Python objects in a dict, so the constant factor is large — measuring
and reducing it is a later concern.

## Concurrency

FastAPI runs synchronous path operations in a thread pool, so requests really do
execute concurrently against the index. Two concurrent writes would interleave
their read-modify-write of the running token total and lose an update; a query
iterating a posting list while a delete mutates it would fail outright.

Phase 1 takes the simplest correct approach: **one `threading.Lock` in the
engine, held across every read and every write**. Queries are therefore
serialised with each other. That costs read throughput and buys an index that is
never observed half-updated — the right trade at this stage, and an honest one
to state rather than claiming a thread-safety property that was never designed.

The alternative — `async` handlers relying on the event loop for atomicity —
needs no lock, but the invariant disappears silently the moment anyone adds an
`await` inside a critical section, and CPU-bound work would block the loop.

The index also has an explicit `validate()` invariant check (total length equals
the sum of document lengths, every posting references a live document, no
posting list is empty, the forward and inverted mappings agree, and each
document's length equals its summed term frequencies). It is a testing and
debugging aid — the concurrency tests assert it after running bounded concurrent
traffic — and is deliberately never called on the request path.

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

```bash
uv sync
```

### Run the application

```bash
uv run uvicorn app.main:app --reload
```

Interactive API documentation is served at `http://127.0.0.1:8000/docs`.

### Run the demo

```bash
uv run python scripts/demo.py
```

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

## Technology stack

**In use now** — Python 3.12+, FastAPI, Pydantic / pydantic-settings, Uvicorn,
pytest, ruff, mypy, uv. **The information-retrieval core uses only the standard
library**; no dependency was added for Phase 1.

**To be evaluated in later phases**, and not dependencies today: Docker and
Docker Compose (multi-node local deployment), a persistence technology, Redis
for caching, gRPC for inter-node communication, NumPy, embedding models, and an
ANN library such as FAISS or an alternative. Each will be chosen when a phase
creates a concrete need for it.

## Roadmap

### Phase 0 — Engineering foundation ✅

FastAPI application bootstrap, environment-driven configuration, structured
logging, testing infrastructure, and repository engineering standards.

### Phase 1 — Core information retrieval ✅ *current*

Document model, text normalization, tokenization, inverted index, posting lists,
term- and document-frequency statistics, BM25 ranking, indexing and search APIs.
Implemented directly rather than delegated to a search library.

### Phase 2 — Storage and index persistence

Persistent document and index storage, write-ahead logging where appropriate,
snapshots, recovery, and a storage abstraction. The storage technology will be
chosen from the requirements that emerge there.

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
├── api/              HTTP layer — parsing, validation, response models
│   ├── dependencies.py
│   ├── documents.py  PUT / DELETE /documents/{id}
│   ├── health.py     GET /health
│   ├── index.py      GET /index/stats
│   └── search.py     GET /search
├── core/             configuration and logging
└── search/           the retrieval core — no FastAPI anywhere in here
    ├── analysis.py   normalization and tokenization
    ├── document.py   the document model
    ├── engine.py     indexing, deletion, querying, concurrency
    ├── errors.py     transport-agnostic errors
    ├── index.py      inverted index, posting lists, corpus statistics
    └── ranking.py    BM25
scripts/demo.py       runnable walkthrough of the core
tests/unit/           analysis, index, ranking, engine, config, logging
tests/integration/    the HTTP API
```

Modules for storage and distribution will be created when the phase that needs
them begins — not in advance.

## Independence and attribution

PySearch is an independent educational project inspired by general distributed
search-engine architecture. It is not affiliated with Elasticsearch or Elastic,
and does not contain Elasticsearch source code.
