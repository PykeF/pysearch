# PySearch

An educational distributed search engine, written from scratch in Python.

> **Current status: Phase 2 — storage and index persistence.**
> PySearch is a working single-node lexical search engine with a durable
> corpus: it analyses text, stores documents in SQLite, builds an inverted
> index in memory, and ranks results with BM25 over an HTTP API. Documents
> survive restarts; the index is rebuilt from them at startup. There is no
> second node and no semantic search — those are the roadmap below, not
> features of this code.

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

**Implemented (Phases 0–2)**

- Unicode-aware text normalization and tokenization, shared by documents and queries
- An in-memory inverted index with posting lists and term frequencies
- Incrementally maintained corpus statistics (`N`, `df`, `tf`, `dl`, `avgdl`)
- BM25 ranking with deterministic ordering and tie-breaking
- Indexing, replacement and deletion of documents, with statistics kept correct
- **A durable document corpus in SQLite, with transactional writes**
- **Startup recovery: the index and document cache are rebuilt from storage**
- **Liveness and readiness endpoints, with an explicit degraded state**
- HTTP APIs for indexing, deletion, search and index statistics
- Structured JSON logging, environment-driven configuration
- 184 tests, strict type checking, linting and formatting gates

**Not implemented** — index snapshots, replication, multiple nodes, sharding,
query caching, phrase or fuzzy search, stemming, embeddings, vector search,
hybrid retrieval, authentication. See the roadmap.

## Architecture

```text
                      +----------------+
                      |    FastAPI     |   app/api/
                      +-------+--------+
                              |
                      +-------v--------+
                      | SearchEngine   |   app/search/engine.py
                      +---+--------+---+
                          |        |
              write/read  |        |  derived search state
                          |        |
                +---------v--+   +-v-----------------+
                | Document   |   | Inverted Index    |
                | Store      |   | + BM25 + analyzer |
                +------+-----+   +-------------------+
                       |
                       v
                +-------------+
                |   SQLite    |   authoritative corpus
                +-------------+
```

The boundary that matters is the one under FastAPI. Nothing in `app/search/` or
`app/storage/` imports FastAPI, so the whole engine is usable from plain Python:

```python
from app.search.document import Document
from app.search.engine import SearchEngine
from app.storage.sqlite_store import SqliteDocumentStore

engine = SearchEngine(SqliteDocumentStore.open("pysearch.db"))
engine.initialize()  # opens the corpus and rebuilds derived state
engine.index_document(Document(document_id="doc-1", text="distributed search"))
engine.search("search", limit=10)
```

`scripts/demo.py` is a runnable version of this, including a restart. The HTTP
layer only parses requests, validates them, calls the engine, and turns the
engine's errors into status codes.

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

## Persistence and recovery

### One authoritative copy, two derived ones

```text
SQLite documents table      authoritative, durable
        |
        v
_documents (cache)          derived, in memory
InvertedIndex               derived, in memory
```

The corpus on disk is the source of truth. Both in-memory structures can be
thrown away at any moment and reconstructed by reading it — which is exactly
what happens at every startup. The document cache is not a second source of
truth: it is a derived copy with a defined direction of reconstruction, and it
exists so the search path does not issue a database lookup per result.

The **inverted index is deliberately not persisted**. Storing it would create a
second durable copy of search state that can disagree with the documents, and
would bring invalidation and versioning problems with it. Rebuilding is
O(corpus) at startup and buys a consistency model with nothing to reconcile.

**No application-level write-ahead log** was written either. SQLite already
provides atomic, journaled, durable commits; a second WAL on top would duplicate
that machinery, add its own recovery path, and create a way for two durability
systems to disagree — for no benefit this phase can point at.

### Storage choice

SQLite through the standard library's `sqlite3`. It gives real ACID durability
and automatic crash recovery for **zero new dependencies**, keeps the corpus in
one portable file, and makes tests durable against a temporary directory.
Alternatives were weighed: a JSON rewrite (rewrites the whole corpus per write),
a JSONL append log (hand-rolling the WAL that SQLite already has), a custom
binary format (large bug surface, little marginal insight), and PostgreSQL (a
driver dependency and a server for a single-process, single-writer store).

SQLite stores documents and nothing else. Analysis, indexing and BM25 remain
PySearch's own work — there is no SQL in the retrieval path.

```sql
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    text TEXT NOT NULL
);
```

`PRAGMA user_version = 1` records the schema version in the database header, so
schema evolution has a hook without a metadata table. `STRICT` tables and upsert
syntax are avoided so the project does not raise its minimum SQLite version for
conveniences it does not need; document validity is the domain model's job.

The default rollback journal is kept rather than WAL, with
`synchronous = FULL`. WAL's advantage is letting readers run alongside a writer,
and the engine's global lock means SQLite never sees concurrent access, so that
advantage is unrealised here. `synchronous = FULL` is what makes "the API
reported success" mean "the write survived".

### Write ordering

Every mutation runs in this order, entirely under the engine lock:

```text
storage transaction -> durable COMMIT -> document cache -> inverted index -> 2xx
```

Storage commits first, so the API can never report success for a write that is
not durable. One transaction per document mutation: creation and replacement are
an existence check plus an insert-or-replace in a single transaction, deletion is
a single statement whose row count reports whether the document existed.

### Crash consistency

Whatever SQLite committed is the truth, and the derived state is rebuilt from it.

| Crash point | State after restart |
| --- | --- |
| Before commit | Rolled back by SQLite; document absent; index rebuilt without it |
| After commit, before the index update | Document present; index rebuilt **with** it |
| Mid-replacement | Atomic: old text or new text, never a blend |
| Mid-deletion | Atomic: present or absent |

In every case the recovered state is consistent with the persisted corpus. This
is the payoff of treating the index as derived: there is no in-process rollback
to get right, because a restart reconstructs correctness.

### Startup recovery

```text
open SQLite -> read documents -> rebuild cache + index -> validate invariants -> ready
```

This runs in a FastAPI lifespan hook, which completes before Uvicorn serves
traffic, so no request can observe a half-built index. Recovery is never lazy.
Failure to open storage or rebuild **aborts startup** rather than serving a
broken service. Startup logs one structured line with the document count and
rebuild duration.

### Degraded state

The one failure a restart cannot fix while the process keeps running is a
derived-state update that raises *after* a durable commit. No compensating
rollback is attempted — the write really happened, and storage is authoritative.
Instead the engine is marked **degraded immediately**, and while degraded it
refuses document mutations, searches and index statistics with `503` rather than
serving results that might disagree with the authoritative corpus. `/ready`
reports `503` with the reason; `/health` still reports `200`, because the process
is alive and restarting it is the orchestrator's call. Reinitialising — normally
a restart — repairs the engine by rebuilding everything from SQLite.

### Failure behaviour

| Situation | Response |
| --- | --- |
| Storage fails during a write | `503 {"detail": "storage unavailable"}`; nothing changed anywhere |
| Storage fails during a delete | `503`; nothing changed anywhere |
| Derived state fails after a commit | `503`; engine degraded; the durable write stands |
| Engine degraded | `503` on mutations, search and `/index/stats`; `/ready` `503` |
| Document not found on delete | `404` |

Database exceptions are wrapped in `StorageError` at the storage boundary and
logged server-side; no SQL, driver text or file path is ever returned to a
client.

### Startup cost

The measured trade-off behind rebuilding instead of snapshotting, from
`scripts/rebuild_benchmark.py` on synthetic 40-word documents. **A local
development measurement on one machine, not a benchmark**, and no claim about
production throughput:

| Documents | Index (s) | Rebuild (s) |
| --- | --- | --- |
| 100 | 0.030 | 0.002 |
| 1,000 | 0.300 | 0.021 |
| 10,000 | 3.363 | 0.398 |

Rebuild is roughly linear and runs an order of magnitude faster than the original
indexing, because indexing pays one fsync per document while recovery is a single
sequential read. Snapshots would only become worth their consistency cost once
this column stops being acceptable.

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

### Health and readiness

`GET /health` returns `{"status": "ok"}` whenever the process is alive. It
deliberately checks nothing else: a liveness probe answers "should this process
be restarted", and failing it because storage blipped would restart a healthy
process.

`GET /ready` returns `{"status": "ready", "detail": "ready"}` once storage is
open and startup recovery has finished, and `503` with
`{"status": "not_ready", ...}` while the engine is degraded. Because a failed
startup aborts the process rather than serving, the 503 case in practice means
the degraded state described above.

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
| Startup recovery | O(total tokens) | every document is re-analysed and re-indexed from storage |

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

The approach is the simplest correct one: **one `threading.Lock` in the engine,
held across every read and every write**. It now spans the storage commit as
well, because the durable write and the derived updates have to be atomic with
respect to each other. Two consequences follow, and both are real costs rather
than details: queries are serialised with each other, and disk latency sits
inside the critical section, so a slow disk stalls searches as well as writes.
What it buys is state that is never observed half-updated — the right trade at
this stage, and an honest one to state rather than claiming a thread-safety
property that was never designed.

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

### Measure startup recovery cost

```bash
uv run python scripts/rebuild_benchmark.py
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
| `PYSEARCH_STORAGE_PATH` | `pysearch.db` | path to the SQLite corpus; parent directories are created |

Invalid values fail loudly at startup rather than being silently ignored.

**Run one process per database.** Two processes over the same file would each
hold their own in-memory index and would not see each other's writes. Sharing a
corpus between nodes is a distributed-systems problem, and it belongs to Phase 3.

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

### Phase 1 — Core information retrieval ✅

Document model, text normalization, tokenization, inverted index, posting lists,
term- and document-frequency statistics, BM25 ranking, indexing and search APIs.
Implemented directly rather than delegated to a search library.

### Phase 2 — Storage and index persistence ✅ *current*

A durable SQLite document corpus behind a narrow `DocumentStore` protocol,
transactional writes, startup recovery that rebuilds all derived state, an
explicit degraded state, and readiness reporting. Index snapshots and an
application-level write-ahead log were both evaluated and deliberately not
built; the reasoning is above.

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
├── search/           the retrieval core — no FastAPI anywhere in here
│   ├── analysis.py   normalization and tokenization
│   ├── document.py   the document model
│   ├── engine.py     lifecycle, write ordering, querying, concurrency
│   ├── errors.py     transport-agnostic errors
│   ├── index.py      inverted index, posting lists, corpus statistics
│   └── ranking.py    BM25
└── storage/          durable corpus — no FastAPI in here either
    ├── base.py       the DocumentStore protocol
    ├── errors.py     StorageError, StorageInitializationError
    └── sqlite_store.py
scripts/
├── demo.py              runnable walkthrough, including a restart
└── rebuild_benchmark.py startup recovery cost by corpus size
tests/unit/           analysis, index, ranking, engine, storage, persistence
tests/integration/    the HTTP API, and restart recovery
```

Modules for distribution will be created when the phase that needs them begins —
not in advance.

## Independence and attribution

PySearch is an independent educational project inspired by general distributed
search-engine architecture. It is not affiliated with Elasticsearch or Elastic,
and does not contain Elasticsearch source code.
