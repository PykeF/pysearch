# PySearch

[![CI](https://github.com/PykeF/pysearch/actions/workflows/ci.yml/badge.svg)](https://github.com/PykeF/pysearch/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A distributed search engine built from scratch in Python: BM25 lexical
retrieval, SQLite durability, deterministic sharding, replication with read
failover, semantic vector retrieval, and RRF hybrid ranking.

The tokenizer, inverted index, corpus statistics, BM25 implementation, vector
index and rank fusion are all written here — **no search or IR library is used
at any point**. Commodity concerns (HTTP, config parsing) use well-understood
libraries so the effort stays on the parts worth understanding.

> **Scope.** This is an educational portfolio project, not production software.
> It runs on one machine, has been verified at a scale of tens of documents and
> benchmarked to ten thousand, and has no authentication. It is not
> Elasticsearch-compatible and implements no distributed consensus.

## Highlights

- **Three independent retrieval modes** over one corpus — BM25, vector
  similarity, and Reciprocal Rank Fusion of the two. Each works on its own.
- **Distributed ranking equals single-node ranking.** Shards score with
  cluster-wide statistics, so a 3-shard cluster returns the same ordering, ties
  and scores as one node holding everything — verified before *and after*
  failover.
- **Synchronous write-all replication** with contiguous generations, automatic
  read failover, and replicas that refuse to serve when they cannot prove they
  are synchronized.
- **Crash-safe by construction.** SQLite is the only source of truth; the index
  and vectors are derived state, rebuilt at every startup, so there is nothing to
  reconcile after a crash.
- **475 tests**, strict `mypy`, `ruff` lint and format gates, CI on Python 3.12
  and 3.13.
- **Honest evaluation.** On the held-out labelled set, hybrid search scored
  **MRR 0.865 against semantic search's 1.000** — it lost. The mechanism is
  understood, documented, and left in place rather than tuned away.

## Architecture

```text
                            Client
                              |
                      +-------v--------+
                      |  Coordinator   |   routes, fans out, merges
                      +---+--------+---+   owns no documents
                          |        |
              logical shard 0      logical shard 1        ...
               /          \          /         \
          primary      replica   primary     replica
             |            |         |           |
        shard-0-p.db  shard-0-r.db  ...       ...     six separate databases
```

Every node — shard, replica, coordinator or standalone — is the same
application; its role decides which routers it exposes. Each shard node is a
complete search engine with its own durable corpus:

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
                +---------v--+   +-v---------------------+
                | Document   |   | InvertedIndex + BM25  |
                | Store      |   | ExactVectorIndex      |
                +------+-----+   +-----------------------+
                       |
                       v
                +-------------+
                |   SQLite    |   authoritative corpus
                +-------------+
```

**Nothing outside `app/api/` imports FastAPI**, so the whole engine is usable
from plain Python:

```python
from app.search.document import Document
from app.search.engine import SearchEngine
from app.storage.sqlite_store import SqliteDocumentStore

engine = SearchEngine(SqliteDocumentStore.open("pysearch.db"))
engine.initialize()  # opens the corpus and rebuilds derived state
engine.index_document(Document(document_id="doc-1", text="distributed search"))
engine.search("search", limit=10)
```

→ **[docs/architecture.md](docs/architecture.md)** for the full design: analysis
pipeline, index structures, BM25, persistence, sharding, both search flows,
fusion, and the decision table behind all of it.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
git clone https://github.com/PykeF/pysearch.git && cd pysearch
```

```bash
uv sync
```

```bash
uv run uvicorn app.main:app
```

Index a document and search for it:

```bash
curl -X PUT localhost:8000/documents/doc-1 -H 'content-type: application/json' -d '{"text": "Distributed systems make search scalable."}'
```

```bash
curl 'localhost:8000/search?q=distributed+search'
```

```json
{
  "query": "distributed search",
  "total": 1,
  "results": [
    {"document_id": "doc-1", "score": 0.5753641449035617, "text": "Distributed systems make search scalable."}
  ]
}
```

Interactive API docs are at `http://127.0.0.1:8000/docs`. No model download is
needed for any of the above.

**Optional — semantic and hybrid search** (downloads a pinned ~30 MB model once):

```bash
uv sync --extra semantic
```

```bash
PYSEARCH_SEMANTIC_ENABLED=true uv run --extra semantic uvicorn app.main:app
```

**See everything at once**, without a server:

```bash
uv run --extra semantic python scripts/demo.py
```

## Search modes

### Lexical — `GET /search`

BM25 over a hand-written inverted index. `k1=1.2`, `b=0.75`, Lucene-variant IDF
so scores never go negative. Ordering is score descending, then `document_id`
ascending, so ties are reproducible.

`total` is the number of **matching** documents. A query that analyses to no
terms returns an empty result set, not an error.

### Semantic — `GET /search/semantic`

Cosine similarity against an exact vector index — one `(N, d)` matrix multiply,
no ANN. Embeddings come from a pinned `model2vec` revision behind a swappable
`Embedder` protocol; it is a static model, so there is no neural network at
inference and no GPU.

`total` is the number of documents **searched**, not matched: a similarity is
defined for every document, so there is no such thing as not matching. Scores
lie in `[-1, 1]` and are **not comparable with BM25 scores**.

### Hybrid — `GET /search/hybrid`

Reciprocal Rank Fusion over the other two:

```text
RRF(d) = sum over the lists containing d of  1 / (k + rank(d))     ranks 1-based
```

Rank-based, because the two score scales cannot be combined. BM25 is unbounded
and corpus-dependent — the same document scored 0.3696 and then 0.4627 for the
same query after one deletion changed `df` — while cosine sits in `[-1, 1]`. A
weighted sum would not be a weighting; it would be an accident.

`score` is a **fusion score**: not relevance, not similarity, not a probability.
`explain=true` adds the underlying scores; the two ranks are always present and
together reconstruct the score exactly.

**Candidate depth matters.** Each path is asked for `min(5 × limit, 100)`
candidates, and fusion runs over that union. Hybrid results are therefore the
exact RRF of the *retrieved* lists, **not necessarily of the whole corpus** —
a real limitation, asserted by a test rather than only documented.

## Distributed system

Documents are partitioned by `blake2b(document_id) % shard_count` — stable
across processes and restarts, unlike Python's randomized `hash()`. Writes go to
exactly one owning shard and are never broadcast or rerouted.

**Lexical search takes two rounds** because BM25 needs corpus-wide statistics:

```text
round 1   ask every shard for N, summed length, and df of the query's terms
round 2   ask every shard for its local top-k, scored with those statistics
          -> merge, sort, truncate
```

Sharing `idf` is what makes scores from different shards comparable, which in
turn makes each shard's local top-k sufficient for an exact global top-k.

**Semantic search takes one round.** A cosine similarity depends on nothing but
the two vectors, so comparability is a property of the metric rather than
something the coordinator has to arrange. The coordinator embeds the query once
so every shard scores an identical vector.

Both rounds of a search, and every routed write, happen under one coordinator
lock — so no write can land between collecting statistics and scoring against
them. The cost is deliberate and real: **coordinator operations serialize**.

**Any shard failing fails the whole query with 503.** Partial results under
cluster-wide statistics would be both incomplete and mis-scored.

## Reliability

**A 2xx means the document is durable on the primary and on every configured
replica of its logical shard.** Nothing weaker.

| Failure | Search | Writes |
| --- | --- | --- |
| Primary down, replica READY | **works**, identical results | **fail** (503) |
| Replica down | works | **fail** (503) — write-all |
| Replica out of sync | works via primary | work |
| Every copy of a shard down | **503** | fail |

Each logical shard carries a **contiguous mutation sequence number**, persisted
in the same transaction as the mutation. A replica applies exactly `local + 1`,
treats anything lower as an already-applied retry, and **refuses a gap** —
because a replica that applied generation 6 while missing 5 would hold a corpus
that never existed, yet would look perfectly synchronized.

**Reads fail over automatically; writes do not.** There is no leader election and
no automatic promotion: safe promotion needs catch-up proof and fencing, which
is consensus, and a half-built consensus is worse than none. Roles are static
and enforced structurally, so two writable primaries cannot arise.

Verified on a real 7-process cluster: failover produced byte-identical results
for all three retrieval modes, an out-of-sync replica correctly refused to
serve, and a restarted replica resynchronized and rejoined.

→ **[docs/consistency.md](docs/consistency.md)** for the guarantees stated
concretely, including the ambiguous-failure window and what is *not* guaranteed.

## Evaluation

A small synthetic corpus written for this project: **67 documents, 32 labelled
queries**, split into 12 development queries (which chose the RRF parameters)
and 20 held-out evaluation queries (measured once, parameters frozen).

| Mode | Recall@5 | Recall@10 | MRR |
| --- | --- | --- | --- |
| BM25 | 0.725 | 0.725 | 0.732 |
| semantic | 0.925 | 0.933 | **1.000** |
| hybrid | 0.858 | 0.933 | 0.865 |

**Hybrid did not beat semantic alone.** It matched the best input on every
lexical, mixed and distractor query — and recovered one BM25 failure — but lost
on four of six paraphrases. The mechanism:

```text
query: "searching by meaning rather than keywords"
BM25      cook-4 ("season in layers RATHER THAN all at the end")  rank 1
semantic  ir-3   (the correct answer)                             rank 1
hybrid    cook-4 first; ir-3 pushed down
```

BM25 is not merely unhelpful there — it is **confidently wrong**, matching on
`rather` and `than`. RRF weighs both retrievers equally, so wrong-but-confident
lexical hits outrank a correct semantic one.

> **BM25 finding nothing is harmless to fusion. BM25 finding the wrong thing is
> what hurts.**

The root cause is a deliberate earlier decision: there is no stop-word filtering.
Adding one to make hybrid win would be tuning to the evaluation set, which is
exactly what the dev/eval split exists to prevent. The result is documented, not
patched.

→ **[docs/evaluation.md](docs/evaluation.md)** for the dataset, per-query
results, parameter sweep, cost benchmarks, and the full limitations list.

## API

| Method | Path | Notes |
| --- | --- | --- |
| `PUT` | `/documents/{id}` | `201` created, `200` replaced, `400` blank id |
| `DELETE` | `/documents/{id}` | `204`, or `404` if absent |
| `GET` | `/search` | BM25. `q`, `limit` (1–100, default 10) |
| `GET` | `/search/semantic` | Cosine. `503` if semantic is disabled |
| `GET` | `/search/hybrid` | RRF. `explain=true` adds underlying scores; `503` if semantic is disabled |
| `GET` | `/index/stats` | `document_count`, `unique_term_count`, `average_document_length` |
| `GET` | `/health` | Liveness. Always `200` while the process is alive |
| `GET` | `/ready` | Readiness. `503` while degraded, or if any shard is unready |
| `GET` | `/cluster/status` | Coordinator only. Per-shard copies, roles, states, generations |
| `*` | `/internal/*` | Node-to-node only. **Unauthenticated** — see [SECURITY.md](SECURITY.md) |

Availability by role:

- **`single`** (default) serves the full public API; `/cluster/status` is absent.
- **`coordinator`** serves the full public API plus `/cluster/status`, and owns
  no documents.
- **`shard`** deliberately exposes **no public `/search`** — querying one shard
  directly would return silently partial, mis-scored results.
- `/search/semantic` and `/search/hybrid` require `PYSEARCH_SEMANTIC_ENABLED`.

Full request/response schemas are served at `/docs` by FastAPI.

## Development

```bash
uv sync
```

```bash
uv run pytest
```

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy app
```

`uv run pytest` runs 468 tests and never touches the network — the suite uses a
deterministic fake embedder. The 7 real-model tests are behind a marker:

```bash
uv run --extra semantic pytest -m semantic_model
```

Run a real 7-process cluster locally, without containers:

```bash
uv run python scripts/run_cluster.py --replication-factor 2
```

Kill a primary and search again: identical results from the replica, while
writes to that shard return 503.

→ **[docs/development.md](docs/development.md)** for every command, the
configuration reference, the cluster walkthrough, Docker status, and repository
conventions. Contributions: [CONTRIBUTING.md](CONTRIBUTING.md).

## Limitations

Each is a real consequence of a deliberate choice, not an oversight.

**Distributed system**

- **The coordinator is a single point of failure.** Replication improved shard
  availability and nothing else.
- **No automatic write failover.** A lost primary means no writes to that shard
  until an operator returns it.
- **Writes need every copy.** One replica down stops writes to that logical
  shard. Read availability was bought with write availability.
- **A failed write may still be durable on the primary**, so an error does not
  prove the write did not land. `PUT` is idempotent; retry is safe.
- **A replica cannot recover while its primary is down**, and resynchronization
  transfers the whole logical shard — there is no incremental catch-up log.
- **Shard count and replication factor are fixed** for a cluster's life; changing
  the shard count moves nearly every document.
- **No consensus, quorum, election, dynamic membership, or rebalancing.**

**Retrieval**

- **No stop-word filtering, stemming or lemmatization** — the direct cause of the
  hybrid failures above.
- **Hybrid is candidate-limited**: exact RRF of the retrieved lists, not of the
  corpus.
- **Vector search is exact**, so query cost grows linearly with the corpus.
- **Vectors are not persisted**, so every start re-embeds everything, and every
  replica embeds independently.
- **Embedding quality is that of a static model** — meaningfully below a real
  transformer.
- **No phrase, fuzzy or wildcard search; no query expansion or reranking.**

**Evaluation and operations**

- **The evaluation is synthetic and tiny** (67 documents, 32 queries) and cannot
  support general ranking-quality claims.
- **Internal endpoints are unauthenticated** and assume a trusted network. No
  TLS, no rate limiting, no audit log.
- **Docker is configured but not runtime-verified** — Docker was unavailable
  throughout development. Static validation only; the multi-process script is
  the verified distributed path.
- **All performance numbers are single-machine local measurements.**

## At larger scale

What would have to change — design awareness, not implemented work:

| Today | At 1M–100M documents |
| --- | --- |
| Exact vector search, O(N·d) | ANN (HNSW/IVF), with exact search kept as the evaluation baseline |
| Fixed modulo sharding | Virtual nodes and migration-aware routing |
| One SQLite file per shard | A storage engine with segment merging and on-disk postings |
| Static topology from config | Service discovery and membership |
| Single coordinator | Replicated stateless frontends behind a load balancer |
| Manual primary recovery | Consensus-backed leader election with fencing |
| Rebuild all derived state at startup | Persisted segments with incremental recovery |
| One lock per engine | Per-segment locking or copy-on-write read snapshots |

## Project history

The commit history is the engineering story — one commit per milestone, each
with its reasoning in the message.

| Phase | What it added |
| --- | --- |
| **0 — Foundation** | FastAPI bootstrap, config, structured logging, uv/pytest/ruff/mypy toolchain |
| **1 — Lexical IR** | Analysis pipeline, inverted index, corpus statistics, BM25, search API |
| **2 — Persistence** | SQLite source of truth, transactional writes, startup recovery, degraded state |
| **3 — Distribution** | Sharding, coordinator, two-round distributed BM25, global top-k, fail-whole policy |
| **4 — Replication** | Logical shards vs physical copies, write-all replication, generations, read failover |
| **5 — Semantic** | Pinned embeddings behind a protocol, exact vector index, one-round distributed semantic search |
| **6 — Hybrid** | Reciprocal Rank Fusion, candidate depth, explainable ranks, held-out three-mode evaluation |
| **7 — Release** | CI, documentation, licensing, repository hygiene, reproducible demos |

## Technology

Python 3.12+, FastAPI, Pydantic / pydantic-settings, Uvicorn, httpx, NumPy,
SQLite (standard library), `model2vec` (optional). Tooling: uv, pytest, ruff,
mypy.

**The information-retrieval core uses only the standard library.** NumPy appears
only in the vector index; no dependency was added for analysis, indexing, BM25
or fusion.

## License

[MIT](LICENSE) — Copyright (c) 2026 Yirun Fu.

## Independence and attribution

PySearch is an independent educational project inspired by general distributed
search-engine architecture. It is not affiliated with Elasticsearch or Elastic,
and does not contain Elasticsearch source code.
