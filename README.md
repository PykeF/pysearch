# PySearch

An educational distributed search engine, written from scratch in Python.

> **Current status: Phase 5 — semantic and vector search.**
> PySearch is a working replicated distributed lexical search engine. Documents
> are partitioned across logical shards by a stable hash; each logical shard has
> a primary and a replica, each with its own durable SQLite corpus and in-memory
> index; and a coordinator fans queries out in parallel, failing over to a
> replica when a primary is lost. Distributed ranking is equivalent to a single
> node holding the whole corpus, before and after failover.
>
> It now has a **second, independent retrieval path**: documents are embedded
> and searched by vector similarity through `/search/semantic`, while `/search`
> remains exactly the BM25 it always was. The two are never combined — that is
> the next phase's question.
>
> **Reads fail over automatically; writes do not.** There is no leader election
> and no automatic promotion, so losing a primary means that logical shard stops
> accepting writes until it returns. Vector search is **exact, not approximate**,
> and there is no hybrid ranking. Those are the roadmap below, not features of
> this code.

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

**Implemented (Phases 0–5)**

- Unicode-aware text normalization and tokenization, shared by documents and queries
- An in-memory inverted index with posting lists and term frequencies
- Incrementally maintained corpus statistics (`N`, `df`, `tf`, `dl`, `avgdl`)
- BM25 ranking with deterministic ordering and tie-breaking
- Indexing, replacement and deletion of documents, with statistics kept correct
- A durable document corpus in SQLite, with transactional writes
- Startup recovery: the index and document cache are rebuilt from storage
- Liveness and readiness endpoints, with an explicit degraded state
- Deterministic sharding across multiple nodes, with a coordinator
- Parallel query fan-out and a deterministic global top-k merge
- Cluster-wide BM25 statistics, so distributed ranking matches single-node
- **Synchronous write-all replication with a contiguous mutation sequence**
- **Automatic read failover to a verified-synchronized replica**
- **Replica resynchronization, with unverified copies refusing to serve**
- Cluster status separating search availability from write availability
- **Document and query embeddings behind a swappable model boundary**
- **An exact vector index with cosine similarity, written here**
- **Distributed semantic search in a single fan-out round, with failover**
- **A labelled BM25-versus-semantic evaluation, reported separately**
- **Docker Compose topology for reproducible multi-node local deployment**
- HTTP APIs for indexing, deletion, search and index statistics
- Structured JSON logging carrying node role and shard id
- 415 tests, strict type checking, linting and formatting gates

**Not implemented** — hybrid retrieval, rank fusion, reranking, approximate
nearest-neighbour search, persisted vectors, leader election, consensus,
automatic primary promotion, writable failover, dynamic membership,
rebalancing, coordinator replication, index snapshots, query caching, phrase or
fuzzy search, stemming, authentication. See the roadmap.

## Architecture

A cluster is a coordinator plus a fixed number of shard nodes:

```text
                            Client
                              |
                      +-------v--------+
                      |  Coordinator   |   routes, fans out, merges
                      +---+----+-----+-+   owns no documents
                          |    |     |
              +-----------+    |     +-----------+
              v                v                 v
          +--------+       +--------+        +--------+
          | Shard 0|       | Shard 1|        | Shard 2|
          +---+----+       +---+----+        +---+----+
              |                |                 |
          shard-0.db       shard-1.db        shard-2.db
```

Each shard is a complete Phase 2 node — its own durable corpus, its own document
cache, its own inverted index — and reuses that code unchanged:

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

## Semantic search

### Two independent paths

```text
                        Document (SQLite = authoritative)
                          /                        \
                    analyze()                   embed()
                        |                          |
                 InvertedIndex               ExactVectorIndex
                        |                          |
                      BM25                 cosine similarity
                        |                          |
                  GET /search             GET /search/semantic
```

Both are derived from the same authoritative documents, and **neither depends on
the other**. `/search` is the BM25 it has always been; `/search/semantic` is a
separate endpoint with its own score scale. They are deliberately not combined:
how to fuse two rankings is a real question with real trade-offs, and answering
it by quietly averaging two incomparable numbers would be the wrong answer.

Semantic search is **off by default**. Enabling it is `PYSEARCH_SEMANTIC_ENABLED`,
which keeps lexical-only deployments — and the whole existing test suite — free
of a model and its startup cost.

### The model

| | |
| --- | --- |
| Implementation | `model2vec` |
| Model | `minishlab/potion-base-8M` |
| Revision | `bf8b0566…` — pinned, not a branch |
| Dimension | 256 |
| Normalization | L2, applied in one place |
| Similarity | cosine, computed as a dot product |

A **static** embedding model: every token has a fixed vector and a document
embedding is the mean of its token vectors, so there is no neural network at
inference time. That buys a ~10 MB dependency instead of the ~1 GB a torch-based
stack costs, CPU speed, and behaviour deterministic enough to test. It is
measurably weaker than a real transformer, and the `Embedder` protocol exists so
that swapping one in is a small change if the evaluation ever says the gigabyte
is worth it.

The model is downloaded once on first use. **`uv run pytest` never downloads
anything**: the suite runs against a deterministic fake embedder, and the tests
that use the real model are behind a marker.

### Semantic identity

Vectors from different models measure different spaces, so comparing them
produces numbers that look like similarities and mean nothing. Five fields must
match exactly before two copies are treated as interchangeable:

```text
implementation : model_id @ revision / dimension / normalization
```

The revision is part of it because a repository can move while keeping its name.
The coordinator sends this identity with every query vector and a shard **refuses
with 409** if it disagrees — checked per request, at no extra round trip.

What is deliberately **not** claimed is bit-for-bit identical vectors. Library
versions and floating-point reduction order can move the last bits. The
guarantee is that copies sharing an identity produce equivalent embeddings, and
therefore **identical ordering and identical tie-breaking**, with scores equal to
within a strict tolerance — which is exactly what the tests assert.

### Exact search, not approximate

The vector index compares the query against every stored vector: one `(N, d)`
matrix multiply, O(N·d), plus O(N log N) to order the results. That is a
deliberate starting point, not a shortcut.

- Approximate search can only be *evaluated* against exact search, so exact has
  to exist first regardless.
- Delete and replace have to be exact, because replication treats a copy's
  derived state as an exact function of its documents. Approximate indexes reach
  that through tombstones and periodic rebuilds, which would put approximation
  underneath a correctness invariant.
- FAISS `IndexFlatIP` would make the same multiply faster without changing its
  complexity, in exchange for a native dependency.

**FAISS and ANN were evaluated and declined.** The measurements that would
overturn that are in the table below, and `scripts/semantic_benchmark.py`
reproduces them.

### Vectors are derived, and not persisted

Documents remain the only authoritative corpus. Vectors are rebuilt by
re-embedding at startup, so there is no second durable derived state and no
model-version invalidation problem: if the configured model changes, every
vector is simply rebuilt from the documents.

Embedding is a **precondition of every mutation**, on both the local and the
replicated write path:

```text
analyze + embed  ->  durable commit (generation N+1)  ->  cache, index, vectors
```

Computing the vector *before* the durable write is what makes an embedding
failure harmless: nothing is committed, no generation is consumed, and the node
stays healthy. A replica follows the same rule, so it can never look
synchronized while its semantic state is missing — and it checks the incoming
generation first, so a redelivery or a gap costs no model inference.

Replacement replaces the vector, deletion removes it, and both are asserted
directly rather than inferred from ranking.

### Replication

Replication ships **documents only**; every copy derives its own vectors. That
keeps one rule instead of two, because resynchronization already transfers
documents and rebuilds derived state — so the replication path and the recovery
path do the same thing. The cost is duplicate inference, cheap with this model
and worth revisiting with an expensive one.

### Distributed semantic search — one round, not two

```text
query -> coordinator embeds once -> one fan-out -> local top-k -> merge
```

This is the architectural contrast worth stating. BM25 needs a **statistics
round** because `idf` depends on corpus-wide `N` and `df`, so scores from
different shards are not comparable until they share those numbers. A cosine
similarity depends on **nothing but the two vectors** — so comparability is a
property of the metric rather than something the coordinator has to arrange.
There is no statistics round, and with no second round there is no pinning
problem either.

The coordinator embeds the query **once** so every shard scores against an
identical vector. Exactly one serving copy per logical shard answers, so
replicas are never double-counted. Local top-k is sufficient for global top-k by
the same argument as the lexical path, and the merge uses the same rule: score
descending, then document id ascending.

Failover, readiness and the fail-whole policy are unchanged: a READY replica can
serve, an out-of-sync one cannot, and losing every copy of a logical shard
returns **503** rather than a partial answer.

### Lexical versus semantic, measured

`scripts/evaluate_retrieval.py` runs 13 labelled queries against a 23-document
synthetic corpus and reports the two modes separately. **A local development
measurement, not a benchmark** — the corpus is far too small to say anything
general.

| | BM25 | semantic |
| --- | --- | --- |
| Recall@5 | 0.58 | 1.00 |
| MRR | 0.57 | 1.00 |

The averages are less interesting than where they come from. BM25 scored **0.00
on four of the six paraphrase queries** — it returned no documents at all for
"car maintenance", because the word "car" appears nowhere in the corpus — and
1.00 on every query that shares vocabulary with its answer.

The honest caveat: this set contains **no query where BM25 beats semantic**,
including the identifier cases where I expected it to, since subword
tokenization captures exact tokens better than I predicted. That is a limitation
of the evaluation, not evidence that lexical retrieval is redundant.

What the measurement does show is that the two answer *different questions*.
BM25 has a notion of not matching: it returns 0 candidates for a paraphrase and
1 for an identifier. A similarity is defined for every document, so semantic
search always ranks the entire corpus and never says "no". Live, through the
coordinator:

```text
query: "car maintenance"
  GET /search           ->  total=0   (nothing contains "car")
  GET /search/semantic  ->  doc-1 0.6011  "automobile repair and servicing..."
```

That difference — one mode that can say nothing matched, one that always ranks
everything — is the actual motivation for the next phase.

### What it costs

Local development measurements on one machine with synthetic 40-word documents,
from `scripts/semantic_benchmark.py`. **Not a benchmark.**

| Documents | Lexical rebuild | Semantic rebuild | Vector search | Vector memory |
| --- | --- | --- | --- | --- |
| 100 | 0.002 s | 0.003 s | 0.032 ms | 0.1 MB |
| 1,000 | 0.026 s | 0.034 s | 0.257 ms | 1.0 MB |
| 5,000 | 0.151 s | 0.171 s | 1.980 ms | 5.1 MB |

Model load 0.197 s; embedding ~20,800 documents/s batched; query embedding
0.050 ms. Vector memory is `N × d × 4` bytes for float32, excluding Python
overhead.

Two decisions were made pending these numbers. **Re-embedding at startup costs
about the same as the lexical rebuild**, which is why vectors are not persisted.
**Search grows linearly** — 2 ms at 5,000 documents implies tens of milliseconds
at a hundred thousand — and that column is what would eventually justify an
approximate index.

## Replication and availability

### Logical shards versus physical nodes

```text
                            Client
                              |
                      +-------v--------+
                      |  Coordinator   |   topology only; owns no documents
                      +---+--------+---+
                          |        |
              logical shard 0      logical shard 1        ...
               /          \          /         \
          primary      replica   primary     replica
             |            |         |           |
        shard-0-p.db  shard-0-r.db  ...       ...        six separate databases
```

A **logical shard** is a deterministic subset of documents. A **physical node**
is one process with one database. Routing keys on the logical shard, so adding
or removing replicas moves no documents at all — `replication_factor` never
enters the hash.

### Write-all replication

**A 2xx means the document is durable on the primary *and* on every configured
replica of its logical shard.** Nothing weaker.

```text
coordinator -> primary: durable commit (generation N+1)
                  -> replica: apply(generation N+1), durable commit
               <- acknowledge
```

Primary-first ordering is deliberate: it guarantees
`generation(primary) >= generation(replica)` always, so the primary is the most
advanced copy, is always the correct recovery source, and there is never a
question about which copy is newer.

Synchronous rather than asynchronous, for a specific reason: a lagging replica
holds a *different* corpus, so global statistics read from it would be wrong and
failover would silently change the ranking. Write-all makes every READY copy
interchangeable, which is exactly what makes read failover safe.

The price is stated rather than hidden: **if any replica is unavailable, writes
to that logical shard fail.**

| Situation | Response | State |
| --- | --- | --- |
| Primary and replica commit | 201/200 | fully replicated |
| Primary unavailable | 503 | nothing written anywhere |
| **Primary commits, replica fails** | **503** | **durable on the primary only** |
| Response lost after commit | client cannot tell | retry is safe; `PUT` is idempotent |

That third row is the honest one: **an error does not prove the write did not
land.** Retrying is safe, and the replica is behind until it resynchronizes.

### Generations

Each logical shard carries a **contiguous mutation sequence number**, persisted
in the *same SQLite transaction* as the mutation itself. A replica applies:

| Incoming | Action |
| --- | --- |
| `local + 1` | apply and advance |
| `<= local` | already applied — idempotent success, so retries are safe |
| `> local + 1` | **gap: refuse, and stop serving** |

Refusing the gap is the point. A replica that applied generation 6 while missing
5 would hold a corpus that never existed anywhere, yet would report the same
generation as its primary and look perfectly synchronized. Equality is only
evidence of synchronization because numbers cannot be skipped.

This is **not** a consensus term or epoch. It confers no authority and elects
nobody; it is a sequence number.

### Node states

```text
STARTING -> RECOVERING -> READY
                 ^          |
                 |          v
                 +---- OUT_OF_SYNC / DEGRADED
```

Only **READY** serves. Every other state refuses search, statistics and
scoring, which is how an unverified or out-of-sync copy is kept out of the
serving set — the coordinator does not have to reason about generations itself,
it simply skips a copy that says no.

### Read failover

Copy selection happens during round 1 of the two-round search: the primary is
tried first, then each replica, until one answers. **That copy is pinned for
round 2.** If it disappears between rounds the query fails rather than switching,
because copies can differ by unacknowledged mutations and pairing statistics
from one with scoring from another could produce a ranking matching no corpus
that ever existed.

Statistics come from **exactly one copy per logical shard**, so six physical
copies of a 5-document corpus still report `N = 5`, not 10.

| Failure | Search | Writes |
| --- | --- | --- |
| Primary down, replica READY | **works**, identical results | **fail** (503) |
| Replica down | works | **fail** (503) — write-all |
| Replica out of sync | works via primary | work |
| Every copy of a shard down | **503** | fail |

### Write failover: deliberately absent

There is **no leader election and no automatic promotion**. Safe automatic
promotion needs proof the replica is caught up *and* fencing so the old primary
cannot return and accept writes — that is consensus, and a half-built version
would be worse than none. Primary identity is static configuration, and the
roles are enforced structurally: a replica has no write path at all and answers
409, while a primary refuses replicated mutations. Two writable primaries
cannot arise.

### Recovery

A replica proves itself before serving:

```text
local recovery -> compare generation with the primary
     equal      -> READY
     behind     -> RECOVERING -> full resync -> rebuild -> validate -> READY
     unreachable-> NOT READY
```

That last line matters: a replica that cannot reach its primary **stays out of
service**. Claiming readiness without evidence would let it answer searches from
a corpus it cannot vouch for.

Recovery is a **full resynchronization** from the primary, not a replication
log: `GET /internal/export` returns a snapshot taken under the primary's lock —
microseconds, since the corpus is already in memory — so a recovering replica
never stitches together documents from different moments, and no write pause is
needed. The storage write is one transaction, so a failed resync leaves the
previous corpus intact rather than a mixture of two.

### Consistency guarantee, stated concretely

> **Every write that returned 2xx is durably present on every configured copy of
> its logical shard.** A read served by any READY copy therefore reflects every
> acknowledged write. A write that returned an error may be present on the
> primary only; whether such a document appears depends on which copy served the
> query, and no guarantee was made about it.

No "strong" or "eventual" without saying what they mean.

### Cluster status

`GET /cluster/status` reports each logical shard's copies with their role,
state and generation, plus `search_available` and `write_available` separately —
because after losing a primary those genuinely differ.

### Security scope

The `/internal/*` endpoints — including replication and export — assume a
**trusted internal network**. They have no authentication and must not be
exposed to the public internet. Nothing about them is secure; saying otherwise
would be false.

## Distributed search

### Roles

| Role | Serves | Owns |
| --- | --- | --- |
| `single` (default) | the full public API | its own corpus |
| `shard` + `primary` | `/internal/*`, `/health`, `/ready` | its logical shard, and writes to it |
| `shard` + `replica` | `/internal/*` (no write path), `/health`, `/ready` | a copy of its logical shard |
| `coordinator` | the full public API | nothing but the topology |

A shard deliberately exposes **no public `/search`**. Querying one shard directly
would return silently partial results scored against only that shard's
statistics, and the cheapest way to prevent that mistake is not to offer the
path. `single` is the default, so running one node requires knowing none of this.

### Routing

```text
shard_id = int(blake2b(document_id, digest_size=8), "big") % shard_count
```

`hash()` cannot be used: Python randomises string hashing per process, so a
document would route differently after a restart and differently again on
another node. BLAKE2b is standard-library, fast, and depends on nothing but the
input bytes; the encoding, digest size and byte order are all fixed because each
would change the answer if left to a default. Pinned vectors for three shards:

| `document_id` | shard |
| --- | --- |
| `doc-1` | 1 |
| `doc-2` | 2 |
| `doc-12` | 0 |
| `""` | 0 |

**Modulo, not consistent hashing.** Modulo is easy to reason about and easy to
verify. The price is that `shard_count` is fixed for the life of a cluster:
changing it moves nearly every document. That limitation is the motivation for
rebalancing work later, not something Phase 3 pre-solves.

Writes go to exactly one shard, are never broadcast, and are **never rerouted on
failure** — rerouting would break the ownership routing depends on.

### Distributed BM25

Scores from different shards are comparable only if they were computed from the
same corpus statistics. A term that is rare across the cluster but common on one
shard would otherwise get a different `idf` there, and merging those numbers
would compare quantities measured on different scales.

So a search takes two rounds, both parallel fan-outs:

```text
round 1   ask every shard for N, summed length, and df of the query's terms
          -> N = sum N_s,  total = sum len_s,  df(t) = sum df_s(t)
round 2   ask every shard for its local top-k, scored with those statistics
          -> merge candidates, sort, truncate
```

The sums are **exact, not estimates**, because every document lives on exactly
one shard — a property replication would break.

The coordinator analyses the query with the same `analyze()` the engine uses.
The full token sequence drives scoring, where a repeated term is meant to count
twice; the distinct terms drive the statistics request, where `df` is a property
of the term alone.

**The result:** distributed ranking is equivalent to a single node holding the
whole corpus — same ordering, same tie-breaking, same scores up to ordinary
floating-point behaviour. This is asserted by tests and was confirmed live
against a running single-node process.

### Global top-k

Each shard returns only its local top-`k`, which is sufficient for an exact
global top-`k` once scores are comparable: if a document is in the global top-k,
fewer than `k` documents outrank it anywhere, so fewer than `k` outrank it on its
own shard, so it is in that shard's local top-k and cannot be lost. The
coordinator merges all `S·k` candidates by score descending, then `document_id`
ascending — the single-node rule — so shard response order never influences the
result.

### Concurrency and the coordinator lock

Fan-out is `asyncio.gather` over one pooled HTTP client, so all shard requests
are in flight together and a search costs the **slowest** shard, not the sum.

One lock is held across both rounds of a search and across every routed
mutation. Without it a write could commit between the statistics round and the
scoring round, leaving the statistics describing one corpus and the scored
documents another. Phase 3 solves that conservatively — no distributed
snapshots, versions, epochs or transactions.

**The cost is real and deliberate: coordinator operations serialise.** One
search at a time, and no write while a search is in flight. Correctness first.

The guarantee covers mutations entering through the coordinator. The shards'
`/internal` endpoints are an implementation interface, **not a supported
external write path**; writing to them directly steps around this lock.

### Failure policy

**Any shard failure fails the whole query, with 503.** Two reasons: under
cluster-wide statistics a missing shard corrupts `N`, `avgdl` and `df`, so
partial results would be both incomplete *and* mis-scored; and without
replication there is nothing that could recover the missing documents anyway.
Nothing incomplete is ever returned as though it were complete.

| Situation | Response |
| --- | --- |
| Any shard unreachable or slow during search | `503`, naming the failing shards |
| Write or delete whose owning shard is down | `503`; never rerouted |
| Write to a healthy shard while another is down | succeeds normally |
| Any shard unready | `/ready` `503`; `/health` still `200` |

Readiness requires **every** shard, which is the only setting consistent with
the fail-whole policy: a cluster missing a shard cannot answer the queries it
advertises.

Every inter-node request is bounded by `PYSEARCH_CONNECT_TIMEOUT` and
`PYSEARCH_REQUEST_TIMEOUT`, so an unresponsive shard can never hang a query.

### Cluster statistics

`document_count` sums. `average_document_length` is **weighted** —
`Σ tokens / Σ documents`, never a mean of shard means, which would be wrong
whenever shards hold different numbers of documents.

There is **no cluster-wide `unique_term_count`**: vocabulary sizes cannot be
summed, because the same term legitimately appears on several shards, and the
true union would mean transferring every shard's vocabulary. Reporting a sum
would be arithmetically false, so per-shard figures are given instead.

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

### Semantic search

```bash
curl 'localhost:8000/search/semantic?q=car%20maintenance&limit=5'
```

Same response shape as `/search`, with one difference worth reading: `total` is
the number of documents **searched**, not matched. A similarity is defined for
every document, so there is no such thing as not matching — the ranking is the
answer. Scores lie in `[-1, 1]` and are **not comparable with BM25 scores**.

Returns `503` when semantic search is disabled, or when a logical shard has no
copy that can serve it.

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
| Routing | O(len(document_id)) | one BLAKE2b digest and a modulo |
| Distributed write | one network round trip + the local work above | routed to a single owning shard |
| Distributed search | 2 parallel fan-outs; latency ≈ the slowest shard | rounds are sequential, shards within a round are not |
| Global merge | O(S·k log(S·k)) | `S` shards each returning `k` candidates |
| Replicated write | primary commit + one call per replica | write latency now includes replication |
| Replicated search | unchanged | one serving copy per logical shard, never every replica |
| Resynchronization | O(documents in the logical shard) | full snapshot transfer, then lexical **and** vector rebuild |
| Embedding a document | O(tokens) | a table lookup per token and a mean; no network at inference |
| Semantic search | O(N·d + N log N) | one matrix multiply, then an ordering |
| Semantic fan-out | 1 parallel round | no statistics round: similarity needs no corpus-wide numbers |
| Vector memory | N·d·4 bytes | float32, 256 dimensions, excluding Python overhead |

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

### Run a local multi-process cluster

No containers needed — these are ordinary OS processes talking real HTTP:

```bash
uv run python scripts/run_cluster.py
```

Add a replica per logical shard:

```bash
uv run python scripts/run_cluster.py --replication-factor 2
```

It starts the shard nodes and a coordinator, each with its own database, waits
for readiness, and prints the coordinator's address. To watch failover, kill one
primary process and search again — results are unchanged, while writes to that
logical shard start returning 503. `GET /cluster/status` shows which copy is
serving. Then:

```bash
curl -X PUT localhost:8000/documents/doc-1 -H 'content-type: application/json' -d '{"text": "distributed search"}'
```

The response names the shard that took the write.

### Run the cluster with Docker Compose

```bash
docker compose up --build
```

The coordinator is published on port 8000. Seven services come up: three
primaries, three replicas and the coordinator. **Every physical node keeps its
own named volume** — a primary and its replica sharing a database would be two
views of one copy, which is not replication. The coordinator has no volume: it
owns no documents.

**These Docker commands are unverified.** Docker is not installed on the machine
this was developed on, so the image has never been built here. The distributed
system itself does not depend on Docker and was verified with real multi-process
clusters; Compose is a packaging convenience, not the architecture.

### Try semantic search

Semantic retrieval needs the optional extra and is off by default:

```bash
uv sync --extra semantic
```

```bash
PYSEARCH_SEMANTIC_ENABLED=true uv run --extra semantic uvicorn app.main:app
```

The pinned model (~30 MB) downloads once on first start. Then compare the two
retrieval modes on the same query:

```bash
curl 'localhost:8000/search/semantic?q=car+maintenance'
```

### Measure retrieval quality and cost

```bash
uv run --extra semantic python scripts/evaluate_retrieval.py
```

```bash
uv run --extra semantic python scripts/semantic_benchmark.py
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

`uv run pytest` never downloads a model: the suite runs against a deterministic
fake embedder. The tests that load the real one are behind a marker:

```bash
uv run --extra semantic pytest -m semantic_model
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
| `PYSEARCH_NODE_ROLE` | `single` | `single`, `shard`, `coordinator` |
| `PYSEARCH_SHARD_COUNT` | `1` | shards in the cluster; fixed for its lifetime |
| `PYSEARCH_SHARD_ID` | — | required on a shard; must be in `[0, shard_count)` |
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

### Phase 2 — Storage and index persistence ✅

A durable SQLite document corpus behind a narrow `DocumentStore` protocol,
transactional writes, startup recovery that rebuilds all derived state, an
explicit degraded state, and readiness reporting. Index snapshots and an
application-level write-ahead log were both evaluated and deliberately not
built; the reasoning is above.

### Phase 3 — Distributed search ✅

Shard nodes and a coordinator, BLAKE2b modulo routing, an internal HTTP node
API, parallel fan-out, cluster-wide BM25 statistics, deterministic global top-k
merging, a fail-whole partial-failure policy, and a Docker Compose topology for
reproducible multi-node local deployment.

### Phase 4 — Replication and availability ✅

Logical shards separated from physical nodes, synchronous write-all replication
with contiguous generations, automatic read failover, replica resynchronization,
split-brain prevention by static roles, and a capability-aware cluster status.
Leader election and automatic promotion were evaluated and deliberately not
built; the reasoning is above.

### Phase 5 — Semantic and vector search ✅ *current*

Embeddings behind a swappable boundary, an exact vector index, a semantic search
API, distributed semantic retrieval with failover, and a labelled evaluation
against BM25. Approximate nearest-neighbour search and FAISS were both evaluated
and deliberately declined; the reasoning and the measurements that would
overturn it are above.

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
├── storage/          durable corpus — no FastAPI in here either
│   ├── base.py       the DocumentStore protocol
│   ├── errors.py     StorageError, StorageInitializationError
│   └── sqlite_store.py
├── semantic/        embeddings and vectors — no FastAPI in here either
│   ├── embedder.py   the Embedder protocol, semantic identity, the model
│   ├── vector_index.py exact cosine search over a NumPy matrix
│   └── errors.py     SemanticDisabledError, EmbeddingError, ...
└── cluster/          distributed logic — no FastAPI in here either
    ├── routing.py    stable hash routing onto logical shards
    ├── topology.py   logical shards and the physical copies holding them
    ├── client.py     the ShardClient protocol and its HTTP implementation
    ├── replication.py write-all replication and replica synchronization
    ├── coordinator.py fan-out, global statistics, merge, failover policy
    └── errors.py     ShardUnavailableError, ReplicationError, ...
scripts/
├── demo.py              runnable walkthrough, including a restart
├── rebuild_benchmark.py startup recovery cost by corpus size
├── evaluate_retrieval.py BM25 versus semantic on a labelled set
├── semantic_benchmark.py embedding, rebuild and vector-search costs
└── run_cluster.py       a real multi-process cluster, without containers
Dockerfile, docker-compose.yml   reproducible multi-node deployment
tests/unit/           analysis, index, ranking, engine, storage, persistence
tests/integration/    the HTTP API, and restart recovery
```

Modules for hybrid retrieval and rank fusion will be created when the phase that
needs them begins — not in advance.

## Known limitations

Stated plainly, because each is a real consequence of a deliberate choice:

- **Writes need every copy.** One replica down stops writes to that logical
  shard. Read availability was bought with write availability.
- **No automatic write failover.** A lost primary means no writes for that shard
  until it returns; recovering write capacity is an operator action.
- **The coordinator is a single point of failure.** Replication improved shard
  availability and nothing else.
- **A failed write may still be durable on the primary.** The copies then differ
  until the replica resynchronizes.
- **A replica cannot recover while its primary is down.** There is no other
  authoritative source, and guessing which stale copy is newest would be worse.
- **Resynchronization transfers the whole logical shard** and materializes it in
  one response; there is no incremental catch-up log.
- **Internal endpoints are unauthenticated** and assume a trusted network.
- **The shard count and replication factor are fixed** for a cluster's life.
- **Vector search is exact**, so query cost grows linearly with the corpus.
- **Vectors are not persisted**, so every start re-embeds the whole corpus.
- **Embedding quality is that of a static model** — meaningfully below a real
  transformer, and the evaluation above is far too small to generalise from.
- **Every replica embeds independently**, doubling inference for each copy.
- **Semantic scores are not comparable with BM25 scores**, and nothing here
  combines them.
- **The first start with semantic enabled downloads the model**, so that node
  needs network access once.

## Independence and attribution

PySearch is an independent educational project inspired by general distributed
search-engine architecture. It is not affiliated with Elasticsearch or Elastic,
and does not contain Elasticsearch source code.
