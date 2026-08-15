# Architecture

How PySearch is put together, and why each piece is shaped the way it is.

- [The single-node engine](#the-single-node-engine)
- [Text analysis](#text-analysis)
- [The inverted index](#the-inverted-index)
- [BM25](#bm25)
- [Persistence and recovery](#persistence-and-recovery)
- [Logical shards and physical nodes](#logical-shards-and-physical-nodes)
- [The coordinator](#the-coordinator)
- [Lexical search flow](#lexical-search-flow)
- [Semantic search flow](#semantic-search-flow)
- [Hybrid search flow](#hybrid-search-flow)
- [Concurrency](#concurrency)
- [Complexity and memory](#complexity-and-memory)
- [Design decisions](#design-decisions)

## The single-node engine

Every node in a PySearch cluster — shard primary, replica, coordinator or a
standalone `single` node — is the same application. The role decides which
routers it exposes, not which engine it runs.

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

The boundary that matters is the one under FastAPI. Nothing in `app/search/`,
`app/storage/`, `app/semantic/`, `app/hybrid/` or `app/cluster/` imports
FastAPI, so the whole engine is usable from plain Python:

```python
from app.search.document import Document
from app.search.engine import SearchEngine
from app.storage.sqlite_store import SqliteDocumentStore

engine = SearchEngine(SqliteDocumentStore.open("pysearch.db"))
engine.initialize()  # opens the corpus and rebuilds derived state
engine.index_document(Document(document_id="doc-1", text="distributed search"))
engine.search("search", limit=10)
```

The HTTP layer only parses requests, validates them, calls the engine, and turns
the engine's transport-agnostic errors into status codes.

### One authoritative copy, three derived ones

```text
SQLite documents table      authoritative, durable
        |
        +--> _documents (cache)     derived, in memory
        +--> InvertedIndex          derived, in memory
        +--> ExactVectorIndex       derived, in memory
```

All three in-memory structures can be thrown away at any moment and
reconstructed by reading the corpus — which is exactly what happens at every
startup. The document cache is not a second source of truth: it is a derived
copy with a defined direction of reconstruction, and it exists so the search
path does not issue a database lookup per result.

The **inverted index is deliberately not persisted**. Storing it would create a
second durable copy of search state that can disagree with the documents, and
would bring invalidation and versioning problems with it. Rebuilding is
O(corpus) at startup and buys a consistency model with nothing to reconcile.
The same argument applies to vectors — see
[Vectors are derived](#vectors-are-derived-and-not-persisted).

## Text analysis

One pipeline, `analyze()`, runs over document text at index time and over query
text at search time. Sharing a single entry point is what guarantees the two
cannot drift apart — a query term normalized differently from the document term
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
CJK characters becomes one token.

**Stemming, lemmatization and stop-word removal are deliberately absent.** They
change what "the same word" means, and deserve to be introduced with evaluation
behind them. This is not a free choice — the missing stop-word filter is the
direct cause of the hybrid search failures documented in
[evaluation.md](evaluation.md#why-fusion-lost-exactly).

## The inverted index

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
matters while the index lives in a dict in RAM, and all of which belongs to a
design with an on-disk format.

`_document_terms` duplicates information derivable from `_postings`. That is
deliberate: without it, deleting one document means scanning the entire
vocabulary. With it, deletion costs O(unique terms in that document).

The index also has an explicit `validate()` invariant check — total length
equals the sum of document lengths, every posting references a live document, no
posting list is empty, the forward and inverted mappings agree, and each
document's length equals its summed term frequencies. It is a testing and
debugging aid, and is deliberately never called on the request path.

## BM25

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
penalizes long documents — a term appearing once in a 20-word document is
stronger evidence than the same term once in a 2000-word one — and `b` controls
how much, from `b=0` (ignore length) to `b=1` (fully normalize).

The IDF is the Lucene variant rather than the classic
`ln((N - df + 0.5) / (df + 0.5))`. The classic form goes negative once a term
appears in more than half the corpus, which lets a common term push a document
*below* one that does not contain it at all. Adding one inside the logarithm
keeps IDF positive for every `df`.

**Ordering is deterministic**: by descending score, then ascending
`document_id`. The second key is what makes ties reproducible rather than
dependent on dictionary iteration order. A term repeated in a query is scored
once per occurrence, so repeating it emphasizes it.

## Persistence and recovery

### Storage choice

SQLite through the standard library's `sqlite3`. It gives real ACID durability
and automatic crash recovery for **zero new dependencies**, keeps the corpus in
one portable file, and makes tests durable against a temporary directory.
Alternatives were weighed: a JSON rewrite (rewrites the whole corpus per write),
a JSONL append log (hand-rolling the WAL that SQLite already has), a custom
binary format (large bug surface, little marginal insight), and PostgreSQL (a
driver dependency and a server for a single-process, single-writer store).

SQLite stores documents and nothing else. Analysis, indexing and BM25 remain
PySearch's own work — **there is no SQL in the retrieval path**.

```sql
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    text TEXT NOT NULL
);
```

`PRAGMA user_version = 1` records the schema version in the database header, so
schema evolution has a hook without a metadata table. The default rollback
journal is kept rather than WAL, with `synchronous = FULL`. WAL's advantage is
letting readers run alongside a writer, and the engine's global lock means
SQLite never sees concurrent access, so that advantage is unrealized here.
`synchronous = FULL` is what makes "the API reported success" mean "the write
survived".

**No application-level write-ahead log** was written either. SQLite already
provides atomic, journaled, durable commits; a second WAL on top would duplicate
that machinery, add its own recovery path, and create a way for two durability
systems to disagree.

### Write ordering

Every mutation runs in this order, entirely under the engine lock:

```text
analyze + embed -> storage transaction -> durable COMMIT -> cache -> index -> vectors -> 2xx
```

Storage commits first, so the API can never report success for a write that is
not durable. Analysis and embedding happen *before* the durable write, which is
what makes an embedding failure harmless: nothing is committed, no generation is
consumed, and the node stays healthy.

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
open SQLite -> read documents -> rebuild cache + index + vectors -> validate -> ready
```

This runs in a FastAPI lifespan hook, which completes before Uvicorn serves
traffic, so no request can observe a half-built index. Recovery is never lazy.
Failure to open storage or rebuild **aborts startup** rather than serving a
broken service.

### Degraded state

The one failure a restart cannot fix while the process keeps running is a
derived-state update that raises *after* a durable commit. No compensating
rollback is attempted — the write really happened, and storage is authoritative.
Instead the engine is marked **degraded immediately**, and while degraded it
refuses document mutations, searches and index statistics with `503` rather than
serving results that might disagree with the authoritative corpus. `/ready`
reports `503` with the reason; `/health` still reports `200`, because the process
is alive and restarting it is the orchestrator's call.

| Situation | Response |
| --- | --- |
| Storage fails during a write | `503 {"detail": "storage unavailable"}`; nothing changed |
| Storage fails during a delete | `503`; nothing changed anywhere |
| Derived state fails after a commit | `503`; engine degraded; the durable write stands |
| Engine degraded | `503` on mutations, search and `/index/stats`; `/ready` `503` |
| Document not found on delete | `404` |

Database exceptions are wrapped in `StorageError` at the storage boundary and
logged server-side; no SQL, driver text or file path is ever returned to a
client.

## Logical shards and physical nodes

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
        shard-0-p.db  shard-0-r.db  ...       ...     six separate databases
```

A **logical shard** is a deterministic subset of documents. A **physical node**
is one process with one database. Routing keys on the logical shard, so adding
or removing replicas moves no documents at all — `replication_factor` never
enters the hash.

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

`hash()` cannot be used: Python randomizes string hashing per process, so a
document would route differently after a restart and differently again on
another node. BLAKE2b is standard-library, fast, and depends on nothing but the
input bytes; the encoding, digest size and byte order are all fixed because each
would change the answer if left to a default.

**Modulo, not consistent hashing.** Modulo is easy to reason about and easy to
verify. The price is that `shard_count` is fixed for the life of a cluster:
changing it moves nearly every document.

Writes go to exactly one shard, are never broadcast, and are **never rerouted on
failure** — rerouting would break the ownership routing depends on.

## The coordinator

The coordinator owns no documents. It holds the topology, routes writes, fans
queries out, merges results, and applies the failure policy. Losing it loses no
data — but it is a single point of failure for availability, since nothing else
can serve the public API in a sharded deployment.

One `asyncio.Lock` is held across every routed mutation and across both rounds
of a lexical search. Without it a write could commit between the statistics
round and the scoring round, leaving the statistics describing one corpus and
the scored documents another.

**The cost is real and deliberate: coordinator operations serialize.** One
search at a time, and no write while a search is in flight. Correctness first.

The guarantee covers mutations entering through the coordinator. The shards'
`/internal` endpoints are an implementation interface, **not a supported
external write path**; writing to them directly steps around this lock.

## Lexical search flow

Scores from different shards are comparable only if they were computed from the
same corpus statistics. A term that is rare across the cluster but common on one
shard would otherwise get a different `idf` there, and merging those numbers
would compare quantities measured on different scales.

So a search takes **two rounds**, both parallel fan-outs:

```text
round 1   ask every shard for N, summed length, and df of the query's terms
          -> N = sum N_s,  total = sum len_s,  df(t) = sum df_s(t)
round 2   ask every shard for its local top-k, scored with those statistics
          -> merge candidates, sort, truncate
```

The sums are **exact, not estimates**, because every document lives on exactly
one shard — a property replication would break, which is why statistics come
from exactly one copy per logical shard.

### Global top-k

Each shard returns only its local top-`k`, which is sufficient for an exact
global top-`k` once scores are comparable: if a document is in the global top-k,
fewer than `k` documents outrank it anywhere, so fewer than `k` outrank it on its
own shard, so it is in that shard's local top-k and cannot be lost. The
coordinator merges all `S·k` candidates by score descending, then `document_id`
ascending — the single-node rule — so shard response order never influences the
result.

**The result:** distributed ranking is equivalent to a single node holding the
whole corpus — same ordering, same tie-breaking, same scores up to ordinary
floating-point behaviour. This is asserted by tests and was confirmed live
against a running single-node process.

### Failure policy

**Any shard failure fails the whole query, with 503.** Under cluster-wide
statistics a missing shard corrupts `N`, `avgdl` and `df`, so partial results
would be both incomplete *and* mis-scored. Nothing incomplete is ever returned
as though it were complete. Readiness requires **every** shard, which is the only
setting consistent with that policy.

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

## Semantic search flow

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
the other**. Semantic search is **off by default** (`PYSEARCH_SEMANTIC_ENABLED`),
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

**`uv run pytest` never downloads anything**: the suite runs against a
deterministic fake embedder, and the tests that use the real model are behind a
marker.

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
overturn that are in [evaluation.md](evaluation.md#what-retrieval-costs), and
`scripts/semantic_benchmark.py` reproduces them.

### Vectors are derived, and not persisted

Documents remain the only authoritative corpus. Vectors are rebuilt by
re-embedding at startup, so there is no second durable derived state and no
model-version invalidation problem: if the configured model changes, every
vector is simply rebuilt from the documents.

Replication ships **documents only**; every copy derives its own vectors. That
keeps one rule instead of two, because resynchronization already transfers
documents and rebuilds derived state — so the replication path and the recovery
path do the same thing. The cost is duplicate inference, cheap with this model
and worth revisiting with an expensive one.

### One round, not two

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
replicas are never double-counted.

## Hybrid search flow

```text
                              Query
                                |
                  +-------------+-------------+
                  |                           |
                  v                           v
          Distributed BM25            Semantic retrieval
          (2 rounds)                  (1 round)
                  |                           |
                  v                           v
           Lexical ranking            Semantic ranking
                  |                           |
                  +-------------+-------------+
                                |
                                v
                     Reciprocal Rank Fusion
                                |
                                v
                         Hybrid ranking
```

`/search` and `/search/semantic` are untouched. `/search/hybrid` composes them —
it re-implements neither fan-out.

### Why not add the two scores

The obvious approach is wrong:

```text
hybrid = 0.5 * bm25 + 0.5 * cosine        # not defensible
```

BM25 is unbounded, depends on corpus statistics, and moves with the query — in
this project's own measurements the same document scored 0.3696 and then 0.4627
for the same query after one deletion changed `df`. Cosine over unit vectors
lives in `[-1, 1]` and occupies a much narrower band in practice. The "weight"
would not be a weight: lexical would dominate on rare terms and vanish on
common ones. And BM25 legitimately returns **nothing** for a paraphrase, so on
exactly the queries semantic search exists to answer, half the weighted sum
would contribute zero.

Normalizing first does not rescue it. Min-max over the returned candidates is
candidate-set dependent: the best candidate maps to 1.0 whether it is excellent
or merely the least bad, and an empty list has no range at all.

### Reciprocal Rank Fusion

Rank, not score:

```text
RRF(d) = sum over the lists containing d of  1 / (k + rank(d))     ranks 1-based
```

A document in only one list contributes only that term — **no penalty rank**,
because "BM25 found nothing" is a normal outcome here. A document in both
appears once, with its contributions summed. Ordering is fusion score
descending, then `document_id` ascending; no underlying score acts as a hidden
tie-break.

`score` is a **fusion score**. Not BM25 relevance, not a cosine, not a
probability. Values are small (≈0.016–0.033) and mean something only relative to
each other within one response.

### Candidate depth, and what it does not guarantee

Each path is asked for `min(5 × limit, 100)` candidates, so `limit=10` retrieves
50 from each. `total` is the **candidate union size** — how many distinct
documents entered fusion.

**Hybrid results are the exact RRF of the retrieved candidate lists, not
necessarily of the whole corpus.** This is a real limitation and unlike the
distributed top-k arguments elsewhere in this project it cannot be argued away.
There, local top-k provably sufficed because scores were globally comparable.
Here truncation happens *before* fusion and RRF consumes ranks, so a document
outside both candidate lists is absent even if its fused score would have placed
it. `tests/unit/test_fusion.py` asserts this rather than only documenting it.

### Execution and the lock

Both the engine and the coordinator split each retrieval into a thin public
wrapper and a lock-free internal, so hybrid acquires the operation lock **once**.
This was not optional: the locks are not reentrant, so a hybrid method calling
the two public methods would have deadlocked rather than merely serialized.

Holding one lock across both retrievals is what stops a write from landing
between them and producing a lexical ranking of one corpus state fused with a
semantic ranking of another.

Inside that lock the two behave differently, on purpose:

| | Coordinator | Single node |
| --- | --- | --- |
| Work | two network fan-outs | in-memory CPU |
| Execution | **concurrent** (`asyncio.gather`) | **sequential** |
| Why | each path spends its time waiting, so they overlap | threads would add contention without shortening anything |

Measured on the live 7-process cluster, concurrency saved a **median 9.3%**
(range −1.4% to +15.4%) — real but well short of `max(lexical, semantic)`,
because the shard nodes serialize the overlapping requests behind their own
per-node locks. Sequential and concurrent produce identical rankings.

### Failure semantics

**Either retrieval path failing fails the request**, with 503. "Hybrid" asserts
that both signals took part, so returning one of them under that name would
misdescribe the result. A lexical result that is merely *empty* is not a
failure — that is the ordinary outcome for a paraphrase, and fusion handles it
as a one-sided contribution.

With semantic search disabled, `/search/hybrid` returns **503** naming the
cause, rather than silently degrading to BM25.

## Concurrency

FastAPI runs synchronous path operations in a thread pool, so requests really do
execute concurrently against the index. Two concurrent writes would interleave
their read-modify-write of the running token total and lose an update; a query
iterating a posting list while a delete mutates it would fail outright.

The approach is the simplest correct one: **one `threading.Lock` in the engine,
held across every read and every write**. It spans the storage commit as well,
because the durable write and the derived updates have to be atomic with respect
to each other. Two consequences follow, and both are real costs rather than
details: queries are serialized with each other, and disk latency sits inside
the critical section, so a slow disk stalls searches as well as writes.

The alternative — `async` handlers relying on the event loop for atomicity —
needs no lock, but the invariant disappears silently the moment anyone adds an
`await` inside a critical section, and CPU-bound work would block the loop.

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
| Fusion | O(L + S + n log n) | both candidate lists walked once, then the union ordered |
| Vector memory | N·d·4 bytes | float32, 256 dimensions, excluding Python overhead |

The query cost is the whole point of an inverted index: it is proportional to
how many documents contain the query terms, not to `N`. A term absent from the
corpus costs a single failed dict lookup.

Memory is dominated by the posting entries: one `document_id` key and one
integer per (term, document) pair, plus the unique-term set and length per
document, plus the original text of every document. All of it is Python objects
in a dict, so the constant factor is large.

## Design decisions

The short form of why the system looks like this. Each links to the reasoning
above.

| Decision | Instead of | Why |
| --- | --- | --- |
| Own inverted index and BM25 | Lucene, Whoosh, Tantivy | The project exists to work through them; delegating hides everything interesting |
| SQLite as source of truth | JSON, JSONL, custom binary, PostgreSQL | Real ACID durability for zero dependencies, one portable file |
| Rebuild derived state at startup | Persisting the index and vectors | One durable copy means nothing to reconcile or invalidate |
| Deterministic modulo sharding | Consistent hashing | Easy to reason about and verify; the price (fixed shard count) is documented |
| Cluster-wide BM25 statistics | Per-shard scoring | Without shared `idf`, scores from different shards are not comparable |
| HTTP between nodes | gRPC | The transport is not what the project is about; HTTP is inspectable with `curl` |
| Synchronous write-all replication | Async replication | A lagging replica holds a different corpus, so failover would silently change rankings |
| No automatic leader election | Raft, Paxos, or ad-hoc promotion | Safe promotion needs catch-up proof and fencing — that is consensus, and a half-built one is worse than none |
| Exact vector search | FAISS, HNSW, ANN | ANN can only be evaluated against exact; delete/replace must be exact for replication |
| Documents-only semantic replication | Shipping vectors | Replication and recovery then follow one rule instead of two |
| RRF over score addition | Weighted sum of BM25 and cosine | The two scales are incompatible, and normalizing is candidate-set dependent |
