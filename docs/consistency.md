# Consistency and availability

What PySearch actually guarantees, stated concretely. "Strongly consistent" and
"eventually consistent" are avoided here as labels — each guarantee below is
written as something an observer could check.

- [The headline guarantee](#the-headline-guarantee)
- [Durable source of truth](#durable-source-of-truth)
- [Write-all replication](#write-all-replication)
- [The ambiguous failure window](#the-ambiguous-failure-window)
- [Generations](#generations)
- [Node states](#node-states)
- [Read failover](#read-failover)
- [No automatic promotion](#no-automatic-promotion)
- [Recovery](#recovery)
- [Two-round BM25 consistency](#two-round-bm25-consistency)
- [Semantic serving equivalence](#semantic-serving-equivalence)
- [The hybrid snapshot window](#the-hybrid-snapshot-window)
- [What is not guaranteed](#what-is-not-guaranteed)

## The headline guarantee

> **Every write that returned 2xx is durably present on every configured copy of
> its logical shard.** A read served by any READY copy therefore reflects every
> acknowledged write. A write that returned an error may be present on the
> primary only; whether such a document appears depends on which copy served the
> query, and no guarantee was made about it.

Everything below is the mechanism behind that sentence, and the places where it
stops applying.

## Durable source of truth

Each physical node has exactly one SQLite database, and that database is the only
authoritative state. The document cache, the inverted index and the vector index
are all derived, held in memory, and rebuilt from the corpus at every startup.

A mutation commits to SQLite with `synchronous = FULL` **before** the API
reports success, so "the API returned 201" means "the write survived a power
loss". Derived structures are updated after the commit, under the same lock.

If a derived update fails after a durable commit, the engine marks itself
**degraded** and refuses to serve rather than answering from state that may
disagree with the corpus. It does not attempt a compensating rollback: the write
really happened, and storage is authoritative.

**Run one process per database.** Two processes over one file would each hold
their own in-memory index and would not see each other's writes.

## Write-all replication

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
to that logical shard fail.** Read availability was bought with write
availability.

| Situation | Response | State |
| --- | --- | --- |
| Primary and replica commit | 201/200 | fully replicated |
| Primary unavailable | 503 | nothing written anywhere |
| **Primary commits, replica fails** | **503** | **durable on the primary only** |
| Response lost after commit | client cannot tell | retry is safe; `PUT` is idempotent |

## The ambiguous failure window

That third row is the honest one, and it deserves its own section because it is
the guarantee's real boundary.

**An error does not prove the write did not land.** If the primary commits and
the replica then fails, the coordinator returns 503, but the document is durably
present on the primary. Until that replica resynchronizes, the two copies differ
by that document, and which one a read observes depends on which copy served it.

Two things make this tolerable rather than dangerous:

- `PUT /documents/{id}` is **idempotent**, so the safe client response to any
  error is to retry. A retry that lands twice is indistinguishable from one that
  lands once.
- The divergence is bounded and self-healing: the replica is behind, it knows it
  is behind (see [Generations](#generations)), and it repairs itself on restart
  rather than serving stale data.

What is *not* claimed: atomic cross-copy writes. There is no two-phase commit
and no distributed transaction. A single-copy commit followed by a replica
failure is a real, reachable state.

## Generations

Each logical shard carries a **contiguous mutation sequence number**, persisted
in the *same SQLite transaction* as the mutation itself. A replica applies:

| Incoming | Action |
| --- | --- |
| `local + 1` | apply and advance |
| `<= local` | already applied — idempotent success, so retries are safe |
| `> local + 1` | **gap: refuse, and stop serving** |

Refusing the gap is the point. A replica that applied generation 6 while missing
5 would hold a corpus that **never existed anywhere**, yet would report the same
generation as its primary and look perfectly synchronized. Equality is only
evidence of synchronization because numbers cannot be skipped.

This is **not** a consensus term or epoch. It confers no authority and elects
nobody; it is a sequence number. Nothing in PySearch votes.

## Node states

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

This is the mechanism that makes the headline guarantee checkable: a copy that
cannot prove it holds every acknowledged write does not answer queries.

## Read failover

Copy selection happens during round 1 of the two-round lexical search: the
primary is tried first, then each replica, until one answers. **That copy is
pinned for round 2.** If it disappears between rounds the query fails rather than
switching, because copies can differ by unacknowledged mutations, and pairing
statistics from one copy with scoring from another could produce a ranking
matching no corpus that ever existed.

Statistics come from **exactly one copy per logical shard**, so six physical
copies of a 5-document corpus still report `N = 5`, not 10.

| Failure | Search | Writes |
| --- | --- | --- |
| Primary down, replica READY | **works**, identical results | **fail** (503) |
| Replica down | works | **fail** (503) — write-all |
| Replica out of sync | works via primary | work |
| Every copy of a shard down | **503** | fail |

Failover producing *identical* results — not merely working results — was
verified byte-for-byte on a real 7-process cluster, for all three retrieval
modes.

## No automatic promotion

There is **no leader election and no automatic promotion**. Safe automatic
promotion needs proof the replica is caught up *and* fencing so the old primary
cannot return and accept writes — that is consensus, and a half-built version
would be worse than none.

Primary identity is static configuration, and the roles are enforced
structurally: a replica has no write path at all and answers 409, while a
primary refuses replicated mutations. **Two writable primaries cannot arise**,
because neither role can be assumed at runtime.

The consequence, stated plainly: losing a primary leaves its logical shard
**readable but not writable** until an operator brings it back.
`GET /cluster/status` reports `search_available` and `write_available`
separately, because after losing a primary they genuinely differ.

## Recovery

A replica proves itself before serving:

```text
local recovery -> compare generation with the primary
     equal      -> READY
     behind     -> RECOVERING -> full resync -> rebuild -> validate -> READY
     unreachable-> NOT READY
```

That last line matters: a replica that cannot reach its primary **stays out of
service**. Claiming readiness without evidence would let it answer searches from
a corpus it cannot vouch for. It also means a replica cannot recover while its
primary is down — there is no other authoritative source, and guessing which
stale copy is newest would be worse.

Recovery is a **full resynchronization**, not a replication log:
`GET /internal/export` returns a snapshot taken under the primary's lock —
microseconds, since the corpus is already in memory — so a recovering replica
never stitches together documents from different moments, and no write pause is
needed. The storage write is one transaction, so a failed resync leaves the
previous corpus intact rather than a mixture of two.

The cost: resynchronization transfers the whole logical shard and materializes
it in one response. There is no incremental catch-up.

## Two-round BM25 consistency

A distributed lexical search collects corpus statistics in round 1 and scores
with them in round 2. Both rounds happen inside **one coordinator lock
acquisition**, so no write routed through the coordinator can land between them.

Without that, statistics would describe one corpus state and the scored
documents another — producing a ranking that corresponds to no moment in the
cluster's history.

The lock covers mutations entering through the coordinator. Writing directly to
a shard's `/internal` endpoints steps around it; those endpoints are an
implementation interface, not a supported external write path.

## Semantic serving equivalence

Semantic search takes one round, so it has no cross-round pinning problem. Its
consistency question is different: are two copies' vectors comparable?

The answer is a **semantic identity** — implementation, model id, pinned
revision, dimension, normalization — sent with every query vector and checked by
the serving copy, which refuses with 409 on any disagreement.

What is guaranteed: copies sharing an identity produce **equivalent embeddings,
identical ordering and identical tie-breaking**, with scores equal to within a
strict tolerance.

What is **not** guaranteed: bit-for-bit identical vectors. Library versions and
floating-point reduction order can move the last bits. The tests assert the
former and deliberately do not assert the latter.

## The hybrid snapshot window

Hybrid search runs a lexical retrieval and a semantic retrieval and fuses them.
Both run inside **one lock acquisition** — one `asyncio.Lock` on the
coordinator, one `threading.Lock` on a single node.

That is what makes the two rankings describe the same corpus state. Without it a
write could land between them, and the fused result would combine a lexical view
of one corpus with a semantic view of another.

The lock is acquired exactly once because both the engine and the coordinator
split each retrieval into a public wrapper and a lock-free internal method. The
locks are **not reentrant**, so the naive implementation — hybrid calling the two
public methods — would have deadlocked rather than merely serialized.

A separate limitation applies to fusion itself: hybrid results are the exact RRF
of the **retrieved candidate lists**, not of the whole corpus. See
[architecture.md](architecture.md#candidate-depth-and-what-it-does-not-guarantee).

## What is not guaranteed

Collected in one place, so none of it has to be inferred:

- **No atomic multi-document writes.** Each document mutation is independent.
- **No atomic cross-copy writes.** See
  [the ambiguous failure window](#the-ambiguous-failure-window).
- **No read-your-writes across a failed write.** A 503 leaves the outcome
  genuinely unknown until the copies reconverge.
- **No monotonic reads across copies.** During the divergence window, two reads
  served by different copies may disagree about one document.
- **No consensus, no quorum, no election.** Generations are sequence numbers.
- **No coordinator redundancy.** It is a single point of failure for
  availability; it holds no data, so losing it loses nothing durable.
- **No dynamic membership.** Shard count and replication topology are fixed for
  a cluster's life; changing the shard count moves nearly every document.
- **No isolation between concurrent searches.** They serialize behind one lock
  rather than running under snapshots.
