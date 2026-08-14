"""The search engine: the seam between the HTTP layer and retrieval internals.

State model
-----------

There is exactly one authoritative copy of the corpus, and it is on disk::

    SQLite documents table      authoritative, durable
            |
            v
    _documents                  derived cache, in memory
    InvertedIndex               derived index, in memory

Both in-memory structures may be discarded at any moment and reconstructed by
reading the store — that is what :meth:`SearchEngine.initialize` does at
startup. Keeping a document cache is not a second source of truth: it is a
derived copy with a defined direction of reconstruction, and it keeps the
search hot path from issuing a database lookup per result.

Write ordering
--------------

Every mutation runs in this order, entirely under the lock::

    storage transaction -> durable COMMIT -> document cache -> inverted index

Storage commits first, so the API can never report success for a write that is
not durable. If the process dies at any point after the commit, the derived
structures are simply rebuilt from storage on the next start.

Degraded state
--------------

The one failure the model cannot repair in flight is a derived-state update
that raises *after* a durable commit. No compensating delete is attempted — the
write really did happen and storage is authoritative. Instead the engine is
marked degraded immediately, and while degraded it refuses document mutations,
searches and statistics rather than serving results that may disagree with the
authoritative corpus. ``/ready`` reports 503, and reinitialisation — normally a
restart — repairs the engine by rebuilding everything from SQLite.

Concurrency
-----------

FastAPI runs synchronous path operations in a thread pool, so several requests
genuinely execute at the same time against this state. One lock is held across
every read and every write, and it now spans the storage commit as well, since
the durable write and the derived updates have to be atomic with respect to
each other. That means disk latency sits inside the critical section: writes
serialise and a slow disk stalls searches too. Correctness first; a
reader-writer scheme is a later optimisation with measurements behind it.
"""

import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum

from app.search.analysis import analyze
from app.search.document import Document
from app.search.errors import DocumentNotFoundError, EngineNotReadyError, IndexInvariantError
from app.search.index import CorpusStats, IndexStats, InvertedIndex
from app.search.ranking import BM25Scorer
from app.storage.base import DocumentStore


@dataclass(frozen=True, slots=True)
class SearchResult:
    """One ranked hit."""

    document_id: str
    score: float
    text: str


@dataclass(frozen=True, slots=True)
class SearchResults:
    """A page of ranked hits, plus how many documents matched in total."""

    total: int
    results: tuple[SearchResult, ...]


@dataclass(frozen=True, slots=True)
class RebuildReport:
    """The outcome of rebuilding derived state from storage."""

    document_count: int
    duration_seconds: float


class NodeState(StrEnum):
    """The serving state of one physical copy.

    Only ``READY`` may serve traffic. The others exist so that a copy which is
    unverified, mid-recovery or known to have missed a mutation cannot be
    selected for search, statistics or failover.
    """

    STARTING = "starting"
    RECOVERING = "recovering"
    READY = "ready"
    DEGRADED = "degraded"
    OUT_OF_SYNC = "out_of_sync"


@dataclass(frozen=True, slots=True)
class EngineStatus:
    """Whether the engine can serve requests, and why not if it cannot."""

    state: NodeState
    ready: bool
    detail: str
    generation: int


class ReplicationOutcome(StrEnum):
    """What a replica did with a replicated mutation."""

    APPLIED = "applied"
    DUPLICATE = "duplicate"
    GAP = "gap"


class SearchEngine:
    """Indexes documents durably and answers queries over derived state."""

    def __init__(self, store: DocumentStore, scorer: BM25Scorer | None = None) -> None:
        self._store = store
        self._index = InvertedIndex()
        self._documents: dict[str, Document] = {}
        self._scorer = scorer if scorer is not None else BM25Scorer()
        self._lock = threading.Lock()
        self._state = NodeState.STARTING
        self._detail = "not initialized"
        self._generation = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> RebuildReport:
        """Rebuild all derived state from storage and make the engine ready.

        Called once at startup, before the application serves traffic, and again
        by anything that wants to repair a degraded engine in place. Recovery is
        never lazy: no request can observe a half-built index.

        A replica calls :meth:`mark_recovering` immediately afterwards, because
        local recovery alone does not prove it holds everything its primary has.

        Raises:
            StorageError: if the corpus cannot be read.
            IndexInvariantError: if the rebuilt index is inconsistent.
        """
        with self._lock:
            report = self._rebuild_locked()
            self._state = NodeState.READY
            self._detail = "ready"
        return report

    def close(self) -> None:
        """Close the underlying storage and stop serving."""
        with self._lock:
            self._state = NodeState.STARTING
            self._detail = "closed"
            self._store.close()

    def status(self) -> EngineStatus:
        """Report this copy's serving state and generation."""
        with self._lock:
            ready = self._state is NodeState.READY
            return EngineStatus(
                state=self._state,
                ready=ready,
                # Non-serving states name themselves, so a readiness response
                # says why it is refusing rather than only that it refuses.
                detail=self._detail if ready else f"{self._state}: {self._detail}",
                generation=self._generation,
            )

    @property
    def generation(self) -> int:
        """The mutation sequence number this copy has applied up to."""
        with self._lock:
            return self._generation

    def mark_recovering(self, detail: str) -> None:
        """Stop serving while synchronisation with the primary is unverified."""
        with self._lock:
            self._state = NodeState.RECOVERING
            self._detail = detail

    def mark_out_of_sync(self, detail: str) -> None:
        """Stop serving because this copy is known to have missed a mutation."""
        with self._lock:
            self._state = NodeState.OUT_OF_SYNC
            self._detail = detail

    def mark_ready(self) -> None:
        """Begin serving, after synchronisation has been verified."""
        with self._lock:
            self._state = NodeState.READY
            self._detail = "ready"

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def index_document(self, document: Document) -> bool:
        """Store a document durably, then reflect it in derived state.

        Returns:
            ``True`` if the document is new, ``False`` if it replaced one.

        Raises:
            EngineNotReadyError: if the engine is not initialized or is degraded.
            StorageError: if the document could not be stored, in which case
                nothing has changed anywhere.
        """
        # Analysis is pure and touches no shared state, so it stays outside the
        # critical section.
        tokens = analyze(document.text)

        with self._lock:
            self._require_operational_locked()
            generation = self._generation + 1
            created = self._store.put(document, generation)
            self._generation = generation

            # Past this point the durable corpus has changed. A failure here
            # cannot be undone by rolling storage back, so it degrades the
            # engine instead and leaves repair to a rebuild.
            with self._degrade_on_failure("failed to update derived state after a durable write"):
                self._documents[document.document_id] = document
                self._index.add_document(document.document_id, tokens)

            return created

    def delete_document(self, document_id: str) -> None:
        """Remove a document durably, then reflect that in derived state.

        Raises:
            DocumentNotFoundError: if the document is not stored.
            EngineNotReadyError: if the engine is not initialized or is degraded.
            StorageError: if storage could not be reached.
        """
        with self._lock:
            self._require_operational_locked()

            # Checked against the cache first, because a delete that matches
            # nothing must not consume a generation: generations number the
            # mutations a copy has applied, and a no-op is not one of them.
            if document_id not in self._documents:
                raise DocumentNotFoundError(f"document {document_id!r} is not indexed")

            generation = self._generation + 1
            self._store.delete(document_id, generation)
            self._generation = generation

            with self._degrade_on_failure("failed to update derived state after a durable delete"):
                del self._documents[document_id]
                self._index.remove_document(document_id)

    # ------------------------------------------------------------------
    # Replication (replica side)
    # ------------------------------------------------------------------

    def apply_replicated_put(self, document: Document, generation: int) -> ReplicationOutcome:
        """Apply a mutation replicated from the primary.

        Generations form a contiguous sequence, and that is the whole point:
        equality with the primary can only be read as evidence of
        synchronisation if a copy cannot skip a number. So exactly one value is
        acceptable, and anything beyond it means a mutation was missed.

        ``generation == local + 1``   apply it and advance
        ``generation <= local``       a redelivery; already applied, so succeed
        ``generation >  local + 1``   a gap: refuse, and stop serving

        Refusing the gap matters more than applying the newest data. A copy that
        applied generation 6 while missing 5 would hold a corpus that never
        existed anywhere, yet would report the same generation as the primary
        and look perfectly synchronised.
        """
        tokens = analyze(document.text)

        with self._lock:
            outcome = self._classify_generation_locked(generation)
            if outcome is not ReplicationOutcome.APPLIED:
                return outcome

            self._store.put(document, generation)
            with self._degrade_on_failure(
                "failed to update derived state after a replicated write"
            ):
                self._documents[document.document_id] = document
                self._index.add_document(document.document_id, tokens)
            self._generation = generation
            return outcome

    def apply_replicated_delete(self, document_id: str, generation: int) -> ReplicationOutcome:
        """Apply a replicated deletion.

        Deliberately a no-op when the document is absent rather than an error.
        The public API answers 404 for a missing document, but a replica may
        legitimately not hold one — so replication treats "already gone" as
        success instead of manufacturing a failure the primary cannot act on.
        """
        with self._lock:
            outcome = self._classify_generation_locked(generation)
            if outcome is not ReplicationOutcome.APPLIED:
                return outcome

            self._store.delete(document_id, generation)
            with self._degrade_on_failure(
                "failed to update derived state after a replicated delete"
            ):
                if document_id in self._documents:
                    del self._documents[document_id]
                    self._index.remove_document(document_id)
            self._generation = generation
            return outcome

    def export_snapshot(self) -> tuple[tuple[Document, ...], int]:
        """Return a consistent snapshot of this copy's corpus and generation.

        Taken under the lock, but only long enough to copy references to
        documents already held in memory — microseconds — so a resynchronising
        replica never stitches together documents from different moments, and
        no write pause is needed to give it a coherent view.
        """
        with self._lock:
            self._require_operational_locked()
            return tuple(self._documents.values()), self._generation

    def resynchronize(self, documents: Sequence[Document], generation: int) -> RebuildReport:
        """Replace this copy's corpus wholesale with a snapshot from the primary.

        The storage write is one transaction, so a failure part-way leaves the
        previous corpus intact rather than a mixture of two. Derived state is
        then rebuilt and validated before the copy is allowed to serve again.
        """
        with self._lock:
            self._state = NodeState.RECOVERING
            self._detail = f"resynchronizing to generation {generation}"

            self._store.replace_all(documents, generation)
            report = self._rebuild_locked()

            if self._generation != generation:
                raise IndexInvariantError(
                    f"resynchronization expected generation {generation} but storage "
                    f"reports {self._generation}"
                )

            self._state = NodeState.READY
            self._detail = "ready"
            return report

    def _classify_generation_locked(self, generation: int) -> ReplicationOutcome:
        """Decide what to do with an incoming generation. Caller holds the lock."""
        if generation <= self._generation:
            return ReplicationOutcome.DUPLICATE
        if generation > self._generation + 1:
            self._state = NodeState.OUT_OF_SYNC
            self._detail = f"generation gap: local {self._generation}, received {generation}"
            return ReplicationOutcome.GAP

        self._require_operational_locked()
        return ReplicationOutcome.APPLIED

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def search(
        self, query: str, limit: int, corpus_stats: CorpusStats | None = None
    ) -> SearchResults:
        """Return the top ``limit`` documents for ``query``, best first.

        The query runs through the same analysis pipeline as documents. A query
        that yields no terms — because it was empty, or was nothing but
        punctuation — matches nothing; that is an empty result, not an error.

        Results are ordered by descending score, then by ascending document id.
        The second key is what makes ties reproducible instead of dependent on
        dictionary iteration order.

        ``corpus_stats`` lets a caller score against statistics wider than this
        engine's own corpus. A shard is given cluster-wide values so its scores
        are comparable with the other shards'; omitted, the engine uses its own,
        which is the single-node case.

        Raises:
            EngineNotReadyError: if the engine is not initialized or is degraded.
        """
        terms = analyze(query)

        with self._lock:
            self._require_operational_locked()

            if not terms:
                return SearchResults(total=0, results=())

            scores = self._scorer.score_query(self._index, terms, corpus_stats)
            ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
            results = tuple(
                SearchResult(
                    document_id=document_id,
                    score=score,
                    text=self._documents[document_id].text,
                )
                for document_id, score in ranked[:limit]
            )

        return SearchResults(total=len(ranked), results=results)

    def corpus_stats(self, terms: Sequence[str]) -> CorpusStats:
        """Return this engine's BM25 inputs for ``terms``.

        A coordinator sums these across shards to obtain cluster-wide statistics
        before asking the shards to score anything.

        Raises:
            EngineNotReadyError: if the engine is not initialized or is degraded.
        """
        with self._lock:
            self._require_operational_locked()
            return self._index.corpus_stats(terms)

    def stats(self) -> IndexStats:
        """Return a snapshot of the corpus statistics.

        Refused while degraded: these numbers are read straight out of the
        derived index, and reporting a document count that disagrees with the
        authoritative corpus is precisely the confusion the degraded state
        exists to prevent. ``/ready`` carries the diagnosis instead.

        Raises:
            EngineNotReadyError: if the engine is not initialized or is degraded.
        """
        with self._lock:
            self._require_operational_locked()
            return self._index.stats()

    # ------------------------------------------------------------------
    # Invariants
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Check that storage, the document cache and the index all agree.

        A testing and debugging aid; the storage comparison makes it O(corpus),
        so it is never called on the request path. It deliberately still works
        while degraded, since that is when it is most informative.

        Raises:
            IndexInvariantError: on the first violation found.
        """
        with self._lock:
            self._index.validate()

            if set(self._documents) != self._index.document_ids():
                raise IndexInvariantError(
                    "the document cache and the index hold different documents"
                )

            stored_generation = self._store.generation()
            if stored_generation != self._generation:
                raise IndexInvariantError(
                    f"storage is at generation {stored_generation} but the engine "
                    f"believes it is at {self._generation}"
                )

            stored_count = self._store.count()
            if stored_count != len(self._documents):
                raise IndexInvariantError(
                    f"storage holds {stored_count} documents but the cache holds "
                    f"{len(self._documents)}"
                )

            for document in self._store.iter_documents():
                cached = self._documents.get(document.document_id)
                if cached is None:
                    raise IndexInvariantError(
                        f"stored document {document.document_id!r} is missing from the cache"
                    )
                if cached.text != document.text:
                    raise IndexInvariantError(
                        f"cached text for {document.document_id!r} differs from storage"
                    )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _rebuild_locked(self) -> RebuildReport:
        """Reconstruct the cache and index from storage. Caller holds the lock.

        Built into fresh structures and swapped in at the end, so a failure
        part-way through leaves the previous state untouched rather than
        half-replaced.
        """
        started = time.perf_counter()

        index = InvertedIndex()
        documents: dict[str, Document] = {}
        for document in self._store.iter_documents():
            documents[document.document_id] = document
            index.add_document(document.document_id, analyze(document.text))

        index.validate()

        self._index = index
        self._documents = documents
        self._generation = self._store.generation()
        return RebuildReport(
            document_count=len(documents),
            duration_seconds=time.perf_counter() - started,
        )

    def _require_operational_locked(self) -> None:
        """Refuse to serve unless this copy is READY. Caller holds the lock.

        Every non-READY state — unverified, recovering, degraded, out of sync —
        is refused here, which is what keeps an unsynchronised copy from being
        used for statistics, scoring or failover.
        """
        if self._state is not NodeState.READY:
            raise EngineNotReadyError(f"engine is {self._state}: {self._detail}")

    @contextmanager
    def _degrade_on_failure(self, reason: str) -> Iterator[None]:
        """Mark the engine degraded if the wrapped block raises. Caller holds the lock.

        Any exception counts. What matters is not which error occurred but that
        the durable write already committed while the derived structures did
        not follow, so they can no longer be trusted.
        """
        try:
            yield
        except Exception as error:
            self._state = NodeState.DEGRADED
            self._detail = reason
            # Reported as a degradation rather than as the original error: the
            # outcome that matters to the caller is that the engine can no
            # longer be trusted, and the request that tripped it should get the
            # same 503 every later request will. The cause is preserved for logs.
            raise EngineNotReadyError(f"engine is degraded: {reason}") from error
