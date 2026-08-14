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


@dataclass(frozen=True, slots=True)
class EngineStatus:
    """Whether the engine can serve requests, and why not if it cannot."""

    ready: bool
    detail: str


class SearchEngine:
    """Indexes documents durably and answers queries over derived state."""

    def __init__(self, store: DocumentStore, scorer: BM25Scorer | None = None) -> None:
        self._store = store
        self._index = InvertedIndex()
        self._documents: dict[str, Document] = {}
        self._scorer = scorer if scorer is not None else BM25Scorer()
        self._lock = threading.Lock()
        self._initialized = False
        self._degraded_reason: str | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> RebuildReport:
        """Rebuild all derived state from storage and make the engine ready.

        Called once at startup, before the application serves traffic, and again
        by anything that wants to repair a degraded engine in place. Recovery is
        never lazy: no request can observe a half-built index.

        Raises:
            StorageError: if the corpus cannot be read.
            IndexInvariantError: if the rebuilt index is inconsistent.
        """
        with self._lock:
            report = self._rebuild_locked()
            self._initialized = True
            self._degraded_reason = None
        return report

    def close(self) -> None:
        """Close the underlying storage and stop serving."""
        with self._lock:
            self._initialized = False
            self._store.close()

    def status(self) -> EngineStatus:
        """Report whether the engine is ready to serve."""
        with self._lock:
            if self._degraded_reason is not None:
                return EngineStatus(ready=False, detail=f"degraded: {self._degraded_reason}")
            if not self._initialized:
                return EngineStatus(ready=False, detail="not initialized")
            return EngineStatus(ready=True, detail="ready")

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
            created = self._store.put(document)

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

            if not self._store.delete(document_id):
                raise DocumentNotFoundError(f"document {document_id!r} is not indexed")

            with self._degrade_on_failure("failed to update derived state after a durable delete"):
                del self._documents[document_id]
                self._index.remove_document(document_id)

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
        return RebuildReport(
            document_count=len(documents),
            duration_seconds=time.perf_counter() - started,
        )

    def _require_operational_locked(self) -> None:
        """Refuse to serve unless initialized and healthy. Caller holds the lock."""
        if self._degraded_reason is not None:
            raise EngineNotReadyError(f"engine is degraded: {self._degraded_reason}")
        if not self._initialized:
            raise EngineNotReadyError("engine is not initialized")

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
            self._degraded_reason = reason
            # Reported as a degradation rather than as the original error: the
            # outcome that matters to the caller is that the engine can no
            # longer be trusted, and the request that tripped it should get the
            # same 503 every later request will. The cause is preserved for logs.
            raise EngineNotReadyError(f"engine is degraded: {reason}") from error
