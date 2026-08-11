"""The search engine: the seam between the HTTP layer and retrieval internals.

The engine owns the two pieces of mutable state — the document store and the
inverted index — and is responsible for keeping them consistent with each
other. Everything below it (analysis, index, ranking) is pure or self-contained.

Concurrency
-----------

FastAPI runs synchronous path operations in a thread pool, so several requests
genuinely execute at the same time against this state. Without protection, two
concurrent writes interleave their read-modify-write of the running token total
and lose an update, and a query iterating a posting list while a delete mutates
it fails outright.

Phase 1 takes the simplest correct approach: one lock, held across every read
and every write. Queries are therefore serialised with each other, which costs
read throughput and buys an index that is never observed half-updated.
Correctness first; a reader-writer scheme or copy-on-write snapshot is a later
optimisation with evidence behind it.

The alternative — making the handlers ``async`` and relying on the event loop
to make each one atomic — needs no lock at all, but it is fragile in a way that
does not announce itself: the moment anyone adds an ``await`` inside a critical
section, the invariant silently disappears. It also blocks the loop for the
duration of the CPU-bound work.
"""

import threading
from dataclasses import dataclass

from app.search.analysis import analyze
from app.search.document import Document
from app.search.errors import DocumentNotFoundError, IndexInvariantError
from app.search.index import IndexStats, InvertedIndex
from app.search.ranking import BM25Scorer


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


class SearchEngine:
    """Indexes documents and answers queries over them."""

    def __init__(self, scorer: BM25Scorer | None = None) -> None:
        self._index = InvertedIndex()
        self._documents: dict[str, Document] = {}
        self._scorer = scorer if scorer is not None else BM25Scorer()
        self._lock = threading.Lock()

    def index_document(self, document: Document) -> bool:
        """Index a document, replacing any existing document with the same id.

        Returns:
            ``True`` if the document is new, ``False`` if it replaced one.
        """
        # Analysis is pure and touches no shared state, so it stays outside the
        # critical section.
        tokens = analyze(document.text)

        with self._lock:
            created = document.document_id not in self._documents
            self._documents[document.document_id] = document
            self._index.add_document(document.document_id, tokens)
            return created

    def delete_document(self, document_id: str) -> None:
        """Remove a document and its index entries.

        Raises:
            DocumentNotFoundError: if the document is not indexed.
        """
        with self._lock:
            if document_id not in self._documents:
                raise DocumentNotFoundError(f"document {document_id!r} is not indexed")
            del self._documents[document_id]
            self._index.remove_document(document_id)

    def search(self, query: str, limit: int) -> SearchResults:
        """Return the top ``limit`` documents for ``query``, best first.

        The query runs through the same analysis pipeline as documents. A query
        that yields no terms — because it was empty, or was nothing but
        punctuation — matches nothing; that is an empty result, not an error.

        Results are ordered by descending score, then by ascending document id.
        The second key is what makes ties reproducible instead of dependent on
        dictionary iteration order.
        """
        terms = analyze(query)
        if not terms:
            return SearchResults(total=0, results=())

        with self._lock:
            scores = self._scorer.score_query(self._index, terms)
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

    def stats(self) -> IndexStats:
        """Return a snapshot of the corpus statistics."""
        with self._lock:
            return self._index.stats()

    def validate(self) -> None:
        """Check that the document store and the index agree.

        A testing and debugging aid; see :meth:`InvertedIndex.validate`.

        Raises:
            IndexInvariantError: on the first violation found.
        """
        with self._lock:
            self._index.validate()
            if set(self._documents) != self._index.document_ids():
                raise IndexInvariantError(
                    "the document store and the index hold different documents"
                )
