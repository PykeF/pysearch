"""The in-memory inverted index and the corpus statistics BM25 depends on.

Layout
------

The index is a term-to-document mapping, which is what lets a query touch only
the documents that actually contain a query term instead of scanning the whole
corpus::

    "search" -> {"doc-1": 2, "doc-3": 1}
    "index"  -> {"doc-2": 1}

Four structures are maintained, all updated incrementally:

``_postings``          ``term -> document_id -> term frequency``
``_document_terms``    ``document_id -> the unique terms it contains``
``_document_lengths``  ``document_id -> length in tokens``
``_total_length``      running sum of all document lengths

``_postings`` is a mapping rather than a sorted list of postings. That makes
term-frequency lookup, posting insertion and single-document removal all O(1),
and makes document frequency ``len(postings[term])``. The price is giving up
what an ordered posting list buys — skip pointers, delta compression, merge
joins — none of which matters while the index lives in a dict in RAM, and all
of which belongs to the phase that designs an on-disk format.

``_document_terms`` duplicates information already derivable from ``_postings``.
That is deliberate: without it, deleting one document means scanning every term
in the vocabulary, O(V). With it, deletion costs O(unique terms in that
document). Correct, cheap deletion is a Phase 1 requirement, so the memory is
worth spending.

``_total_length`` exists so that ``average_document_length`` is O(1) instead of
a full pass over the corpus on every query.
"""

from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

from app.search.errors import IndexInvariantError

_EMPTY_POSTINGS: Mapping[str, int] = {}


@dataclass(frozen=True, slots=True)
class Posting:
    """One entry of a posting list: a document and its term frequency.

    This is the readable, external view of a posting. The internal
    representation is a plain mapping, and ranking iterates that directly
    rather than allocating one of these per posting per query.
    """

    document_id: str
    term_frequency: int


@dataclass(frozen=True, slots=True)
class IndexStats:
    """A snapshot of corpus-level statistics."""

    document_count: int
    unique_term_count: int
    average_document_length: float
    total_token_count: int


class InvertedIndex:
    """An in-memory inverted index with incrementally maintained statistics."""

    def __init__(self) -> None:
        self._postings: dict[str, dict[str, int]] = {}
        self._document_terms: dict[str, frozenset[str]] = {}
        self._document_lengths: dict[str, int] = {}
        self._total_length = 0

    # ------------------------------------------------------------------
    # Corpus statistics
    # ------------------------------------------------------------------

    @property
    def document_count(self) -> int:
        """``N``: the number of indexed documents."""
        return len(self._document_lengths)

    @property
    def unique_term_count(self) -> int:
        """The vocabulary size."""
        return len(self._postings)

    @property
    def total_token_count(self) -> int:
        """The summed length of every indexed document."""
        return self._total_length

    @property
    def average_document_length(self) -> float:
        """``avgdl``, or ``0.0`` for an empty corpus."""
        if not self._document_lengths:
            return 0.0
        return self._total_length / len(self._document_lengths)

    def stats(self) -> IndexStats:
        """Return the corpus statistics as a snapshot."""
        return IndexStats(
            document_count=self.document_count,
            unique_term_count=self.unique_term_count,
            average_document_length=self.average_document_length,
            total_token_count=self.total_token_count,
        )

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def contains(self, document_id: str) -> bool:
        """Return whether the document is indexed."""
        return document_id in self._document_lengths

    def document_ids(self) -> frozenset[str]:
        """Return the identifiers of every indexed document."""
        return frozenset(self._document_lengths)

    def document_length(self, document_id: str) -> int:
        """``dl``: the document's length in tokens, or ``0`` if unknown."""
        return self._document_lengths.get(document_id, 0)

    def document_frequency(self, term: str) -> int:
        """``df``: the number of documents containing the term."""
        return len(self._postings.get(term, _EMPTY_POSTINGS))

    def term_frequency(self, term: str, document_id: str) -> int:
        """``tf``: occurrences of the term in the document."""
        return self._postings.get(term, _EMPTY_POSTINGS).get(document_id, 0)

    def posting_map(self, term: str) -> Mapping[str, int]:
        """Return the term's posting list as a read-only ``document_id -> tf`` map.

        This is the hot path used by ranking; it allocates nothing.
        """
        return self._postings.get(term, _EMPTY_POSTINGS)

    def postings(self, term: str) -> Iterator[Posting]:
        """Yield the term's posting list as explicit :class:`Posting` values."""
        for document_id, term_frequency in self._postings.get(term, _EMPTY_POSTINGS).items():
            yield Posting(document_id=document_id, term_frequency=term_frequency)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_document(self, document_id: str, tokens: Sequence[str]) -> None:
        """Index a document, replacing it wholesale if the identifier exists.

        Replacement is a removal followed by an insertion, which is what keeps
        terms that disappeared from the new text from lingering in the index.
        """
        if document_id in self._document_lengths:
            self.remove_document(document_id)

        frequencies = Counter(tokens)
        for term, term_frequency in frequencies.items():
            self._postings.setdefault(term, {})[document_id] = term_frequency

        self._document_terms[document_id] = frozenset(frequencies)
        self._document_lengths[document_id] = len(tokens)
        self._total_length += len(tokens)

    def remove_document(self, document_id: str) -> None:
        """Remove a document and every trace of it from the index.

        Raises:
            KeyError: if the document is not indexed.
        """
        if document_id not in self._document_lengths:
            raise KeyError(document_id)

        for term in self._document_terms.pop(document_id):
            postings = self._postings[term]
            del postings[document_id]
            # A term nobody contains is not part of the vocabulary any more.
            if not postings:
                del self._postings[term]

        self._total_length -= self._document_lengths.pop(document_id)

    # ------------------------------------------------------------------
    # Invariants
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Check that the index's structures agree with one another.

        A debugging and testing aid, not part of the request path: the cost is
        proportional to the total number of postings, so running it per request
        would defeat the purpose of having an index at all.

        Raises:
            IndexInvariantError: on the first violation found.
        """
        if set(self._document_terms) != set(self._document_lengths):
            raise IndexInvariantError(
                "document term sets and document lengths cover different documents"
            )

        expected_total = sum(self._document_lengths.values())
        if self._total_length != expected_total:
            raise IndexInvariantError(
                f"total length {self._total_length} does not match the sum of "
                f"document lengths {expected_total}"
            )

        if any(length < 0 for length in self._document_lengths.values()):
            raise IndexInvariantError("a document length is negative")

        # Postings must reference live documents, and must never be empty.
        for term, postings in self._postings.items():
            if not postings:
                raise IndexInvariantError(f"term {term!r} has an empty posting list")
            for document_id, term_frequency in postings.items():
                if document_id not in self._document_lengths:
                    raise IndexInvariantError(
                        f"term {term!r} references unknown document {document_id!r}"
                    )
                if term_frequency < 1:
                    raise IndexInvariantError(
                        f"term {term!r} has term frequency {term_frequency} "
                        f"in document {document_id!r}"
                    )
                if term not in self._document_terms[document_id]:
                    raise IndexInvariantError(
                        f"document {document_id!r} is missing term {term!r} from its term set"
                    )

        # The forward mapping must agree with the postings, and the recorded
        # document length must equal the summed term frequencies.
        for document_id, terms in self._document_terms.items():
            summed = 0
            for term in terms:
                term_postings = self._postings.get(term, _EMPTY_POSTINGS)
                if document_id not in term_postings:
                    raise IndexInvariantError(
                        f"term {term!r} of document {document_id!r} has no posting"
                    )
                summed += term_postings[document_id]
            if summed != self._document_lengths[document_id]:
                raise IndexInvariantError(
                    f"document {document_id!r} has length "
                    f"{self._document_lengths[document_id]} but its term frequencies sum "
                    f"to {summed}"
                )
