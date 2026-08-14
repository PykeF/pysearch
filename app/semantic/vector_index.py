"""An exact in-memory vector index.

Search compares the query against every stored vector — no approximation, no
graph, no quantization. That is a deliberate starting point rather than a
shortcut:

* Approximate search can only be *evaluated* against exact search, so an exact
  index has to exist before an approximate one is worth building.
* Delete and replace have to be exact here, because the replication model treats
  a copy's derived state as an exact function of its documents. Approximate
  indexes reach that through tombstones and periodic rebuilds, which would put
  approximation underneath a correctness invariant.
* At this corpus size the whole search is one matrix multiply, measured in
  microseconds. A native library would make that multiply faster without
  changing its complexity.

The cost is stated rather than glossed: query work is O(N·d) and grows linearly
with the corpus. Approximate search is what fixes that, when measurement says it
is time.

Layout
------

``_matrix`` holds one row per document, with rows ``[0, count)`` live::

    _ids        position -> document_id
    _positions  document_id -> position
    _matrix     (capacity, dimension) float32

Deleting swaps the last live row into the freed slot, so removal is O(d) rather
than O(N·d). Ordering of the rows carries no meaning — results are ordered by
score and document id — so the swap is free.

All vectors are expected unit length, so the dot product below *is* cosine
similarity.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from app.semantic.errors import EmbeddingError

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    from numpy.typing import NDArray

_INITIAL_CAPACITY = 64

#: How far a stored vector's length may drift from 1 before the index calls it
#: a violation. Generous enough for float32 accumulation, tight enough that an
#: unnormalized vector cannot slip through.
_NORM_TOLERANCE = 1e-3


@dataclass(frozen=True, slots=True)
class ScoredDocument:
    """One semantic hit: a document and its similarity to the query."""

    document_id: str
    score: float


class VectorIndexError(EmbeddingError):
    """The vector index was asked to hold something it must not."""


class ExactVectorIndex:
    """Maps document identifiers to unit vectors and searches them exhaustively."""

    def __init__(self, dimension: int) -> None:
        if dimension < 1:
            raise ValueError(f"dimension must be at least 1, got {dimension}")
        self._dimension = dimension
        self._ids: list[str] = []
        self._positions: dict[str, int] = {}
        self._matrix: NDArray[np.float32] = np.zeros(
            (_INITIAL_CAPACITY, dimension), dtype=np.float32
        )

    @property
    def dimension(self) -> int:
        """The width every vector in this index must have."""
        return self._dimension

    @property
    def count(self) -> int:
        """How many documents have a vector."""
        return len(self._ids)

    def contains(self, document_id: str) -> bool:
        """Whether this document currently has a vector."""
        return document_id in self._positions

    def vector_for(self, document_id: str) -> "NDArray[np.float32]":
        """Return a copy of one document's vector.

        Raises:
            KeyError: if the document has no vector.
        """
        return np.array(self._matrix[self._positions[document_id]], dtype=np.float32)

    def document_ids(self) -> frozenset[str]:
        """Every document holding a vector."""
        return frozenset(self._positions)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, document_id: str, vector: "NDArray[np.float32]") -> None:
        """Store a document's vector, replacing any vector it already had.

        Replacing in place is what keeps a superseded document from remaining
        semantically findable under its old text.
        """
        checked = self._validate(vector)

        position = self._positions.get(document_id)
        if position is None:
            position = len(self._ids)
            self._grow_if_needed(position + 1)
            self._ids.append(document_id)
            self._positions[document_id] = position

        self._matrix[position] = checked

    def remove(self, document_id: str) -> bool:
        """Remove a document's vector.

        Returns:
            ``True`` if a vector was removed, ``False`` if there was none.
        """
        position = self._positions.pop(document_id, None)
        if position is None:
            return False

        last = len(self._ids) - 1
        if position != last:
            # Move the final live row into the hole so the live rows stay
            # contiguous and search remains one slice.
            self._matrix[position] = self._matrix[last]
            moved = self._ids[last]
            self._ids[position] = moved
            self._positions[moved] = position

        self._ids.pop()
        return True

    def clear(self) -> None:
        """Drop every vector, keeping the configured dimension."""
        self._ids.clear()
        self._positions.clear()
        self._matrix = np.zeros((_INITIAL_CAPACITY, self._dimension), dtype=np.float32)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: "NDArray[np.float32]", limit: int) -> list[ScoredDocument]:
        """Return the ``limit`` most similar documents, best first.

        Cost is O(N·d) for the similarities plus O(N log N) for the ordering.

        Ties break on ascending document id, matching the lexical path, so the
        result is fully determined by the data rather than by insertion order or
        by how NumPy happened to lay the rows out.
        """
        if limit < 1:
            raise ValueError(f"limit must be at least 1, got {limit}")
        if not self._ids:
            return []

        checked = self._validate(query)
        # Unit vectors, so this dot product is cosine similarity.
        scores = self._matrix[: len(self._ids)] @ checked

        ranked = sorted(
            zip(self._ids, scores.tolist(), strict=True),
            key=lambda hit: (-hit[1], hit[0]),
        )
        return [
            ScoredDocument(document_id=document_id, score=float(score))
            for document_id, score in ranked[:limit]
        ]

    def __iter__(self) -> Iterator[str]:
        return iter(self._ids)

    # ------------------------------------------------------------------
    # Invariants
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Check the index's internal consistency.

        A testing and recovery aid, not part of the request path: it touches
        every vector, so running it per query would cost as much as the search.

        Raises:
            VectorIndexError: on the first violation found.
        """
        if len(self._ids) != len(self._positions):
            raise VectorIndexError(f"{len(self._ids)} vectors but {len(self._positions)} positions")

        for position, document_id in enumerate(self._ids):
            if self._positions.get(document_id) != position:
                raise VectorIndexError(
                    f"document {document_id!r} is at row {position} but indexed elsewhere"
                )

        live = self._matrix[: len(self._ids)]
        if live.shape[1] != self._dimension:
            raise VectorIndexError(f"vectors are {live.shape[1]} wide, expected {self._dimension}")
        if live.size and not bool(np.isfinite(live).all()):
            raise VectorIndexError("a stored vector holds non-finite values")

        if live.size:
            norms = np.linalg.norm(live, axis=1)
            # A zero vector is allowed: text that embeds to nothing has no
            # direction and simply never resembles a query.
            drift = np.abs(norms - 1.0)
            offending = np.where((norms != 0.0) & (drift > _NORM_TOLERANCE))[0]
            if offending.size:
                position = int(offending[0])
                raise VectorIndexError(
                    f"vector for {self._ids[position]!r} has length "
                    f"{float(norms[position]):.6f}, expected 1"
                )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _validate(self, vector: "NDArray[np.float32]") -> "NDArray[np.float32]":
        """Check one vector and return it as float32."""
        checked = np.asarray(vector, dtype=np.float32)
        if checked.ndim != 1:
            raise VectorIndexError(f"expected a 1-D vector, got {checked.ndim} dimensions")
        if checked.shape[0] != self._dimension:
            raise VectorIndexError(f"expected dimension {self._dimension}, got {checked.shape[0]}")
        if not bool(np.isfinite(checked).all()):
            raise VectorIndexError("vector holds non-finite values")
        return checked

    def _grow_if_needed(self, required: int) -> None:
        """Double the backing array when it runs out of rows."""
        capacity = self._matrix.shape[0]
        if required <= capacity:
            return

        new_capacity = max(required, capacity * 2)
        grown = np.zeros((new_capacity, self._dimension), dtype=np.float32)
        grown[:capacity] = self._matrix
        self._matrix = grown
