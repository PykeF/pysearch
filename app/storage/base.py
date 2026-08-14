"""The document store interface.

The engine depends on this protocol rather than on SQLite, which is what lets
tests substitute an in-memory or deliberately failing store without a mocking
framework.

It is intentionally narrow. ``contains`` was considered and left out: its only
caller would be an existence check before deletion, and ``delete`` already
reports whether the document existed — in one atomic statement instead of two
round trips.
"""

from collections.abc import Iterable, Iterator
from typing import Protocol

from app.search.document import Document


class DocumentStore(Protocol):
    """Durable storage for the document corpus.

    Every mutation carries the generation it advances the copy to, and the two
    are committed in one transaction. Persisting them separately would allow a
    crash to leave a document without its generation or the reverse, and the
    whole point of the generation is that it describes exactly which mutations
    a copy has applied.
    """

    def put(self, document: Document, generation: int) -> bool:
        """Store a document, replacing any document with the same id.

        Returns:
            ``True`` if the document is new, ``False`` if it replaced one.
        """
        ...

    def get(self, document_id: str) -> Document | None:
        """Return a document, or ``None`` if it is not stored."""
        ...

    def delete(self, document_id: str, generation: int) -> bool:
        """Remove a document.

        Returns:
            ``True`` if a document was removed, ``False`` if none existed.
        """
        ...

    def replace_all(self, documents: Iterable[Document], generation: int) -> None:
        """Replace the entire corpus in one transaction.

        Used by resynchronisation. Atomicity matters here more than anywhere:
        a resync that failed half way through would leave a copy holding a
        mixture of two corpora while claiming a single generation.
        """
        ...

    def generation(self) -> int:
        """Return the mutation sequence number this copy has applied up to."""
        ...

    def iter_documents(self) -> Iterator[Document]:
        """Yield every stored document, in a stable order."""
        ...

    def count(self) -> int:
        """Return the number of stored documents."""
        ...

    def close(self) -> None:
        """Release the underlying resources."""
        ...
