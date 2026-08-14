"""The document store interface.

The engine depends on this protocol rather than on SQLite, which is what lets
tests substitute an in-memory or deliberately failing store without a mocking
framework.

It is intentionally narrow. ``contains`` was considered and left out: its only
caller would be an existence check before deletion, and ``delete`` already
reports whether the document existed — in one atomic statement instead of two
round trips.
"""

from collections.abc import Iterator
from typing import Protocol

from app.search.document import Document


class DocumentStore(Protocol):
    """Durable storage for the document corpus."""

    def put(self, document: Document) -> bool:
        """Store a document, replacing any document with the same id.

        Returns:
            ``True`` if the document is new, ``False`` if it replaced one.
        """
        ...

    def get(self, document_id: str) -> Document | None:
        """Return a document, or ``None`` if it is not stored."""
        ...

    def delete(self, document_id: str) -> bool:
        """Remove a document.

        Returns:
            ``True`` if a document was removed, ``False`` if none existed.
        """
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
