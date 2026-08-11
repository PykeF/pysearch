"""The document model."""

from dataclasses import dataclass

from app.search.errors import InvalidDocumentError


@dataclass(frozen=True, slots=True)
class Document:
    """A unit of indexable content.

    Deliberately minimal: an identifier and the text to index. Storage-oriented
    fields (timestamps, versions, shard placement) belong to the phases that
    actually need them.

    Empty text is valid. Such a document produces no tokens, so it can never
    match a query, but it still counts towards the corpus size and can be
    replaced or deleted like any other. Rejecting it would also require
    defining what happens to text that merely normalises to nothing (``"!!!"``),
    which is a distinction without a difference.
    """

    document_id: str
    text: str

    def __post_init__(self) -> None:
        if not self.document_id.strip():
            raise InvalidDocumentError("document_id must not be empty or blank")
