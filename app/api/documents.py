"""Document indexing and deletion endpoints."""

from fastapi import APIRouter, Response, status
from pydantic import BaseModel, Field

from app.api.dependencies import EngineDep
from app.search.document import Document

router = APIRouter(prefix="/documents", tags=["documents"])


class IndexDocumentRequest(BaseModel):
    """Body of an indexing request."""

    text: str = Field(description="The document text to analyse and index.")


class IndexDocumentResponse(BaseModel):
    """Result of an indexing request."""

    document_id: str
    created: bool = Field(description="True if the document is new, false if it was replaced.")


@router.put(
    "/{document_id}",
    summary="Index or replace a document",
    responses={
        status.HTTP_200_OK: {"description": "The existing document was replaced."},
        status.HTTP_201_CREATED: {"description": "A new document was indexed."},
        status.HTTP_400_BAD_REQUEST: {"description": "The document identifier is blank."},
    },
)
def index_document(
    document_id: str,
    payload: IndexDocumentRequest,
    engine: EngineDep,
    response: Response,
) -> IndexDocumentResponse:
    """Index a document, replacing any existing document with the same id.

    Replacement is a full reindex: terms that are no longer present disappear
    from the index, and the corpus statistics are updated accordingly.
    """
    document = Document(document_id=document_id, text=payload.text)
    created = engine.index_document(document)
    response.status_code = status.HTTP_201_CREATED if created else status.HTTP_200_OK
    return IndexDocumentResponse(document_id=document_id, created=created)


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document",
    responses={
        status.HTTP_204_NO_CONTENT: {"description": "The document was deleted."},
        status.HTTP_404_NOT_FOUND: {"description": "No such document is indexed."},
    },
)
def delete_document(document_id: str, engine: EngineDep) -> None:
    """Delete a document and remove it from the index."""
    engine.delete_document(document_id)
