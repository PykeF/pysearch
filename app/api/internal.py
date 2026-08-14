"""Internal shard endpoints, used by the coordinator.

These are an implementation interface between nodes, not a public API. In
particular they are **not** a supported external write path: the coordinator's
operation lock is what keeps a distributed query's statistics and its scored
documents in the same logical state, and writing here directly steps around it.

Every route delegates to the same :class:`SearchEngine` a single node uses.
There is no second search implementation for distributed mode.
"""

from typing import Annotated

from fastapi import APIRouter, Query, Response, status
from pydantic import BaseModel, Field

from app.api.dependencies import EngineDep
from app.search.document import Document
from app.search.index import CorpusStats

router = APIRouter(prefix="/internal", tags=["internal"])

MAX_LIMIT = 100


class IndexDocumentRequest(BaseModel):
    """Body of an internal indexing request."""

    text: str


class IndexDocumentResponse(BaseModel):
    """Result of an internal indexing request."""

    document_id: str
    created: bool


class CorpusStatsModel(BaseModel):
    """The BM25 inputs for a set of terms, on the wire."""

    document_count: int
    total_document_length: int
    document_frequencies: dict[str, int]

    def to_corpus_stats(self) -> CorpusStats:
        return CorpusStats(
            document_count=self.document_count,
            total_document_length=self.total_document_length,
            document_frequencies=self.document_frequencies,
        )


class InternalSearchRequest(BaseModel):
    """A shard-local search, scored with cluster-wide statistics."""

    query: str
    limit: int = Field(ge=1, le=MAX_LIMIT)
    corpus_stats: CorpusStatsModel


class InternalHit(BaseModel):
    """One shard-local hit."""

    document_id: str
    score: float
    text: str


class InternalSearchResponse(BaseModel):
    """A shard's local top-k and its total match count."""

    total: int
    results: list[InternalHit]


class InternalIndexStatsResponse(BaseModel):
    """A shard's local index statistics."""

    document_count: int
    unique_term_count: int
    average_document_length: float
    total_token_count: int


@router.put("/documents/{document_id}", summary="Index or replace a document on this shard")
def index_document(
    document_id: str,
    payload: IndexDocumentRequest,
    engine: EngineDep,
    response: Response,
) -> IndexDocumentResponse:
    """Store a document that routing has already assigned to this shard."""
    created = engine.index_document(Document(document_id=document_id, text=payload.text))
    response.status_code = status.HTTP_201_CREATED if created else status.HTTP_200_OK
    return IndexDocumentResponse(document_id=document_id, created=created)


@router.delete(
    "/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document from this shard",
)
def delete_document(document_id: str, engine: EngineDep) -> None:
    """Remove a document this shard owns."""
    engine.delete_document(document_id)


@router.post("/search", summary="Search this shard using cluster-wide statistics")
def search(payload: InternalSearchRequest, engine: EngineDep) -> InternalSearchResponse:
    """Return this shard's local top-k, scored with the supplied statistics.

    The raw query is analysed here by the same pipeline the coordinator used to
    decide which terms to collect statistics for, so the terms scored and the
    terms measured are the same ones.
    """
    outcome = engine.search(payload.query, payload.limit, payload.corpus_stats.to_corpus_stats())
    return InternalSearchResponse(
        total=outcome.total,
        results=[
            InternalHit(document_id=hit.document_id, score=hit.score, text=hit.text)
            for hit in outcome.results
        ],
    )


@router.get("/corpus-stats", summary="Report this shard's BM25 inputs for some terms")
def corpus_stats(
    engine: EngineDep,
    term: Annotated[list[str] | None, Query(description="Analysed query terms.")] = None,
) -> CorpusStatsModel:
    """Return N, the summed document length, and df for each requested term."""
    stats = engine.corpus_stats(term or [])
    return CorpusStatsModel(
        document_count=stats.document_count,
        total_document_length=stats.total_document_length,
        document_frequencies=dict(stats.document_frequencies),
    )


@router.get("/stats", summary="Report this shard's local index statistics")
def index_stats(engine: EngineDep) -> InternalIndexStatsResponse:
    """Return the local index statistics used to build cluster-level ones."""
    stats = engine.stats()
    return InternalIndexStatsResponse(
        document_count=stats.document_count,
        unique_term_count=stats.unique_term_count,
        average_document_length=stats.average_document_length,
        total_token_count=stats.total_token_count,
    )
