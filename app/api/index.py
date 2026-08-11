"""Index statistics endpoint."""

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.api.dependencies import EngineDep

router = APIRouter(prefix="/index", tags=["index"])


class IndexStatsResponse(BaseModel):
    """Corpus-level statistics.

    These are exactly the quantities BM25 scores with, which is what makes them
    worth exposing: they turn ranking behaviour and the effect of updates and
    deletions into something observable. Posting lists and other internal
    structures are deliberately not exposed.
    """

    document_count: int = Field(description="Number of indexed documents (N).")
    unique_term_count: int = Field(description="Vocabulary size.")
    average_document_length: float = Field(description="Mean document length in tokens (avgdl).")


@router.get("/stats", summary="Report index statistics")
def index_stats(engine: EngineDep) -> IndexStatsResponse:
    """Return corpus statistics for the current index."""
    stats = engine.stats()
    return IndexStatsResponse(
        document_count=stats.document_count,
        unique_term_count=stats.unique_term_count,
        average_document_length=stats.average_document_length,
    )
