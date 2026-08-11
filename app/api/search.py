"""Search endpoint."""

from typing import Annotated

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from app.api.dependencies import EngineDep

router = APIRouter(tags=["search"])

MAX_LIMIT = 100
DEFAULT_LIMIT = 10


class SearchHit(BaseModel):
    """One ranked search result."""

    document_id: str
    score: float = Field(description="BM25 score; higher is more relevant.")
    text: str


class SearchResponse(BaseModel):
    """A page of ranked search results."""

    query: str
    total: int = Field(description="Documents matching the query, before the limit is applied.")
    results: list[SearchHit]


@router.get("/search", summary="Search the index")
def search(
    engine: EngineDep,
    q: Annotated[str, Query(description="The query text.")],
    limit: Annotated[
        int, Query(ge=1, le=MAX_LIMIT, description="Maximum results to return.")
    ] = DEFAULT_LIMIT,
) -> SearchResponse:
    """Rank documents against a query using BM25.

    The query is analysed exactly like document text. A query with no terms —
    empty, or only punctuation — returns no results rather than an error.
    """
    outcome = engine.search(q, limit)
    return SearchResponse(
        query=q,
        total=outcome.total,
        results=[
            SearchHit(document_id=hit.document_id, score=hit.score, text=hit.text)
            for hit in outcome.results
        ],
    )
