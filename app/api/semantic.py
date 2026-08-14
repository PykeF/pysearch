"""Semantic search on a single node.

``GET /search`` is untouched and remains BM25. This is a separate path, because
the two retrievals answer different questions and their scores are not on the
same scale — reconciling them is a later phase's problem, not something to
smuggle into an existing endpoint.
"""

from typing import Annotated

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from app.api.dependencies import EngineDep
from app.api.search import DEFAULT_LIMIT, MAX_LIMIT, SearchHit

router = APIRouter(tags=["semantic"])


class SemanticSearchResponse(BaseModel):
    """A page of semantically ranked results.

    ``total`` means something different here than it does for lexical search.
    BM25 counts the documents containing a query term; a vector similarity is
    defined for *every* document, so there is no such thing as not matching.
    ``total`` is therefore the number of documents searched, and the ranking
    is the interesting part.
    """

    query: str
    total: int = Field(description="Documents searched, not documents matched.")
    results: list[SearchHit]


@router.get("/search/semantic", summary="Search by meaning rather than by words")
def semantic_search(
    engine: EngineDep,
    q: Annotated[str, Query(description="The query text.")],
    limit: Annotated[
        int, Query(ge=1, le=MAX_LIMIT, description="Maximum results to return.")
    ] = DEFAULT_LIMIT,
) -> SemanticSearchResponse:
    """Rank documents by cosine similarity between their embeddings and the query's.

    Scores lie in [-1, 1] and are **not** comparable with BM25 scores.
    """
    outcome = engine.semantic_search(engine.embed_query(q), limit)
    return SemanticSearchResponse(
        query=q,
        total=outcome.total,
        results=[
            SearchHit(document_id=hit.document_id, score=hit.score, text=hit.text)
            for hit in outcome.results
        ],
    )
