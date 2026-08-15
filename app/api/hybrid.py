"""Hybrid search: the lexical and semantic rankings, fused.

``/search`` and ``/search/semantic`` are untouched. This is a third endpoint,
because a hybrid ranking answers a different question from either input and its
score lives on a different scale again.
"""

from typing import Annotated

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from app.api.dependencies import EngineDep
from app.api.search import DEFAULT_LIMIT, MAX_LIMIT
from app.hybrid.fusion import HybridResults

router = APIRouter(tags=["hybrid"])


class HybridHitModel(BaseModel):
    """One fused result.

    The two ranks are always present because together they explain the score
    exactly: a document at lexical rank 2 and semantic rank 1 scores
    ``1/(k+2) + 1/(k+1)``. ``null`` means the document was not in that list at
    all — which is normal, not a penalty.

    The underlying scores are filled in only when ``explain=true``; they answer
    the different question of why the ranks fell where they did.
    """

    document_id: str
    score: float = Field(description="Fusion score: a sum of reciprocal ranks.")
    text: str
    lexical_rank: int | None
    semantic_rank: int | None
    lexical_score: float | None = None
    semantic_score: float | None = None


class HybridSearchResponse(BaseModel):
    """A page of fused results.

    ``total`` is the **candidate union size** — how many distinct documents
    entered fusion, bounded by twice the candidate depth. Each of the three
    search endpoints means something different by ``total``: documents
    containing a query term for ``/search``, documents searched for
    ``/search/semantic``, and candidates fused here.
    """

    query: str
    total: int = Field(description="Distinct documents that entered fusion.")
    results: list[HybridHitModel]


def to_response(query: str, outcome: HybridResults, explain: bool) -> HybridSearchResponse:
    """Render a fused ranking, including the underlying scores only on request."""
    return HybridSearchResponse(
        query=query,
        total=outcome.total,
        results=[
            HybridHitModel(
                document_id=hit.document_id,
                score=hit.score,
                text=hit.text,
                lexical_rank=hit.lexical_rank,
                semantic_rank=hit.semantic_rank,
                lexical_score=hit.lexical_score if explain else None,
                semantic_score=hit.semantic_score if explain else None,
            )
            for hit in outcome.results
        ],
    )


@router.get("/search/hybrid", summary="Search lexically and semantically, then fuse")
def hybrid_search(
    engine: EngineDep,
    q: Annotated[str, Query(description="The query text.")],
    limit: Annotated[
        int, Query(ge=1, le=MAX_LIMIT, description="Maximum results to return.")
    ] = DEFAULT_LIMIT,
    explain: Annotated[
        bool, Query(description="Include the underlying lexical and semantic scores.")
    ] = False,
) -> HybridSearchResponse:
    """Combine BM25 and vector rankings with Reciprocal Rank Fusion.

    Returns 503 when semantic search is not enabled: a "hybrid" result computed
    from one retrieval path would be a lie about what produced it.
    """
    return to_response(q, engine.hybrid_search(q, limit), explain)
