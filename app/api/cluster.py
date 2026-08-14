"""Public coordinator endpoints.

The paths and payloads are the single-node ones, so a client cannot tell
whether it is talking to one node or to a cluster, and never has to know which
shard owns a document. The one deliberate difference is ``/index/stats``, which
reports cluster-level figures.
"""

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query, Request, Response, status
from pydantic import BaseModel, Field

from app.api.documents import IndexDocumentRequest
from app.api.search import DEFAULT_LIMIT, MAX_LIMIT, SearchHit, SearchResponse
from app.api.semantic import SemanticSearchResponse
from app.cluster.coordinator import Coordinator
from app.search.document import Document

router = APIRouter(tags=["cluster"])


def get_coordinator(request: Request) -> Coordinator:
    """Return the coordinator attached to the running application."""
    coordinator: Coordinator = request.app.state.coordinator
    return coordinator


CoordinatorDep = Annotated[Coordinator, Depends(get_coordinator)]


class IndexDocumentResponse(BaseModel):
    """Result of an indexing request, plus where the document landed."""

    document_id: str
    created: bool
    shard_id: int = Field(description="The shard that owns this document.")


class ShardStatsResponse(BaseModel):
    """One shard's contribution to the cluster statistics."""

    shard_id: int
    document_count: int
    unique_term_count: int
    average_document_length: float


class ClusterStatsResponse(BaseModel):
    """Cluster-wide index statistics.

    There is no cluster-wide ``unique_term_count``: vocabulary sizes cannot be
    summed, because the same term legitimately appears on several shards, and
    the true union would mean transferring every shard's vocabulary. Reporting a
    sum would be arithmetically false, so per-shard figures are given instead.
    """

    document_count: int
    average_document_length: float
    shard_count: int
    shards: list[ShardStatsResponse]


class ReadinessResponse(BaseModel):
    """Whether the cluster can honour its search contract."""

    status: Literal["ready", "not_ready"]
    detail: str


@router.put(
    "/documents/{document_id}",
    summary="Index or replace a document",
    responses={
        status.HTTP_200_OK: {"description": "The existing document was replaced."},
        status.HTTP_201_CREATED: {"description": "A new document was indexed."},
        status.HTTP_400_BAD_REQUEST: {"description": "The document identifier is blank."},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "The owning shard is unavailable."},
    },
)
async def index_document(
    document_id: str,
    payload: IndexDocumentRequest,
    coordinator: CoordinatorDep,
    response: Response,
) -> IndexDocumentResponse:
    """Route a document to its owning shard and store it durably there.

    Exactly one shard receives the write, and a failed write is never retried
    elsewhere: rerouting would break the ownership that routing depends on.
    """
    document = Document(document_id=document_id, text=payload.text)
    created = await coordinator.index_document(document)
    response.status_code = status.HTTP_201_CREATED if created else status.HTTP_200_OK
    return IndexDocumentResponse(
        document_id=document_id,
        created=created,
        shard_id=coordinator.shard_for(document_id),
    )


@router.delete(
    "/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document",
    responses={
        status.HTTP_204_NO_CONTENT: {"description": "The document was deleted."},
        status.HTTP_404_NOT_FOUND: {"description": "No such document is indexed."},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "The owning shard is unavailable."},
    },
)
async def delete_document(document_id: str, coordinator: CoordinatorDep) -> None:
    """Delete a document from the shard that owns it."""
    await coordinator.delete_document(document_id)


@router.get(
    "/search",
    summary="Search the cluster",
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "A shard failed, so the results would be incomplete."
        }
    },
)
async def search(
    coordinator: CoordinatorDep,
    q: Annotated[str, Query(description="The query text.")],
    limit: Annotated[
        int, Query(ge=1, le=MAX_LIMIT, description="Maximum results to return.")
    ] = DEFAULT_LIMIT,
) -> SearchResponse:
    """Rank documents across every shard using BM25.

    Shards score with cluster-wide statistics, so their scores are comparable
    and the merged ranking is the same one an equivalent single node would
    produce. If any shard fails the whole query fails, rather than returning a
    result set that silently omits part of the corpus.
    """
    outcome = await coordinator.search(q, limit)
    return SearchResponse(
        query=q,
        total=outcome.total,
        results=[
            SearchHit(document_id=hit.document_id, score=hit.score, text=hit.text)
            for hit in outcome.results
        ],
    )


@router.get(
    "/search/semantic",
    summary="Search the cluster by meaning rather than by words",
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "Semantic search is disabled, or a logical shard has no usable copy."
        }
    },
)
async def semantic_search(
    coordinator: CoordinatorDep,
    q: Annotated[str, Query(description="The query text.")],
    limit: Annotated[
        int, Query(ge=1, le=MAX_LIMIT, description="Maximum results to return.")
    ] = DEFAULT_LIMIT,
) -> SemanticSearchResponse:
    """Rank documents across every shard by embedding similarity.

    One fan-out round rather than two: a cosine similarity depends only on the
    query and document vectors, so — unlike BM25 — no corpus-wide statistics
    have to be gathered before the shards can produce comparable scores.

    Scores lie in [-1, 1] and are **not** comparable with BM25 scores.
    """
    outcome = await coordinator.semantic_search(q, limit)
    return SemanticSearchResponse(
        query=q,
        total=outcome.total,
        results=[
            SearchHit(document_id=hit.document_id, score=hit.score, text=hit.text)
            for hit in outcome.results
        ],
    )


@router.get("/index/stats", summary="Report cluster index statistics")
async def index_stats(coordinator: CoordinatorDep) -> ClusterStatsResponse:
    """Aggregate index statistics across the cluster."""
    stats = await coordinator.index_stats()
    return ClusterStatsResponse(
        document_count=stats.document_count,
        average_document_length=stats.average_document_length,
        shard_count=stats.shard_count,
        shards=[
            ShardStatsResponse(
                shard_id=shard.shard_id,
                document_count=shard.document_count,
                unique_term_count=shard.unique_term_count,
                average_document_length=shard.average_document_length,
            )
            for shard in stats.shards
        ],
    )


@router.get(
    "/ready",
    summary="Report whether the cluster can serve",
    responses={
        status.HTTP_200_OK: {"description": "Every shard is ready."},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "At least one shard is not ready."},
    },
)
async def ready(coordinator: CoordinatorDep, response: Response) -> ReadinessResponse:
    """Report readiness, which requires every shard.

    All-or-nothing, to match the search policy: a cluster missing a shard cannot
    answer the queries it advertises, so it must not claim it can.
    """
    readiness = await coordinator.readiness()
    if not readiness.ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadinessResponse(status="not_ready", detail=readiness.detail)
    return ReadinessResponse(status="ready", detail=readiness.detail)


class CopyStatusResponse(BaseModel):
    """One physical copy of a logical shard."""

    role: str
    reachable: bool
    node_id: str
    state: str
    ready: bool
    generation: int | None


class ShardStatusResponse(BaseModel):
    """What one logical shard can currently do."""

    shard_id: int
    search_available: bool
    write_available: bool
    copies: list[CopyStatusResponse]


class SemanticStatusResponse(BaseModel):
    """Whether the cluster can answer semantic queries, and in which space."""

    enabled: bool
    identity: str | None = None


class ClusterStatusResponse(BaseModel):
    """Cluster capability, shard by shard.

    Search and write availability are reported separately because losing a
    primary leaves its logical shard readable through a replica but not
    writable: nothing is promoted automatically, by design.
    """

    shard_count: int
    replication_factor: int
    search_available: bool
    write_available: bool
    semantic: SemanticStatusResponse
    shards: list[ShardStatusResponse]


@router.get("/cluster/status", summary="Report cluster topology and capability")
async def cluster_status(coordinator: CoordinatorDep) -> ClusterStatusResponse:
    """Report each logical shard's copies, and what the cluster can currently do."""
    health = await coordinator.cluster_status()
    return ClusterStatusResponse(
        shard_count=health.shard_count,
        replication_factor=health.replication_factor,
        search_available=health.search_available,
        write_available=health.write_available,
        semantic=SemanticStatusResponse(
            enabled=coordinator.semantic_enabled,
            identity=(
                coordinator.semantic_identity.fingerprint if coordinator.semantic_identity else None
            ),
        ),
        shards=[
            ShardStatusResponse(
                shard_id=shard.shard_id,
                search_available=shard.search_available,
                write_available=shard.write_available,
                copies=[
                    CopyStatusResponse(
                        role=copy.role,
                        reachable=copy.reachable,
                        node_id=copy.node_id,
                        state=copy.state,
                        ready=copy.ready,
                        generation=copy.generation,
                    )
                    for copy in shard.copies
                ],
            )
            for shard in health.shards
        ],
    )
