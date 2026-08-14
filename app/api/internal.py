"""Internal shard endpoints, used by the coordinator.

These are an implementation interface between nodes, not a public API. In
particular they are **not** a supported external write path: the coordinator's
operation lock is what keeps a distributed query's statistics and its scored
documents in the same logical state, and writing here directly steps around it.

Every route delegates to the same :class:`SearchEngine` a single node uses.
There is no second search implementation for distributed mode.
"""

from typing import Annotated

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Response, status
from pydantic import BaseModel, Field

from app.api.dependencies import EngineDep, PrimaryDep, ReplicaDep, SettingsDep
from app.search.document import Document
from app.search.engine import ReplicationOutcome
from app.search.index import CorpusStats
from app.semantic.embedder import SemanticIdentity

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


@router.put(
    "/documents/{document_id}",
    summary="Index or replace a document on this logical shard",
    responses={status.HTTP_409_CONFLICT: {"description": "This node is not the primary."}},
)
def index_document(
    document_id: str,
    payload: IndexDocumentRequest,
    writer: PrimaryDep,
    response: Response,
) -> IndexDocumentResponse:
    """Store a document durably here and on every replica before answering.

    Only the primary accepts this. Success means the document is durable on
    every configured copy; an error may still mean it is durable here, and the
    caller is expected to retry rather than assume nothing happened.
    """
    created = writer.index_document(Document(document_id=document_id, text=payload.text))
    response.status_code = status.HTTP_201_CREATED if created else status.HTTP_200_OK
    return IndexDocumentResponse(document_id=document_id, created=created)


@router.delete(
    "/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document from this logical shard",
    responses={status.HTTP_409_CONFLICT: {"description": "This node is not the primary."}},
)
def delete_document(document_id: str, writer: PrimaryDep) -> None:
    """Remove a document here and on every replica before answering."""
    writer.delete_document(document_id)


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


class ReplicateDocumentRequest(BaseModel):
    """A mutation being replicated from the primary."""

    text: str
    generation: int = Field(ge=1, description="The contiguous sequence number to advance to.")


class ReplicationResponse(BaseModel):
    """What the replica did with a replicated mutation."""

    outcome: str
    generation: int


class NodeStatusResponse(BaseModel):
    """This node's identity and serving state."""

    node_id: str
    shard_id: int
    replica_role: str
    state: str
    ready: bool
    generation: int
    semantic_enabled: bool = False
    semantic_identity: str | None = None
    vector_count: int | None = None


class ExportedDocument(BaseModel):
    """One document in a resynchronization snapshot."""

    document_id: str
    text: str


class ExportResponse(BaseModel):
    """A consistent snapshot of a logical shard's corpus."""

    generation: int
    documents: list[ExportedDocument]


def _replication_response(outcome: ReplicationOutcome, engine: EngineDep) -> ReplicationResponse:
    """Turn a replication outcome into a response, refusing on a gap.

    A gap is reported as 409 rather than a success: the replica has refused the
    mutation and taken itself out of service, and the primary must not read that
    as an acknowledgement.
    """
    if outcome is ReplicationOutcome.GAP:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="generation gap; this replica needs resynchronization",
        )
    return ReplicationResponse(outcome=str(outcome), generation=engine.generation)


@router.put(
    "/replicate/{document_id}",
    summary="Apply a replicated write",
    responses={status.HTTP_409_CONFLICT: {"description": "Not a replica, or a generation gap."}},
)
def replicate_document(
    document_id: str,
    payload: ReplicateDocumentRequest,
    engine: EngineDep,
    synchronizer: ReplicaDep,
) -> ReplicationResponse:
    """Apply a write replicated from this logical shard's primary."""
    del synchronizer  # required only to refuse this call on a non-replica
    outcome = engine.apply_replicated_put(
        Document(document_id=document_id, text=payload.text), payload.generation
    )
    return _replication_response(outcome, engine)


@router.delete(
    "/replicate/{document_id}",
    summary="Apply a replicated deletion",
    responses={status.HTTP_409_CONFLICT: {"description": "Not a replica, or a generation gap."}},
)
def replicate_delete(
    document_id: str,
    engine: EngineDep,
    synchronizer: ReplicaDep,
    generation: Annotated[int, Query(ge=1)],
) -> ReplicationResponse:
    """Apply a deletion replicated from this logical shard's primary."""
    del synchronizer
    outcome = engine.apply_replicated_delete(document_id, generation)
    return _replication_response(outcome, engine)


@router.get("/export", summary="Export this logical shard's corpus")
def export_documents(engine: EngineDep, writer: PrimaryDep) -> ExportResponse:
    """Return a consistent snapshot for a replica to resynchronize from.

    Served by the primary only, because the primary is always the most advanced
    copy — recovering from anywhere else could move a replica backwards.
    """
    del writer  # required only to refuse this call on a non-primary
    documents, generation = engine.export_snapshot()
    return ExportResponse(
        generation=generation,
        documents=[
            ExportedDocument(document_id=document.document_id, text=document.text)
            for document in documents
        ],
    )


@router.get("/node-status", summary="Report this node's identity and serving state")
def node_status(engine: EngineDep, settings: SettingsDep) -> NodeStatusResponse:
    """Report role, state and generation, for cluster status and synchronization."""
    engine_status = engine.status()
    return NodeStatusResponse(
        node_id=settings.node_id or f"shard-{settings.shard_id}-{settings.replica_role}",
        shard_id=settings.shard_id if settings.shard_id is not None else -1,
        replica_role=settings.replica_role or "single",
        state=str(engine_status.state),
        ready=engine_status.ready,
        generation=engine_status.generation,
        semantic_enabled=engine.semantic_enabled,
        semantic_identity=(
            engine.semantic_identity.fingerprint if engine.semantic_identity else None
        ),
        vector_count=engine.vector_count,
    )


@router.post("/replica/resync", summary="Resynchronize this replica from its primary")
def resynchronize(synchronizer: ReplicaDep) -> NodeStatusResponse | ReplicationResponse:
    """Pull a fresh snapshot from the primary and return to service if it succeeds."""
    outcome = synchronizer.resynchronize()
    if not outcome.synchronized:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=outcome.detail)
    return ReplicationResponse(outcome="resynchronized", generation=outcome.local_generation)


class SemanticIdentityModel(BaseModel):
    """The embedding space a semantic request belongs to."""

    implementation: str
    model_id: str
    model_revision: str
    dimension: int
    normalization: str

    def to_identity(self) -> SemanticIdentity:
        return SemanticIdentity(
            implementation=self.implementation,
            model_id=self.model_id,
            model_revision=self.model_revision,
            dimension=self.dimension,
            normalization=self.normalization,
        )


class InternalSemanticSearchRequest(BaseModel):
    """A query vector to rank this shard's documents against."""

    vector: list[float]
    limit: int = Field(ge=1, le=MAX_LIMIT)
    identity: SemanticIdentityModel


@router.post(
    "/search/semantic",
    summary="Search this shard against a query vector",
    responses={status.HTTP_409_CONFLICT: {"description": "A different embedding model is in use."}},
)
def semantic_search(
    payload: InternalSemanticSearchRequest, engine: EngineDep
) -> InternalSearchResponse:
    """Rank this shard's documents against a vector the coordinator embedded.

    The identity is checked first. Vectors from different models measure
    different spaces, so a mismatch is refused rather than answered with numbers
    that would look like similarities and mean nothing.
    """
    local = engine.semantic_identity
    if local is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="semantic search is not enabled on this node",
        )

    incoming = payload.identity.to_identity()
    if not local.is_compatible_with(incoming):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"semantic identity mismatch: this node uses {local.fingerprint}, "
                f"the request carries {incoming.fingerprint}"
            ),
        )

    outcome = engine.semantic_search(np.asarray(payload.vector, dtype=np.float32), payload.limit)
    return InternalSearchResponse(
        total=outcome.total,
        results=[
            InternalHit(document_id=hit.document_id, score=hit.score, text=hit.text)
            for hit in outcome.results
        ],
    )
