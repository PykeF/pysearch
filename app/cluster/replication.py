"""Replication between the copies of one logical shard.

Model
-----

Every logical shard has one statically configured primary and zero or more
replicas. A write is acknowledged only once **every** copy has durably
committed it — read-one, write-all. That is what makes any READY copy
interchangeable for reads, which is in turn what makes read failover safe
without weakening the ranking guarantee: a failover target scored against
different corpus statistics would silently produce a different ranking.

The price is stated plainly: if any replica is unavailable, writes to that
logical shard fail.

Ordering
--------

The primary commits locally first, then replicates::

    primary durable commit (generation N+1)
        -> replica applies (generation N+1)
            -> acknowledge

Primary-first is deliberate. It guarantees ``generation(primary) >=
generation(replica)`` at all times, so the primary is always the most advanced
copy — always the correct recovery source, with no question about which copy is
newer. Replica-first would let a replica run ahead, and resynchronising it from
the primary would then destroy data.

The consequence is that a failed replication leaves the mutation committed on
the primary and absent from the replica. That write is reported as **failed**,
so no guarantee is broken; the replica is behind by an unacknowledged mutation
and the next replicated mutation will expose the gap.

Idempotency
-----------

Generations are a contiguous per-shard sequence, so a replica accepts exactly
``local + 1``, treats anything at or below its own as an already-applied
redelivery, and refuses anything beyond as a gap. Retries are therefore safe
without idempotency keys.
"""

import logging
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import httpx2

from app.cluster.errors import (
    ReplicationError,
    ShardTimeoutError,
    ShardUnavailableError,
)
from app.search.document import Document
from app.search.engine import RebuildReport, SearchEngine

logger = logging.getLogger(__name__)


class ReplicationTarget(Protocol):
    """One replica, as its primary sees it."""

    def apply_put(self, document: Document, generation: int) -> None:
        """Apply a replicated write, raising if it cannot be applied."""
        ...

    def apply_delete(self, document_id: str, generation: int) -> None:
        """Apply a replicated deletion, raising if it cannot be applied."""
        ...


class PrimarySource(Protocol):
    """A primary, as its replica sees it during synchronisation."""

    def generation(self) -> int:
        """Return the primary's current generation."""
        ...

    def export(self) -> tuple[Sequence[Document], int]:
        """Return a consistent snapshot of the corpus and its generation."""
        ...


class NodeLink(ReplicationTarget, PrimarySource, Protocol):
    """Both halves of the peer protocol, as one connection to another node."""


@dataclass(frozen=True, slots=True)
class SyncOutcome:
    """What a replica did when it checked itself against its primary."""

    synchronized: bool
    detail: str
    local_generation: int
    primary_generation: int | None


class Replicator:
    """The primary's write path: commit locally, then replicate to every replica."""

    def __init__(self, engine: SearchEngine, targets: Sequence[ReplicationTarget]) -> None:
        self._engine = engine
        self._targets = tuple(targets)
        # Held across commit-and-replicate so two mutations can never reach a
        # replica out of order. Out-of-order delivery would look like a gap and
        # take an otherwise healthy replica out of service.
        self._lock = threading.Lock()

    @property
    def replica_count(self) -> int:
        """How many replicas every write must reach."""
        return len(self._targets)

    def index_document(self, document: Document) -> bool:
        """Store a document on the primary and on every replica.

        Returns:
            ``True`` if the document is new, ``False`` if it replaced one.

        Raises:
            ReplicationError: if any replica failed to apply the mutation. The
                write is durable on the primary but is **not** acknowledged.
        """
        with self._lock:
            created = self._engine.index_document(document)
            generation = self._engine.generation
            self._replicate(
                lambda target: target.apply_put(document, generation),
                document.document_id,
                generation,
            )
            return created

    def delete_document(self, document_id: str) -> None:
        """Delete a document on the primary and on every replica.

        Raises:
            DocumentNotFoundError: if the primary does not hold the document,
                in which case nothing is replicated.
            ReplicationError: if any replica failed to apply the deletion.
        """
        with self._lock:
            self._engine.delete_document(document_id)
            generation = self._engine.generation
            self._replicate(
                lambda target: target.apply_delete(document_id, generation),
                document_id,
                generation,
            )

    def _replicate(
        self,
        apply: Callable[[ReplicationTarget], None],
        document_id: str,
        generation: int,
    ) -> None:
        """Send one mutation to every replica, failing the write if any refuses."""
        for index, target in enumerate(self._targets):
            try:
                apply(target)
            except Exception as error:
                logger.error(
                    "replication failed",
                    extra={
                        "document_id": document_id,
                        "generation": generation,
                        "replica_index": index,
                    },
                    exc_info=error,
                )
                raise ReplicationError(
                    f"replication of generation {generation} failed; the mutation is "
                    f"durable on the primary but was not acknowledged"
                ) from error


class ReplicaSynchronizer:
    """A replica's startup check: prove synchronisation, or do not serve.

    A replica that merely recovered its own database has proved nothing about
    what its primary accepted while it was away. So it compares generations and
    resynchronises if it is behind — and if the primary cannot be reached at
    all, it stays **not ready** rather than serving state it cannot vouch for.
    Refusing to serve is the safe answer; claiming readiness without evidence
    is not.
    """

    def __init__(self, engine: SearchEngine, primary: PrimarySource) -> None:
        self._engine = engine
        self._primary = primary

    def synchronize(self) -> SyncOutcome:
        """Verify this replica against its primary, resynchronising if needed."""
        self._engine.mark_recovering("verifying synchronization with the primary")
        local = self._engine.generation

        try:
            remote = self._primary.generation()
        except Exception as error:
            detail = "primary unreachable, synchronization unverified"
            logger.error(detail, exc_info=error)
            self._engine.mark_recovering(detail)
            return SyncOutcome(False, detail, local, None)

        if local == remote:
            self._engine.mark_ready()
            return SyncOutcome(True, "synchronized with the primary", local, remote)

        if local > remote:
            # The invariant says this cannot happen while the primary keeps its
            # data, so it means the primary lost state or the topology is
            # misconfigured. Either way, guessing would be worse than stopping.
            detail = f"ahead of the primary: local {local}, primary {remote}"
            self._engine.mark_out_of_sync(detail)
            return SyncOutcome(False, detail, local, remote)

        return self._resynchronize(local, remote)

    def resynchronize(self) -> SyncOutcome:
        """Repair this replica by pulling a fresh snapshot from the primary."""
        return self._resynchronize(self._engine.generation, None)

    def _resynchronize(self, local: int, remote: int | None) -> SyncOutcome:
        """Pull a snapshot, replace the corpus, rebuild, and become ready."""
        self._engine.mark_recovering(f"resynchronizing from generation {local}")
        try:
            documents, generation = self._primary.export()
        except Exception as error:
            detail = "resynchronization failed: could not read the primary's corpus"
            logger.error(detail, exc_info=error)
            self._engine.mark_out_of_sync(detail)
            return SyncOutcome(False, detail, local, remote)

        try:
            report: RebuildReport = self._engine.resynchronize(documents, generation)
        except Exception as error:
            detail = "resynchronization failed while rebuilding derived state"
            logger.error(detail, exc_info=error)
            self._engine.mark_out_of_sync(detail)
            return SyncOutcome(False, detail, local, remote)

        logger.info(
            "resynchronized from the primary",
            extra={
                "documents": report.document_count,
                "from_generation": local,
                "to_generation": generation,
            },
        )
        return SyncOutcome(
            True,
            f"resynchronized to generation {generation}",
            generation,
            generation,
        )


class HttpNodeLink:
    """A synchronous HTTP link to another copy of the same logical shard.

    Synchronous on purpose: shard nodes serve requests from FastAPI's thread
    pool, so a primary replicating from inside a request handler is already on a
    worker thread and an async client would buy nothing.

    One class satisfies both directions — a primary uses the mutation methods on
    its replicas, a replica uses the synchronisation methods on its primary —
    because they are the same wire protocol seen from two sides.
    """

    def __init__(self, base_url: str, http: httpx2.Client) -> None:
        self._base_url = base_url.rstrip("/")
        self._http = http

    def apply_put(self, document: Document, generation: int) -> None:
        self._request(
            "PUT",
            f"/internal/replicate/{document.document_id}",
            json={"text": document.text, "generation": generation},
        )

    def apply_delete(self, document_id: str, generation: int) -> None:
        self._request(
            "DELETE",
            f"/internal/replicate/{document_id}",
            params=httpx2.QueryParams({"generation": generation}),
        )

    def generation(self) -> int:
        response = self._request("GET", "/internal/node-status")
        generation: int = response.json()["generation"]
        return generation

    def export(self) -> tuple[Sequence[Document], int]:
        response = self._request("GET", "/internal/export")
        payload = response.json()
        documents = [
            Document(document_id=entry["document_id"], text=entry["text"])
            for entry in payload["documents"]
        ]
        return documents, payload["generation"]

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: object | None = None,
        params: httpx2.QueryParams | None = None,
    ) -> httpx2.Response:
        """Issue one request, turning transport and server failures into domain errors."""
        try:
            response = self._http.request(
                method, f"{self._base_url}{path}", json=json, params=params
            )
        except httpx2.TimeoutException as error:
            raise ShardTimeoutError(f"node at {self._base_url} timed out") from error
        except httpx2.HTTPError as error:
            raise ShardUnavailableError(f"node at {self._base_url} is unreachable") from error

        if response.status_code >= httpx2.codes.BAD_REQUEST:
            raise ShardUnavailableError(f"node at {self._base_url} answered {response.status_code}")
        return response
