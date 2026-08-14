"""The coordinator's view of a shard.

Coordinator logic talks to shards through :class:`ShardClient` and never to an
HTTP library. That boundary earns its place for two reasons: the transport is
now a real failure source that has to be translated into domain errors in one
place, and tests need shards whose latency and failures they control.

It is a client, not an RPC framework — six methods, each mirroring one internal
endpoint.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import httpx2

from app.cluster.errors import ShardTimeoutError, ShardUnavailableError
from app.search.document import Document
from app.search.engine import SearchResult, SearchResults
from app.search.errors import DocumentNotFoundError
from app.search.index import CorpusStats, IndexStats


@dataclass(frozen=True, slots=True)
class NodeStatus:
    """One physical node's identity and serving state."""

    node_id: str
    shard_id: int
    replica_role: str
    state: str
    ready: bool
    generation: int


class ShardClient(Protocol):
    """One physical copy of a logical shard, as the coordinator sees it."""

    async def put_document(self, document: Document) -> bool:
        """Store a document on this shard, returning ``True`` if it is new."""
        ...

    async def delete_document(self, document_id: str) -> None:
        """Delete a document from this shard.

        Raises:
            DocumentNotFoundError: if the shard does not hold it.
        """
        ...

    async def search(self, query: str, limit: int, corpus_stats: CorpusStats) -> SearchResults:
        """Return this shard's local top ``limit``, scored with cluster statistics."""
        ...

    async def corpus_stats(self, terms: Sequence[str]) -> CorpusStats:
        """Return this shard's contribution to the cluster's BM25 statistics."""
        ...

    async def index_stats(self) -> IndexStats:
        """Return this shard's local index statistics."""
        ...

    async def is_ready(self) -> bool:
        """Return whether this shard reports itself ready to serve."""
        ...

    async def node_status(self) -> "NodeStatus":
        """Return this node's role, serving state and generation."""
        ...


class HttpShardClient:
    """A :class:`ShardClient` that speaks HTTP to a shard node."""

    def __init__(self, base_url: str, http: httpx2.AsyncClient) -> None:
        self._base_url = base_url.rstrip("/")
        self._http = http

    async def put_document(self, document: Document) -> bool:
        response = await self._request(
            "PUT",
            f"/internal/documents/{document.document_id}",
            json={"text": document.text},
        )
        created: bool = response.json()["created"]
        return created

    async def delete_document(self, document_id: str) -> None:
        response = await self._request(
            "DELETE",
            f"/internal/documents/{document_id}",
            expected_absent=True,
        )
        if response.status_code == httpx2.codes.NOT_FOUND:
            raise DocumentNotFoundError(f"document {document_id!r} is not indexed")

    async def search(self, query: str, limit: int, corpus_stats: CorpusStats) -> SearchResults:
        response = await self._request(
            "POST",
            "/internal/search",
            json={
                "query": query,
                "limit": limit,
                "corpus_stats": {
                    "document_count": corpus_stats.document_count,
                    "total_document_length": corpus_stats.total_document_length,
                    "document_frequencies": dict(corpus_stats.document_frequencies),
                },
            },
        )
        payload = response.json()
        return SearchResults(
            total=payload["total"],
            results=tuple(
                SearchResult(
                    document_id=hit["document_id"],
                    score=hit["score"],
                    text=hit["text"],
                )
                for hit in payload["results"]
            ),
        )

    async def corpus_stats(self, terms: Sequence[str]) -> CorpusStats:
        response = await self._request(
            "GET",
            "/internal/corpus-stats",
            params=httpx2.QueryParams([("term", term) for term in terms]),
        )
        payload = response.json()
        return CorpusStats(
            document_count=payload["document_count"],
            total_document_length=payload["total_document_length"],
            document_frequencies=payload["document_frequencies"],
        )

    async def index_stats(self) -> IndexStats:
        response = await self._request("GET", "/internal/stats")
        payload = response.json()
        return IndexStats(
            document_count=payload["document_count"],
            unique_term_count=payload["unique_term_count"],
            average_document_length=payload["average_document_length"],
            total_token_count=payload["total_token_count"],
        )

    async def is_ready(self) -> bool:
        try:
            response = await self._http.get(f"{self._base_url}/ready")
        except httpx2.HTTPError:
            # Readiness answers a question, it does not raise one: a shard that
            # cannot be reached is simply not ready.
            return False
        return response.status_code == httpx2.codes.OK

    async def node_status(self) -> NodeStatus:
        response = await self._request("GET", "/internal/node-status")
        payload = response.json()
        return NodeStatus(
            node_id=payload["node_id"],
            shard_id=payload["shard_id"],
            replica_role=payload["replica_role"],
            state=payload["state"],
            ready=payload["ready"],
            generation=payload["generation"],
        )

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: object | None = None,
        params: httpx2.QueryParams | None = None,
        expected_absent: bool = False,
    ) -> httpx2.Response:
        """Issue one request, translating transport and server failures.

        ``expected_absent`` lets a 404 through for callers that treat "no such
        document" as an answer rather than a failure.
        """
        try:
            response = await self._http.request(
                method, f"{self._base_url}{path}", json=json, params=params
            )
        except httpx2.TimeoutException as error:
            raise ShardTimeoutError(f"shard at {self._base_url} timed out") from error
        except httpx2.HTTPError as error:
            raise ShardUnavailableError(f"shard at {self._base_url} is unreachable") from error

        if expected_absent and response.status_code == httpx2.codes.NOT_FOUND:
            return response
        if response.status_code >= httpx2.codes.BAD_REQUEST:
            raise ShardUnavailableError(
                f"shard at {self._base_url} answered {response.status_code}"
            )
        return response
