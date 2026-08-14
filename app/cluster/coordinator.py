"""The coordinator: routing, fan-out, global statistics and merging.

It owns no documents. Its entire state is the shard topology, so restarting it
loses nothing and routing resolves identically afterwards.

Distributed scoring
-------------------

Scores from different shards are only comparable if they were computed from the
same corpus statistics. A term that is rare across the cluster but common on one
shard would otherwise get a different ``idf`` there, and merging those numbers
would compare quantities measured on different scales.

So a search takes two rounds, both of them parallel fan-outs::

    round 1   ask every shard for N, summed length, and df of the query's terms
              -> sum them into cluster-wide statistics
    round 2   ask every shard for its local top-k, scored with those statistics
              -> merge the candidates and truncate

The sums in round 1 are exact, not estimates, because every document lives on
exactly one shard.

Consistency
-----------

Rounds 1 and 2 must see the same corpus, or the statistics would describe one
logical state and the scored documents another. Phase 3 solves this
conservatively, with one coordinator-level lock held across both rounds and
across every routed mutation. No distributed snapshots, no versions, no epochs,
no distributed transactions — those would be premature at this size.

The cost is that coordinator operations serialise: one search at a time, and no
write while a search is in flight. That is a real throughput ceiling and it is
accepted deliberately for correctness.

The guarantee holds only for mutations that enter through the coordinator. The
shards' ``/internal`` endpoints are an implementation interface, not a supported
external write path; writing to them directly bypasses this lock.

Top-k
-----

Each shard returns only its local top-k, which is sufficient for an exact global
top-k once scores are comparable: if a document is in the global top-k, fewer
than k documents outrank it anywhere, so fewer than k outrank it on its own
shard, so it is in that shard's local top-k. Nothing can be missed.

Failure
-------

If any shard fails to take part, the whole query fails. Under cluster-wide
statistics a missing shard corrupts N, avgdl and df, so partial results would be
both incomplete and mis-scored. Without replication there is also nothing that
could recover the missing documents, so partial availability would buy nothing
real. Nothing incomplete is ever returned as though it were complete.
"""

import asyncio
import logging
from collections.abc import Awaitable, Sequence
from dataclasses import dataclass

from app.cluster.client import ShardClient
from app.cluster.errors import DistributedSearchError
from app.cluster.routing import ShardRouter
from app.search.analysis import analyze
from app.search.document import Document
from app.search.engine import SearchResults
from app.search.index import CorpusStats, merge_corpus_stats

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ShardIndexStats:
    """One shard's statistics, as reported at cluster level."""

    shard_id: int
    document_count: int
    unique_term_count: int
    average_document_length: float


@dataclass(frozen=True, slots=True)
class ClusterStats:
    """Corpus statistics for the whole cluster.

    There is deliberately no cluster-wide ``unique_term_count``. Vocabulary
    sizes cannot be summed, because the same term legitimately appears on
    several shards, and computing the true union would mean shipping every
    shard's vocabulary. Reporting a sum would be arithmetically wrong, so the
    per-shard figures are given instead and the global one is omitted.
    """

    document_count: int
    average_document_length: float
    shard_count: int
    shards: tuple[ShardIndexStats, ...]


@dataclass(frozen=True, slots=True)
class ClusterReadiness:
    """Whether the cluster can honour its API contract."""

    ready: bool
    ready_shards: tuple[int, ...]
    unready_shards: tuple[int, ...]

    @property
    def detail(self) -> str:
        """A one-line explanation suitable for a readiness response."""
        if self.ready:
            return f"all {len(self.ready_shards)} shards ready"
        return f"shards not ready: {list(self.unready_shards)}"


class Coordinator:
    """Routes writes to one shard and fans searches out to all of them."""

    def __init__(self, router: ShardRouter, clients: Sequence[ShardClient]) -> None:
        if len(clients) != router.shard_count:
            raise ValueError(
                f"router expects {router.shard_count} shards but {len(clients)} clients were given"
            )
        self._router = router
        self._clients = tuple(clients)
        # Guards the whole of a distributed operation. See the module docstring.
        self._lock = asyncio.Lock()

    @property
    def shard_count(self) -> int:
        """The number of shards in the cluster."""
        return self._router.shard_count

    def shard_for(self, document_id: str) -> int:
        """Return the shard that owns a document."""
        return self._router.shard_for(document_id)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    async def index_document(self, document: Document) -> bool:
        """Route a document to its owning shard and store it there.

        Exactly one shard receives it; writes are never broadcast, and never
        rerouted on failure, because rerouting would break the ownership that
        routing depends on.

        Returns:
            ``True`` if the document is new, ``False`` if it replaced one.
        """
        shard_id = self._router.shard_for(document.document_id)
        logger.info(
            "routing document write",
            extra={"document_id": document.document_id, "target_shard": shard_id},
        )

        async with self._lock:
            return await self._clients[shard_id].put_document(document)

    async def delete_document(self, document_id: str) -> None:
        """Delete a document from its owning shard.

        Raises:
            DocumentNotFoundError: if the owning shard does not hold it.
        """
        shard_id = self._router.shard_for(document_id)
        logger.info(
            "routing document delete",
            extra={"document_id": document_id, "target_shard": shard_id},
        )

        async with self._lock:
            await self._clients[shard_id].delete_document(document_id)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    async def search(self, query: str, limit: int) -> SearchResults:
        """Search every shard and return the global top ``limit``.

        Raises:
            DistributedSearchError: if any shard fails to take part.
        """
        # The same analyzer the engine uses, so the statistics collected here
        # describe exactly the terms the shards will score. Duplicates are kept
        # for scoring, where repetition is meaningful, and collapsed for the
        # statistics request, where df is a property of the term alone.
        terms = analyze(query)
        if not terms:
            return SearchResults(total=0, results=())
        unique_terms = sorted(set(terms))

        async with self._lock:
            corpus_stats = await self._collect_corpus_stats(unique_terms)
            if corpus_stats.document_count == 0:
                return SearchResults(total=0, results=())
            shard_results = await self._fan_out_search(query, limit, corpus_stats)

        return self._merge(shard_results, limit)

    async def index_stats(self) -> ClusterStats:
        """Aggregate index statistics across the cluster."""
        async with self._lock:
            per_shard = await self._gather(
                [client.index_stats() for client in self._clients],
                "collecting index statistics",
            )

        document_count = sum(stats.document_count for stats in per_shard)
        total_tokens = sum(stats.total_token_count for stats in per_shard)
        return ClusterStats(
            document_count=document_count,
            # Weighted by document count. A mean of shard means would be wrong
            # whenever the shards hold different numbers of documents.
            average_document_length=(total_tokens / document_count) if document_count else 0.0,
            shard_count=self.shard_count,
            shards=tuple(
                ShardIndexStats(
                    shard_id=shard_id,
                    document_count=stats.document_count,
                    unique_term_count=stats.unique_term_count,
                    average_document_length=stats.average_document_length,
                )
                for shard_id, stats in enumerate(per_shard)
            ),
        )

    async def readiness(self) -> ClusterReadiness:
        """Report whether every shard is ready.

        All shards are required, because the search contract is all-or-nothing:
        a cluster missing a shard cannot answer the queries it advertises.

        Deliberately not taken under the operation lock — a readiness probe that
        queued behind a slow search would report staleness as unreadiness.
        """
        outcomes = await asyncio.gather(
            *(client.is_ready() for client in self._clients), return_exceptions=True
        )

        ready: list[int] = []
        unready: list[int] = []
        for shard_id, outcome in enumerate(outcomes):
            if outcome is True:
                ready.append(shard_id)
            else:
                unready.append(shard_id)

        return ClusterReadiness(
            ready=not unready,
            ready_shards=tuple(ready),
            unready_shards=tuple(unready),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _collect_corpus_stats(self, terms: Sequence[str]) -> CorpusStats:
        """Round one: sum every shard's statistics into cluster-wide ones."""
        parts = await self._gather(
            [client.corpus_stats(terms) for client in self._clients],
            "collecting corpus statistics",
        )
        return merge_corpus_stats(parts)

    async def _fan_out_search(
        self, query: str, limit: int, corpus_stats: CorpusStats
    ) -> Sequence[SearchResults]:
        """Round two: every shard scores its own documents with cluster statistics."""
        return await self._gather(
            [client.search(query, limit, corpus_stats) for client in self._clients],
            "executing distributed search",
        )

    @staticmethod
    def _merge(shard_results: Sequence[SearchResults], limit: int) -> SearchResults:
        """Merge shard candidates into a single deterministic ranking.

        Ordering is the single-node rule — score descending, then document id
        ascending — so ties break identically however the shards happened to
        reply, and response arrival order never influences the result.
        """
        candidates = [hit for results in shard_results for hit in results.results]
        candidates.sort(key=lambda hit: (-hit.score, hit.document_id))
        return SearchResults(
            total=sum(results.total for results in shard_results),
            results=tuple(candidates[:limit]),
        )

    async def _gather[T](self, awaitables: Sequence[Awaitable[T]], action: str) -> list[T]:
        """Await every shard concurrently, failing the operation if any one does.

        Requests are all in flight together, so the cost of a fan-out is the
        slowest shard rather than the sum of the shards.
        """
        outcomes = await asyncio.gather(*awaitables, return_exceptions=True)

        results: list[T] = []
        failed: list[int] = []
        causes: list[BaseException] = []
        for shard_id, outcome in enumerate(outcomes):
            if isinstance(outcome, BaseException):
                failed.append(shard_id)
                causes.append(outcome)
            else:
                results.append(outcome)

        if failed:
            logger.error(
                "distributed operation failed",
                extra={"action": action, "failed_shards": failed},
                exc_info=causes[0],
            )
            raise DistributedSearchError(
                f"{action} failed on shards {failed}", shard_ids=tuple(failed)
            )

        return results
