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

Replication and failover
------------------------

Each logical shard may have several physical copies. Statistics and scoring are
taken from **exactly one** copy per logical shard, never from every replica —
counting a replica separately would double every document in N and df.

Copy selection happens during round 1: the primary is tried first, then each
replica, until one answers. A copy that is not READY answers 503, so an
unverified, recovering or out-of-sync replica is skipped automatically rather
than being trusted.

The copy that answered round 1 is **pinned** for round 2. Copies can differ by
mutations that were never acknowledged, so pairing statistics from one copy with
scoring from another could produce a ranking matching no corpus that ever
existed. If a pinned copy disappears between rounds the query fails instead.

Failure
-------

If every copy of a logical shard fails, the whole query fails. Under
cluster-wide statistics a missing shard corrupts N, avgdl and df, so partial
results would be both incomplete and mis-scored. Nothing incomplete is ever
returned as though it were complete.

Writes go only to the configured primary and are never rerouted to a replica.
There is no automatic promotion, so a lost primary means writes to that logical
shard fail while reads keep working from a replica.
"""

import asyncio
import logging
from collections.abc import Awaitable, Sequence
from dataclasses import dataclass

from app.cluster.client import NodeStatus, ShardClient
from app.cluster.errors import (
    ClusterError,
    DistributedSearchError,
    ShardCopiesExhaustedError,
)
from app.cluster.routing import ShardRouter
from app.cluster.topology import ClusterTopology, ShardCopies
from app.search.analysis import analyze
from app.search.document import Document
from app.search.engine import SearchResults
from app.search.index import CorpusStats, IndexStats, merge_corpus_stats

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


@dataclass(frozen=True, slots=True)
class CopyStatus:
    """One physical copy of a logical shard, as reported operationally."""

    role: str
    reachable: bool
    node_id: str
    state: str
    ready: bool
    generation: int | None


@dataclass(frozen=True, slots=True)
class ShardHealth:
    """What one logical shard can currently do.

    Search and write availability are reported separately because they now
    genuinely differ: losing a primary leaves a logical shard readable through
    its replica but not writable, since nothing is promoted automatically.
    """

    shard_id: int
    copies: tuple[CopyStatus, ...]
    search_available: bool
    write_available: bool


@dataclass(frozen=True, slots=True)
class ClusterHealth:
    """The cluster's capability, shard by shard."""

    shard_count: int
    replication_factor: int
    search_available: bool
    write_available: bool
    shards: tuple[ShardHealth, ...]


class Coordinator:
    """Routes writes to one shard and fans searches out to all of them."""

    def __init__(self, router: ShardRouter, topology: ClusterTopology) -> None:
        if topology.shard_count != router.shard_count:
            raise ValueError(
                f"router expects {router.shard_count} shards but the topology has "
                f"{topology.shard_count}"
            )
        self._router = router
        self._topology = topology
        # Guards the whole of a distributed operation. See the module docstring.
        self._lock = asyncio.Lock()

    @property
    def shard_count(self) -> int:
        """The number of logical shards in the cluster."""
        return self._router.shard_count

    @property
    def replication_factor(self) -> int:
        """Physical copies per logical shard."""
        return self._topology.replication_factor

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
            # The configured primary, never a replica: rerouting a write would
            # mean two nodes had accepted writes for one logical shard.
            return await self._topology.primary_for(shard_id).put_document(document)

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
            await self._topology.primary_for(shard_id).delete_document(document_id)

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
            # Round 1 both collects statistics and decides which physical copy
            # of each logical shard is serving this query.
            selection = await self._select_and_collect(unique_terms)
            corpus_stats = merge_corpus_stats(stats for _, stats in selection)
            if corpus_stats.document_count == 0:
                return SearchResults(total=0, results=())
            shard_results = await self._fan_out_search(
                query, limit, corpus_stats, [client for client, _ in selection]
            )

        return self._merge(shard_results, limit)

    async def index_stats(self) -> ClusterStats:
        """Aggregate index statistics across the cluster."""
        async with self._lock:
            per_shard = await self._gather(
                [self._index_stats_from_any_copy(copies) for copies in self._topology],
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
            *(self._shard_has_serving_copy(copies) for copies in self._topology),
            return_exceptions=True,
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

    async def cluster_status(self) -> ClusterHealth:
        """Report each logical shard's copies and what the cluster can do.

        Deliberately not taken under the operation lock: an operator asking why
        the cluster is unhappy should not have to queue behind the traffic they
        are investigating.
        """
        shards = await asyncio.gather(*(self._shard_health(copies) for copies in self._topology))

        return ClusterHealth(
            shard_count=self.shard_count,
            replication_factor=self.replication_factor,
            search_available=all(shard.search_available for shard in shards),
            write_available=all(shard.write_available for shard in shards),
            shards=tuple(shards),
        )

    async def _shard_health(self, copies: ShardCopies) -> ShardHealth:
        """Ask every copy of one logical shard how it is doing."""
        roles = ["primary", *["replica"] * len(copies.replicas)]
        outcomes = await asyncio.gather(
            *(client.node_status() for client in copies.serving_order),
            return_exceptions=True,
        )

        statuses: list[CopyStatus] = []
        for role, outcome in zip(roles, outcomes, strict=True):
            if isinstance(outcome, NodeStatus):
                statuses.append(
                    CopyStatus(
                        role=role,
                        reachable=True,
                        node_id=outcome.node_id,
                        state=outcome.state,
                        ready=outcome.ready,
                        generation=outcome.generation,
                    )
                )
            else:
                statuses.append(
                    CopyStatus(
                        role=role,
                        reachable=False,
                        node_id="",
                        state="unreachable",
                        ready=False,
                        generation=None,
                    )
                )

        return ShardHealth(
            shard_id=copies.shard_id,
            copies=tuple(statuses),
            search_available=any(status.ready for status in statuses),
            # Only the configured primary may accept writes, so a ready replica
            # does not make a logical shard writable.
            write_available=statuses[0].ready,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _select_and_collect(
        self, terms: Sequence[str]
    ) -> Sequence[tuple[ShardClient, CorpusStats]]:
        """Round one: pick one serving copy per logical shard and read its statistics.

        Exactly one copy per logical shard contributes, so a replica never
        double-counts its primary's documents.
        """
        return await self._gather(
            [self._stats_from_any_copy(copies, terms) for copies in self._topology],
            "collecting corpus statistics",
        )

    async def _stats_from_any_copy(
        self, copies: ShardCopies, terms: Sequence[str]
    ) -> tuple[ShardClient, CorpusStats]:
        """Try each copy of a logical shard until one answers.

        A copy that is not READY refuses with 503 and is skipped, which is how
        an unsynchronised replica is kept out of the serving set without the
        coordinator having to reason about generations itself.
        """
        failures: list[Exception] = []
        for client in copies.serving_order:
            try:
                return client, await client.corpus_stats(terms)
            except ClusterError as error:
                failures.append(error)

        raise ShardCopiesExhaustedError(
            f"every copy of logical shard {copies.shard_id} is unavailable",
            shard_id=copies.shard_id,
        ) from (failures[0] if failures else None)

    async def _fan_out_search(
        self,
        query: str,
        limit: int,
        corpus_stats: CorpusStats,
        pinned: Sequence[ShardClient],
    ) -> Sequence[SearchResults]:
        """Round two: the copies that answered round one score with cluster statistics.

        Pinned deliberately. Failing over to a different copy here could pair
        statistics from one corpus state with scoring from another, so a copy
        that disappears between rounds fails the query instead.
        """
        return await self._gather(
            [client.search(query, limit, corpus_stats) for client in pinned],
            "executing distributed search",
        )

    async def _index_stats_from_any_copy(self, copies: ShardCopies) -> IndexStats:
        """Read one logical shard's index statistics from any serving copy."""
        failures: list[Exception] = []
        for client in copies.serving_order:
            try:
                return await client.index_stats()
            except ClusterError as error:
                failures.append(error)

        raise ShardCopiesExhaustedError(
            f"every copy of logical shard {copies.shard_id} is unavailable",
            shard_id=copies.shard_id,
        ) from (failures[0] if failures else None)

    @staticmethod
    async def _shard_has_serving_copy(copies: ShardCopies) -> bool:
        """Whether at least one copy of a logical shard can serve searches."""
        for client in copies.serving_order:
            try:
                if await client.is_ready():
                    return True
            except ClusterError:
                continue
        return False

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
