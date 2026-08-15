"""Tests for coordinator routing, fan-out, merging and failure policy.

Shards here are fakes, so failures and slowness are triggered by flags rather
than by timing. Nothing in this module sleeps or races.
"""

import asyncio
from collections.abc import Sequence

import pytest

from app.cluster.client import NodeStatus
from app.cluster.coordinator import Coordinator
from app.cluster.errors import DistributedSearchError, ShardTimeoutError, ShardUnavailableError
from app.cluster.routing import ShardRouter
from app.cluster.topology import build_topology
from app.search.document import Document
from app.search.engine import SearchEngine
from app.search.errors import DocumentNotFoundError
from app.search.index import CorpusStats, IndexStats
from app.search.results import SearchResult, SearchResults
from app.storage.sqlite_store import IN_MEMORY, SqliteDocumentStore


class FakeShardClient:
    """A shard backed by a real in-memory engine, with controllable failures."""

    def __init__(self) -> None:
        self.engine = SearchEngine(SqliteDocumentStore.open(IN_MEMORY))
        self.engine.initialize()
        self.calls: list[str] = []
        self.fail_with: Exception | None = None
        self.ready = True

    def _guard(self, call: str) -> None:
        self.calls.append(call)
        if self.fail_with is not None:
            raise self.fail_with

    async def put_document(self, document: Document) -> bool:
        self._guard("put")
        return self.engine.index_document(document)

    async def delete_document(self, document_id: str) -> None:
        self._guard("delete")
        self.engine.delete_document(document_id)

    async def search(self, query: str, limit: int, corpus_stats: CorpusStats) -> SearchResults:
        self._guard("search")
        return self.engine.search(query, limit, corpus_stats)

    async def corpus_stats(self, terms: Sequence[str]) -> CorpusStats:
        self._guard("corpus_stats")
        return self.engine.corpus_stats(terms)

    async def index_stats(self) -> IndexStats:
        self._guard("index_stats")
        return self.engine.stats()

    async def is_ready(self) -> bool:
        self.calls.append("is_ready")
        return self.ready

    async def node_status(self) -> NodeStatus:
        self._guard("node_status")
        return NodeStatus(
            node_id="fake",
            shard_id=0,
            replica_role="primary",
            state="ready" if self.ready else "recovering",
            ready=self.ready,
            generation=self.engine.generation,
        )


@pytest.fixture
def shards() -> list[FakeShardClient]:
    return [FakeShardClient() for _ in range(3)]


@pytest.fixture
def coordinator(shards: list[FakeShardClient]) -> Coordinator:
    return Coordinator(ShardRouter(shard_count=3), build_topology(shards, []))


def run(awaitable):  # type: ignore[no-untyped-def]
    """Drive one coordinator call to completion."""
    return asyncio.run(awaitable)


# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------


def test_client_count_must_match_the_shard_count() -> None:
    with pytest.raises(ValueError, match="expects 3 shards"):
        Coordinator(ShardRouter(shard_count=3), build_topology([FakeShardClient()], []))


# ----------------------------------------------------------------------
# Write and delete routing
# ----------------------------------------------------------------------


def test_a_write_goes_to_exactly_one_shard(
    coordinator: Coordinator, shards: list[FakeShardClient]
) -> None:
    # "doc-1" routes to shard 1 under the pinned vectors.
    created = run(coordinator.index_document(Document(document_id="doc-1", text="search")))

    assert created is True
    assert shards[1].calls == ["put"]
    assert shards[0].calls == []
    assert shards[2].calls == []


def test_writes_are_not_broadcast(coordinator: Coordinator, shards: list[FakeShardClient]) -> None:
    for n in range(10):
        run(coordinator.index_document(Document(document_id=f"doc-{n}", text="search")))

    # Every document landed on exactly one shard, so the counts sum to ten.
    assert sum(shard.calls.count("put") for shard in shards) == 10


def test_replacing_a_document_reaches_the_same_shard(
    coordinator: Coordinator, shards: list[FakeShardClient]
) -> None:
    run(coordinator.index_document(Document(document_id="doc-1", text="first")))
    created = run(coordinator.index_document(Document(document_id="doc-1", text="second")))

    assert created is False
    assert shards[1].calls == ["put", "put"]


def test_delete_is_routed_to_the_owning_shard(
    coordinator: Coordinator, shards: list[FakeShardClient]
) -> None:
    run(coordinator.index_document(Document(document_id="doc-1", text="search")))
    run(coordinator.delete_document("doc-1"))

    assert shards[1].calls == ["put", "delete"]
    assert shards[1].engine.stats().document_count == 0


def test_deleting_an_unknown_document_reports_not_found(coordinator: Coordinator) -> None:
    with pytest.raises(DocumentNotFoundError):
        run(coordinator.delete_document("doc-1"))


def test_routing_is_exposed_for_reporting(coordinator: Coordinator) -> None:
    assert coordinator.shard_for("doc-1") == 1
    assert coordinator.shard_count == 3


# ----------------------------------------------------------------------
# Fan-out and merging
# ----------------------------------------------------------------------


def index_across_shards(coordinator: Coordinator, corpus: dict[str, str]) -> None:
    for document_id, text in corpus.items():
        run(coordinator.index_document(Document(document_id=document_id, text=text)))


def test_search_reaches_every_shard(
    coordinator: Coordinator, shards: list[FakeShardClient]
) -> None:
    index_across_shards(coordinator, {"doc-1": "search", "doc-2": "search"})
    for shard in shards:
        shard.calls.clear()

    run(coordinator.search("search", limit=10))

    # Both rounds, on every shard, including the ones holding nothing.
    for shard in shards:
        assert shard.calls == ["corpus_stats", "search"]


def test_results_from_several_shards_are_merged(coordinator: Coordinator) -> None:
    index_across_shards(
        coordinator,
        {f"doc-{n}": "distributed search" for n in range(1, 7)},
    )

    outcome = run(coordinator.search("search", limit=10))

    assert outcome.total == 6
    assert len(outcome.results) == 6


def test_the_merged_order_is_by_score_then_document_id(coordinator: Coordinator) -> None:
    index_across_shards(
        coordinator,
        {
            "doc-1": "search",
            "doc-2": "search",
            "doc-3": "search",
            "doc-4": "search plus several additional unrelated filler words here",
        },
    )

    outcome = run(coordinator.search("search", limit=10))
    ordering = [(-hit.score, hit.document_id) for hit in outcome.results]

    assert ordering == sorted(ordering)


def test_cross_shard_ties_break_on_document_id(coordinator: Coordinator) -> None:
    # Identical text on shards 1, 2 and 2 respectively means identical scores,
    # so ordering is decided purely by the tie-break rule.
    index_across_shards(
        coordinator, {"doc-1": "identical", "doc-2": "identical", "doc-4": "identical"}
    )

    outcome = run(coordinator.search("identical", limit=10))

    assert len({hit.score for hit in outcome.results}) == 1
    assert [hit.document_id for hit in outcome.results] == ["doc-1", "doc-2", "doc-4"]


def test_the_limit_applies_globally_not_per_shard(coordinator: Coordinator) -> None:
    index_across_shards(coordinator, {f"doc-{n}": "search" for n in range(1, 7)})

    outcome = run(coordinator.search("search", limit=2))

    assert outcome.total == 6
    assert len(outcome.results) == 2


def test_an_empty_query_short_circuits(
    coordinator: Coordinator, shards: list[FakeShardClient]
) -> None:
    index_across_shards(coordinator, {"doc-1": "search"})
    for shard in shards:
        shard.calls.clear()

    outcome = run(coordinator.search("!!!", limit=10))

    assert outcome.total == 0
    # No point asking shards anything when nothing can match.
    assert all(shard.calls == [] for shard in shards)


def test_searching_an_empty_cluster_returns_nothing(coordinator: Coordinator) -> None:
    outcome = run(coordinator.search("search", limit=10))

    assert outcome.total == 0
    assert outcome.results == ()


def test_shard_response_order_does_not_affect_the_result(coordinator: Coordinator) -> None:
    index_across_shards(coordinator, {f"doc-{n}": "search" for n in range(1, 7)})

    first = run(coordinator.search("search", limit=10))
    second = run(coordinator.search("search", limit=10))

    assert first == second


# ----------------------------------------------------------------------
# Global statistics
# ----------------------------------------------------------------------


def test_statistics_are_collected_before_scoring(
    coordinator: Coordinator, shards: list[FakeShardClient]
) -> None:
    index_across_shards(coordinator, {"doc-1": "search", "doc-2": "search"})
    for shard in shards:
        shard.calls.clear()

    run(coordinator.search("search", limit=10))

    assert shards[0].calls.index("corpus_stats") < shards[0].calls.index("search")


def test_repeated_query_terms_are_deduplicated_for_statistics(
    coordinator: Coordinator, shards: list[FakeShardClient]
) -> None:
    index_across_shards(coordinator, {"doc-1": "search"})

    # Round one asks for df once per distinct term, even though scoring will
    # weigh the repeated term twice.
    selection = run(coordinator._select_and_collect(["search"]))
    frequencies = [stats.document_frequencies for _, stats in selection]

    assert {"search": 1} in frequencies


# ----------------------------------------------------------------------
# Failure policy
# ----------------------------------------------------------------------


def test_a_failing_shard_fails_the_whole_search(
    coordinator: Coordinator, shards: list[FakeShardClient]
) -> None:
    index_across_shards(coordinator, {f"doc-{n}": "search" for n in range(1, 7)})
    shards[2].fail_with = ShardUnavailableError("shard 2 is down")

    with pytest.raises(DistributedSearchError) as raised:
        run(coordinator.search("search", limit=10))

    assert raised.value.shard_ids == (2,)


def test_a_timing_out_shard_fails_the_whole_search(
    coordinator: Coordinator, shards: list[FakeShardClient]
) -> None:
    index_across_shards(coordinator, {"doc-1": "search"})
    shards[0].fail_with = ShardTimeoutError("shard 0 timed out")

    with pytest.raises(DistributedSearchError):
        run(coordinator.search("search", limit=10))


def test_every_failing_shard_is_reported(
    coordinator: Coordinator, shards: list[FakeShardClient]
) -> None:
    shards[0].fail_with = ShardUnavailableError("down")
    shards[2].fail_with = ShardTimeoutError("timed out")

    with pytest.raises(DistributedSearchError) as raised:
        run(coordinator.search("search", limit=10))

    assert raised.value.shard_ids == (0, 2)


def test_a_failing_owner_fails_the_write(
    coordinator: Coordinator, shards: list[FakeShardClient]
) -> None:
    shards[1].fail_with = ShardUnavailableError("shard 1 is down")

    with pytest.raises(ShardUnavailableError):
        run(coordinator.index_document(Document(document_id="doc-1", text="search")))

    # Never rerouted: the other shards were not asked to take the write.
    assert shards[0].calls == []
    assert shards[2].calls == []


def test_a_failing_owner_fails_the_delete(
    coordinator: Coordinator, shards: list[FakeShardClient]
) -> None:
    run(coordinator.index_document(Document(document_id="doc-1", text="search")))
    shards[1].fail_with = ShardUnavailableError("shard 1 is down")

    with pytest.raises(ShardUnavailableError):
        run(coordinator.delete_document("doc-1"))


# ----------------------------------------------------------------------
# Readiness and statistics
# ----------------------------------------------------------------------


def test_the_cluster_is_ready_when_every_shard_is(coordinator: Coordinator) -> None:
    readiness = run(coordinator.readiness())

    assert readiness.ready is True
    assert readiness.ready_shards == (0, 1, 2)
    assert readiness.unready_shards == ()


def test_one_unready_shard_makes_the_cluster_unready(
    coordinator: Coordinator, shards: list[FakeShardClient]
) -> None:
    shards[2].ready = False

    readiness = run(coordinator.readiness())

    assert readiness.ready is False
    assert readiness.unready_shards == (2,)
    assert "2" in readiness.detail


def test_cluster_statistics_sum_document_counts(coordinator: Coordinator) -> None:
    index_across_shards(coordinator, {f"doc-{n}": "search" for n in range(1, 7)})

    stats = run(coordinator.index_stats())

    assert stats.document_count == 6
    assert stats.shard_count == 3
    assert sum(shard.document_count for shard in stats.shards) == 6


def test_average_document_length_is_weighted_not_averaged(
    coordinator: Coordinator,
) -> None:
    # doc-3 routes to shard 2 and doc-1 to shard 1, so the shards hold documents
    # of different lengths; a mean of shard means would give 2.5 rather than 3.
    index_across_shards(coordinator, {"doc-1": "one two three four", "doc-3": "one two"})

    stats = run(coordinator.index_stats())

    assert stats.document_count == 2
    assert stats.average_document_length == pytest.approx(3.0)


def test_cluster_statistics_report_no_global_vocabulary_size(
    coordinator: Coordinator,
) -> None:
    index_across_shards(coordinator, {"doc-1": "search", "doc-2": "search"})

    stats = run(coordinator.index_stats())

    # Summing shard vocabularies would double-count "search"; the contract
    # deliberately exposes per-shard figures instead of a false global one.
    assert not hasattr(stats, "unique_term_count")
    assert [shard.unique_term_count for shard in stats.shards] == [0, 1, 1]


def test_empty_cluster_statistics(coordinator: Coordinator) -> None:
    stats = run(coordinator.index_stats())

    assert stats.document_count == 0
    assert stats.average_document_length == 0.0


def test_a_failing_shard_fails_cluster_statistics(
    coordinator: Coordinator, shards: list[FakeShardClient]
) -> None:
    shards[1].fail_with = ShardUnavailableError("down")

    with pytest.raises(DistributedSearchError):
        run(coordinator.index_stats())


# ----------------------------------------------------------------------
# Serialization
# ----------------------------------------------------------------------


def test_the_operation_lock_serialises_searches_and_writes(
    coordinator: Coordinator, shards: list[FakeShardClient]
) -> None:
    """A write cannot interleave between a search's two rounds.

    The fake shard records a marker when scoring begins; if the write had been
    admitted between the statistics round and the scoring round, the recorded
    order would show it.
    """
    index_across_shards(coordinator, {"doc-1": "search"})
    for shard in shards:
        shard.calls.clear()

    async def exercise() -> None:
        # doc-5 routes to shard 1, so that shard sees both operations and its
        # call log records how they interleaved.
        await asyncio.gather(
            coordinator.search("search", limit=10),
            coordinator.index_document(Document(document_id="doc-5", text="search")),
        )

    asyncio.run(exercise())

    # Shard 1 saw either the whole search then the write, or the write then the
    # whole search — never a write wedged between the two rounds.
    calls = [call for call in shards[1].calls if call in {"corpus_stats", "search", "put"}]
    assert calls in (
        ["corpus_stats", "search", "put"],
        ["put", "corpus_stats", "search"],
    )


def test_results_carry_document_text(coordinator: Coordinator) -> None:
    index_across_shards(coordinator, {"doc-1": "distributed search"})

    outcome = run(coordinator.search("search", limit=10))

    assert outcome.results[0] == SearchResult(
        document_id="doc-1", score=outcome.results[0].score, text="distributed search"
    )
