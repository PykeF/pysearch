"""Tests for deterministic document routing.

The pinned vectors below are the point of this module. Routing has to survive
restarts and be identical on every node, so the expected shard for a given id is
written down as a constant rather than recomputed — a change in the hash, the
encoding or the byte order would otherwise pass unnoticed.
"""

import pytest

from app.cluster.routing import ShardRouter

# Computed once from blake2b(document_id, digest_size=8), big-endian, % 3.
KNOWN_VECTORS_3_SHARDS = {
    "doc-1": 1,
    "doc-2": 2,
    "doc-3": 2,
    "doc-4": 2,
    "doc-5": 1,
    "doc-6": 2,
    "": 0,
    "a": 2,
    "分布式搜索": 0,
}


def test_known_routing_vectors_are_stable() -> None:
    router = ShardRouter(shard_count=3)

    assert {
        document_id: router.shard_for(document_id) for document_id in KNOWN_VECTORS_3_SHARDS
    } == KNOWN_VECTORS_3_SHARDS


def test_routing_is_repeatable_within_one_router() -> None:
    router = ShardRouter(shard_count=4)

    assert router.shard_for("doc-1") == router.shard_for("doc-1")


def test_routing_is_identical_across_separate_routers() -> None:
    # Two routers stand in for two processes: routing must not depend on any
    # per-instance or per-process state, such as Python's randomised str hash.
    first = ShardRouter(shard_count=5)
    second = ShardRouter(shard_count=5)

    for n in range(50):
        document_id = f"doc-{n}"
        assert first.shard_for(document_id) == second.shard_for(document_id)


def test_every_shard_is_within_range() -> None:
    router = ShardRouter(shard_count=7)

    shards = {router.shard_for(f"doc-{n}") for n in range(200)}

    assert shards <= set(range(7))


def test_a_fixed_corpus_reaches_every_shard() -> None:
    # A deterministic spread check on a fixed corpus, not a statistical claim:
    # this exact set of ids touches all three shards, and would fail loudly if
    # routing collapsed onto one.
    router = ShardRouter(shard_count=3)

    shards = {router.shard_for(f"doc-{n}") for n in range(30)}

    assert shards == {0, 1, 2}


def test_a_single_shard_cluster_routes_everything_to_shard_zero() -> None:
    router = ShardRouter(shard_count=1)

    assert all(router.shard_for(f"doc-{n}") == 0 for n in range(20))


def test_changing_the_shard_count_changes_the_mapping() -> None:
    # The documented limitation of modulo sharding, asserted so it stays visible:
    # resizing a cluster moves documents, which is why the count is fixed.
    three = ShardRouter(shard_count=3)
    four = ShardRouter(shard_count=4)

    moved = [n for n in range(50) if three.shard_for(f"doc-{n}") != four.shard_for(f"doc-{n}")]

    assert moved


@pytest.mark.parametrize("shard_count", [0, -1])
def test_an_invalid_shard_count_is_rejected(shard_count: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        ShardRouter(shard_count=shard_count)


def test_shard_count_is_exposed() -> None:
    assert ShardRouter(shard_count=3).shard_count == 3
