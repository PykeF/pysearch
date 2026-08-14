"""Replication across a real replicated cluster.

Every node here is a separate application with its own SQLite file, and the
primaries replicate over real HTTP. Assertions go to each physical database
independently, so "replicated" means both copies genuinely hold the data — not
that a shared object was mutated once.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.search.engine import NodeState, SearchEngine
from tests.conftest import CLUSTER_SHARD_COUNT, Cluster

# doc-1 and doc-5 route to logical shard 1, doc-12 to shard 0, the rest to 2.
CORPUS = {
    "doc-1": "Distributed systems make search scalable across many machines.",
    "doc-2": "Search engines rank documents by relevance to a query.",
    "doc-5": "Sharding splits an index so that each node holds part of the data.",
    "doc-12": "Replication keeps a second durable copy of every logical shard.",
}


def engine_of(cluster: Cluster, name: str) -> SearchEngine:
    """The engine inside one physical node, for reading its database directly."""
    engine: SearchEngine = cluster.nodes[name].state.engine
    return engine


def corpus_of(cluster: Cluster, name: str) -> dict[str, str]:
    documents, _ = engine_of(cluster, name).export_snapshot()
    return {document.document_id: document.text for document in documents}


def index_corpus(client: TestClient) -> None:
    for document_id, text in CORPUS.items():
        response = client.put(f"/documents/{document_id}", json={"text": text})
        assert response.status_code in (200, 201), response.text


# ----------------------------------------------------------------------
# Topology
# ----------------------------------------------------------------------


def test_every_physical_node_has_its_own_database(replicated_cluster: Cluster) -> None:
    paths = replicated_cluster.shard_paths

    assert len(paths) == CLUSTER_SHARD_COUNT * 2
    # No shared file anywhere: a primary and its replica sharing a database
    # would not be replication, just two views of one copy.
    assert len(set(paths)) == len(paths)


def test_the_cluster_reports_its_replication_factor(replicated_cluster: Cluster) -> None:
    status = replicated_cluster.client.get("/cluster/status").json()

    assert status["replication_factor"] == 2
    assert status["shard_count"] == CLUSTER_SHARD_COUNT
    assert status["search_available"] is True
    assert status["write_available"] is True
    assert [copy["role"] for copy in status["shards"][0]["copies"]] == ["primary", "replica"]


def test_every_copy_starts_ready_and_synchronized(replicated_cluster: Cluster) -> None:
    status = replicated_cluster.client.get("/cluster/status").json()

    for shard in status["shards"]:
        generations = {copy["generation"] for copy in shard["copies"]}
        assert all(copy["ready"] for copy in shard["copies"])
        assert len(generations) == 1


# ----------------------------------------------------------------------
# Normal replication
# ----------------------------------------------------------------------


def test_a_write_reaches_both_copies(replicated_cluster: Cluster) -> None:
    response = replicated_cluster.client.put(
        "/documents/doc-1", json={"text": "distributed search"}
    )

    assert response.status_code == 201
    assert response.json()["shard_id"] == 1
    assert corpus_of(replicated_cluster, "shard-1-primary") == {"doc-1": "distributed search"}
    assert corpus_of(replicated_cluster, "shard-1-replica") == {"doc-1": "distributed search"}


def test_both_copies_advance_to_the_same_generation(replicated_cluster: Cluster) -> None:
    index_corpus(replicated_cluster.client)

    for shard_id in range(CLUSTER_SHARD_COUNT):
        primary = engine_of(replicated_cluster, f"shard-{shard_id}-primary")
        replica = engine_of(replicated_cluster, f"shard-{shard_id}-replica")
        assert primary.generation == replica.generation


def test_replication_does_not_touch_other_shards(replicated_cluster: Cluster) -> None:
    replicated_cluster.client.put("/documents/doc-1", json={"text": "search"})

    # Exactly one logical shard took the write; a broadcast would show here.
    assert corpus_of(replicated_cluster, "shard-0-primary") == {}
    assert corpus_of(replicated_cluster, "shard-2-primary") == {}


def test_replacement_reaches_both_copies(replicated_cluster: Cluster) -> None:
    replicated_cluster.client.put("/documents/doc-1", json={"text": "first version"})

    response = replicated_cluster.client.put("/documents/doc-1", json={"text": "second version"})

    assert response.status_code == 200
    assert corpus_of(replicated_cluster, "shard-1-primary")["doc-1"] == "second version"
    assert corpus_of(replicated_cluster, "shard-1-replica")["doc-1"] == "second version"


def test_deletion_reaches_both_copies(replicated_cluster: Cluster) -> None:
    replicated_cluster.client.put("/documents/doc-1", json={"text": "search"})

    assert replicated_cluster.client.delete("/documents/doc-1").status_code == 204

    assert corpus_of(replicated_cluster, "shard-1-primary") == {}
    assert corpus_of(replicated_cluster, "shard-1-replica") == {}


def test_deleting_an_unknown_document_still_returns_404(replicated_cluster: Cluster) -> None:
    # Public semantics are unchanged by replication; only the internal
    # replication path treats an absent document as a no-op.
    assert replicated_cluster.client.delete("/documents/doc-1").status_code == 404


def test_search_reflects_replicated_state(replicated_cluster: Cluster) -> None:
    index_corpus(replicated_cluster.client)

    payload = replicated_cluster.client.get("/search", params={"q": "search"}).json()

    assert payload["total"] == 2
    assert {hit["document_id"] for hit in payload["results"]} == {"doc-1", "doc-2"}


# ----------------------------------------------------------------------
# Statistics must count logical shards, not physical copies
# ----------------------------------------------------------------------


def test_replicas_are_not_double_counted(replicated_cluster: Cluster) -> None:
    index_corpus(replicated_cluster.client)

    stats = replicated_cluster.client.get("/index/stats").json()

    # Six physical copies hold this corpus; N is the logical corpus size.
    assert stats["document_count"] == len(CORPUS)
    assert len(stats["shards"]) == CLUSTER_SHARD_COUNT


def test_cluster_statistics_match_an_unreplicated_cluster(
    replicated_cluster: Cluster, cluster: Cluster
) -> None:
    index_corpus(replicated_cluster.client)
    index_corpus(cluster.client)

    replicated = replicated_cluster.client.get("/index/stats").json()
    single_copy = cluster.client.get("/index/stats").json()

    assert replicated["document_count"] == single_copy["document_count"]
    assert replicated["average_document_length"] == pytest.approx(
        single_copy["average_document_length"]
    )


# ----------------------------------------------------------------------
# Split-brain prevention
# ----------------------------------------------------------------------


def test_a_replica_refuses_a_coordinator_write(replicated_cluster: Cluster) -> None:
    """Only one node per logical shard can ever accept a write."""
    replica = TestClient(replicated_cluster.nodes["shard-1-replica"])

    assert replica.put("/internal/documents/doc-1", json={"text": "x"}).status_code == 409
    assert replica.delete("/internal/documents/doc-1").status_code == 409


def test_a_primary_refuses_a_replicated_mutation(replicated_cluster: Cluster) -> None:
    """A primary is the only node allowed to decide what its shard contains."""
    primary = TestClient(replicated_cluster.nodes["shard-1-primary"])

    response = primary.put("/internal/replicate/doc-1", json={"text": "x", "generation": 1})

    assert response.status_code == 409


def test_only_the_primary_serves_an_export(replicated_cluster: Cluster) -> None:
    # Recovering from a replica could move a copy backwards, since the primary
    # is by construction the most advanced copy.
    replica = TestClient(replicated_cluster.nodes["shard-1-replica"])
    primary = TestClient(replicated_cluster.nodes["shard-1-primary"])

    assert replica.get("/internal/export").status_code == 409
    assert primary.get("/internal/export").status_code == 200


def test_a_replica_reports_its_role_and_generation(replicated_cluster: Cluster) -> None:
    replicated_cluster.client.put("/documents/doc-1", json={"text": "search"})
    replica = TestClient(replicated_cluster.nodes["shard-1-replica"])

    status = replica.get("/internal/node-status").json()

    assert status["replica_role"] == "replica"
    assert status["shard_id"] == 1
    assert status["state"] == str(NodeState.READY)
    assert status["generation"] == 1


# ----------------------------------------------------------------------
# Restart
# ----------------------------------------------------------------------


def test_both_copies_survive_a_full_restart(tmp_path: Path, start_cluster) -> None:  # type: ignore[no-untyped-def]
    from contextlib import ExitStack

    with ExitStack() as stack:
        running = start_cluster(stack, tmp_path, replication_factor=2)
        index_corpus(running.client)
        before = running.client.get("/search", params={"q": "search"}).json()
        generations = {name: engine_of(running, name).generation for name in running.nodes}

    with ExitStack() as stack:
        restarted = start_cluster(stack, tmp_path, replication_factor=2)

        assert restarted.client.get("/ready").status_code == 200
        assert restarted.client.get("/search", params={"q": "search"}).json() == before
        assert {
            name: engine_of(restarted, name).generation for name in restarted.nodes
        } == generations
