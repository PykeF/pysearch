"""Failover and recovery.

This is the phase's headline claim: losing one physical node should not make
distributed search unavailable. Nodes are taken out of service structurally —
by closing their engine or breaking their index — never by sleeping, so nothing
here depends on timing.
"""

from contextlib import ExitStack
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.search.document import Document
from app.search.engine import NodeState, SearchEngine
from tests.conftest import Cluster

CORPUS = {
    "doc-1": "distributed search across machines",
    "doc-2": "search engines rank documents",
    "doc-5": "sharding splits an index across nodes",
    "doc-12": "replication keeps a second durable copy",
}


def engine_of(cluster: Cluster, name: str) -> SearchEngine:
    engine: SearchEngine = cluster.nodes[name].state.engine
    return engine


def stop(cluster: Cluster, name: str) -> None:
    """Take a node out of service the way a crash would: it stops answering."""
    engine_of(cluster, name).close()


def index_corpus(cluster: Cluster) -> None:
    for document_id, text in CORPUS.items():
        response = cluster.client.put(f"/documents/{document_id}", json={"text": text})
        assert response.status_code in (200, 201), response.text


def search(cluster: Cluster, query: str) -> dict[str, object]:
    response = cluster.client.get("/search", params={"q": query, "limit": 100})
    assert response.status_code == 200, response.text
    payload: dict[str, object] = response.json()
    return payload


# ----------------------------------------------------------------------
# Read failover
# ----------------------------------------------------------------------


def test_search_survives_losing_a_primary(replicated_cluster: Cluster) -> None:
    """The whole point of the phase."""
    index_corpus(replicated_cluster)
    before = search(replicated_cluster, "search")

    stop(replicated_cluster, "shard-1-primary")

    assert search(replicated_cluster, "search") == before


def test_search_survives_losing_a_replica(replicated_cluster: Cluster) -> None:
    index_corpus(replicated_cluster)
    before = search(replicated_cluster, "search")

    stop(replicated_cluster, "shard-1-replica")

    # The primary is preferred anyway, so nothing changes.
    assert search(replicated_cluster, "search") == before


def test_search_survives_losing_one_copy_of_every_shard(
    replicated_cluster: Cluster,
) -> None:
    index_corpus(replicated_cluster)
    before = search(replicated_cluster, "search")

    for shard_id in range(3):
        stop(replicated_cluster, f"shard-{shard_id}-primary")

    assert search(replicated_cluster, "search") == before


def test_failover_results_are_identical_not_merely_present(
    replicated_cluster: Cluster,
) -> None:
    """Failover must not change ranking, only where the answer came from.

    Write-all replication is what buys this: the replica holds exactly the
    acknowledged corpus, so the statistics and scores are the same ones.
    """
    index_corpus(replicated_cluster)
    before = search(replicated_cluster, "search engines sharding")
    stats_before = replicated_cluster.client.get("/index/stats").json()

    stop(replicated_cluster, "shard-1-primary")
    stop(replicated_cluster, "shard-2-primary")

    assert search(replicated_cluster, "search engines sharding") == before
    assert replicated_cluster.client.get("/index/stats").json() == stats_before


def test_losing_every_copy_of_one_shard_fails_the_search(
    replicated_cluster: Cluster,
) -> None:
    """Completeness still wins: a missing logical shard is not a partial answer."""
    index_corpus(replicated_cluster)

    stop(replicated_cluster, "shard-1-primary")
    stop(replicated_cluster, "shard-1-replica")

    response = replicated_cluster.client.get("/search", params={"q": "search"})

    assert response.status_code == 503
    assert "1" in response.json()["detail"]


def test_readiness_tracks_search_availability(replicated_cluster: Cluster) -> None:
    stop(replicated_cluster, "shard-1-primary")
    # One copy left, so search still works and readiness says so.
    assert replicated_cluster.client.get("/ready").status_code == 200

    stop(replicated_cluster, "shard-1-replica")
    assert replicated_cluster.client.get("/ready").status_code == 503


def test_cluster_status_separates_search_from_write_availability(
    replicated_cluster: Cluster,
) -> None:
    stop(replicated_cluster, "shard-1-primary")

    status = replicated_cluster.client.get("/cluster/status").json()
    shard = status["shards"][1]

    # Readable through the replica, but not writable: nothing is promoted.
    assert shard["search_available"] is True
    assert shard["write_available"] is False
    assert status["search_available"] is True
    assert status["write_available"] is False


# ----------------------------------------------------------------------
# Write failover: deliberately absent
# ----------------------------------------------------------------------


def test_writes_fail_when_the_primary_is_gone(replicated_cluster: Cluster) -> None:
    """No automatic promotion, so writes stop rather than risk two primaries."""
    stop(replicated_cluster, "shard-1-primary")

    response = replicated_cluster.client.put("/documents/doc-1", json={"text": "search"})

    assert response.status_code == 503
    # And the write was not quietly rerouted to the replica.
    assert engine_of(replicated_cluster, "shard-1-replica").stats().document_count == 0


def test_writes_to_healthy_shards_still_work(replicated_cluster: Cluster) -> None:
    stop(replicated_cluster, "shard-1-primary")

    # doc-12 belongs to logical shard 0, which is untouched.
    assert (
        replicated_cluster.client.put("/documents/doc-12", json={"text": "search"}).status_code
        == 201
    )


def test_writes_fail_when_a_replica_is_gone(replicated_cluster: Cluster) -> None:
    """Write-all: an unavailable replica costs write availability, by design."""
    stop(replicated_cluster, "shard-1-replica")

    response = replicated_cluster.client.put("/documents/doc-1", json={"text": "search"})

    assert response.status_code == 503
    # The mutation is durable on the primary even though the write failed, which
    # is the ambiguity the contract states rather than hides.
    assert engine_of(replicated_cluster, "shard-1-primary").stats().document_count == 1


def test_search_still_works_after_an_unacknowledged_write(
    replicated_cluster: Cluster,
) -> None:
    stop(replicated_cluster, "shard-1-replica")
    replicated_cluster.client.put("/documents/doc-1", json={"text": "search"})

    # The primary is still serving, so the query is answered from the copy that
    # does have the document.
    assert search(replicated_cluster, "search")["total"] == 1


# ----------------------------------------------------------------------
# Query target pinning
# ----------------------------------------------------------------------


def test_a_copy_lost_between_rounds_fails_the_query(
    replicated_cluster: Cluster, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pinning: no failover between the statistics round and the scoring round.

    Switching copies mid-query could pair statistics from one corpus state with
    scoring from another, producing a ranking that matches no state at all.
    """
    index_corpus(replicated_cluster)
    primary = engine_of(replicated_cluster, "shard-1-primary")

    original_search = primary.search
    calls = {"count": 0}

    def fail_on_scoring(*args: object, **kwargs: object) -> object:
        calls["count"] += 1
        raise RuntimeError("this copy disappeared after answering round one")

    # Statistics succeed on the primary, and only then does it stop answering.
    monkeypatch.setattr(primary, "search", fail_on_scoring)

    response = replicated_cluster.client.get("/search", params={"q": "search"})

    assert calls["count"] == 1
    assert response.status_code == 503
    del original_search


# ----------------------------------------------------------------------
# Out-of-sync replicas are not eligible
# ----------------------------------------------------------------------


def test_an_out_of_sync_replica_is_not_used_for_failover(
    replicated_cluster: Cluster,
) -> None:
    index_corpus(replicated_cluster)
    replica = engine_of(replicated_cluster, "shard-1-replica")
    replica.mark_out_of_sync("forced for the test")

    # While the primary is up, nothing changes.
    assert search(replicated_cluster, "search")["total"] == 2

    stop(replicated_cluster, "shard-1-primary")

    # With the primary gone and the only replica unsynchronized, the shard has
    # no eligible copy, so completeness wins over answering from stale state.
    assert replicated_cluster.client.get("/search", params={"q": "search"}).status_code == 503
    assert replicated_cluster.client.get("/ready").status_code == 503


def test_an_out_of_sync_replica_is_visible_in_cluster_status(
    replicated_cluster: Cluster,
) -> None:
    engine_of(replicated_cluster, "shard-1-replica").mark_out_of_sync("forced")

    shard = replicated_cluster.client.get("/cluster/status").json()["shards"][1]

    assert shard["copies"][1]["state"] == str(NodeState.OUT_OF_SYNC)
    assert shard["copies"][1]["ready"] is False
    # Still searchable and writable through the primary.
    assert shard["search_available"] is True
    assert shard["write_available"] is True


# ----------------------------------------------------------------------
# Recovery
# ----------------------------------------------------------------------


def test_a_replica_that_missed_writes_recovers_and_returns_to_service(
    replicated_cluster: Cluster,
) -> None:
    """The full recovery path, end to end over HTTP."""
    index_corpus(replicated_cluster)
    primary = engine_of(replicated_cluster, "shard-1-primary")
    replica = engine_of(replicated_cluster, "shard-1-replica")

    # The replica misses a mutation: the write fails, but the primary keeps it.
    replica.mark_out_of_sync("simulated missed replication")
    primary.index_document(Document(document_id="doc-1", text="text written while away"))
    assert primary.generation > replica.generation

    response = TestClient(replicated_cluster.nodes["shard-1-replica"]).post(
        "/internal/replica/resync"
    )

    assert response.status_code == 200
    assert replica.status().state is NodeState.READY
    assert replica.generation == primary.generation
    documents, _ = replica.export_snapshot()
    assert {document.document_id for document in documents} == {
        stored.document_id for stored in primary.export_snapshot()[0]
    }
    replica.validate()


def test_a_recovered_replica_can_serve_failover_again(
    replicated_cluster: Cluster,
) -> None:
    index_corpus(replicated_cluster)
    replica_app = replicated_cluster.nodes["shard-1-replica"]
    engine_of(replicated_cluster, "shard-1-replica").mark_out_of_sync("forced")

    assert TestClient(replica_app).post("/internal/replica/resync").status_code == 200
    before = search(replicated_cluster, "search")
    stop(replicated_cluster, "shard-1-primary")

    assert search(replicated_cluster, "search") == before


def test_a_replica_cannot_recover_from_a_missing_primary(
    replicated_cluster: Cluster,
) -> None:
    engine_of(replicated_cluster, "shard-1-replica").mark_out_of_sync("forced")
    stop(replicated_cluster, "shard-1-primary")

    response = TestClient(replicated_cluster.nodes["shard-1-replica"]).post(
        "/internal/replica/resync"
    )

    # There is no other authoritative source, and guessing would be worse.
    assert response.status_code == 503
    assert engine_of(replicated_cluster, "shard-1-replica").status().ready is False


def test_a_replica_starting_without_its_primary_refuses_to_serve(
    tmp_path: Path,
    shard_settings,
    start_cluster,  # type: ignore[no-untyped-def]
) -> None:
    """Startup must not claim readiness it cannot evidence."""
    from app.main import create_app

    settings = shard_settings(
        1, tmp_path, replica_role="replica", primary_url="http://nowhere-primary"
    )
    with TestClient(create_app(settings)) as replica:
        response = replica.get("/ready")

        assert response.status_code == 503
        assert "primary unreachable" in response.json()["detail"]
        # It also refuses the queries a coordinator would send it.
        assert replica.get("/internal/corpus-stats", params={"term": "x"}).status_code == 503


def test_a_replica_starting_behind_its_primary_resynchronizes(
    tmp_path: Path,
) -> None:
    """A restarted replica catches up before it is allowed back into service."""
    with ExitStack() as stack:
        from tests.conftest import launch_cluster

        running = launch_cluster(stack, tmp_path, replication_factor=2)
        index_corpus(running)
        primary_generation = engine_of(running, "shard-1-primary").generation

        # The replica's database is left behind while the primary moves on.
        replica_engine = engine_of(running, "shard-1-replica")
        replica_engine.mark_out_of_sync("simulated outage")
        engine_of(running, "shard-1-primary").index_document(
            Document(document_id="doc-5", text="written while the replica was away")
        )

        assert engine_of(running, "shard-1-primary").generation > primary_generation
        assert replica_engine.generation < engine_of(running, "shard-1-primary").generation

    # Restarting the whole cluster: the replica verifies itself, finds it is
    # behind, resynchronizes, and only then reports ready.
    with ExitStack() as stack:
        from tests.conftest import launch_cluster

        restarted = launch_cluster(stack, tmp_path, replication_factor=2)

        assert restarted.client.get("/ready").status_code == 200
        primary = engine_of(restarted, "shard-1-primary")
        replica = engine_of(restarted, "shard-1-replica")
        assert replica.generation == primary.generation
        assert replica.status().state is NodeState.READY
        assert replica.export_snapshot()[0] == primary.export_snapshot()[0]
        replica.validate()
