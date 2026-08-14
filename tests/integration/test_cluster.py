"""End-to-end distributed tests over real shard applications.

Each shard here is a genuine FastAPI application with its own engine and its own
SQLite file, and the coordinator reaches them over real HTTP — request routing,
JSON serialization, status codes and error translation all execute. The
transport is `ASGITransport` rather than a socket, which keeps the suite fast
and free of ports and timing, and the equivalent verification over separate OS
processes is done live rather than here.
"""

from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from tests.conftest import CLUSTER_SHARD_COUNT as SHARD_COUNT
from tests.conftest import Cluster

# Routing is deterministic, so this corpus lands in a known spread: doc-12 on
# shard 0, doc-1 and doc-5 on shard 1, the rest on shard 2. It is chosen to
# cover every shard, which is what makes the fan-out tests meaningful — and the
# spread is visibly uneven at this size, which is honest about modulo sharding.
CORPUS = {
    "doc-1": "Distributed systems make search scalable across many machines.",
    "doc-2": "Search engines rank documents by relevance to a query.",
    "doc-3": "BM25 is the ranking function used by most lexical search engines.",
    "doc-4": "An inverted index maps each term to the documents containing it.",
    "doc-5": "Sharding splits an index so that each node holds part of the data.",
    "doc-6": "Cooking pasta well requires salted boiling water.",
    "doc-12": "Query latency depends on the slowest node in a fan-out.",
}


def index_corpus(client: TestClient, corpus: dict[str, str] | None = None) -> None:
    for document_id, text in (corpus or CORPUS).items():
        response = client.put(f"/documents/{document_id}", json={"text": text})
        assert response.status_code in (200, 201)


def search(client: TestClient, query: str, **params: object) -> dict[str, object]:
    response = client.get("/search", params={"q": query, **params})
    assert response.status_code == 200, response.text
    payload: dict[str, object] = response.json()
    return payload


def ranked_ids(client: TestClient, query: str, **params: object) -> list[str]:
    results = search(client, query, **params)["results"]
    assert isinstance(results, list)
    return [hit["document_id"] for hit in results]


# ----------------------------------------------------------------------
# Topology
# ----------------------------------------------------------------------


def test_the_cluster_reports_ready(cluster: Cluster) -> None:
    response = cluster.client.get("/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_the_coordinator_is_alive_independently_of_shards(cluster: Cluster) -> None:
    assert cluster.client.get("/health").json() == {"status": "ok"}


def test_documents_are_spread_across_shards(cluster: Cluster) -> None:
    index_corpus(cluster.client)

    stats = cluster.client.get("/index/stats").json()

    assert stats["document_count"] == len(CORPUS)
    assert stats["shard_count"] == SHARD_COUNT
    # Every shard holds some of the corpus, and the parts sum to the whole.
    assert sum(shard["document_count"] for shard in stats["shards"]) == len(CORPUS)
    assert all(shard["document_count"] > 0 for shard in stats["shards"])


def test_each_shard_uses_its_own_database(cluster: Cluster) -> None:
    index_corpus(cluster.client)

    assert len(set(cluster.shard_paths)) == SHARD_COUNT
    assert all(path.exists() for path in cluster.shard_paths)


def test_the_write_response_names_the_owning_shard(cluster: Cluster) -> None:
    response = cluster.client.put("/documents/doc-1", json={"text": "search"})

    assert response.status_code == 201
    assert response.json() == {"document_id": "doc-1", "created": True, "shard_id": 1}


def test_a_shard_exposes_no_public_search(
    shard_settings: Callable[[int, Path], Settings], tmp_path: Path
) -> None:
    # Querying one shard would return silently partial results, so the path does
    # not exist on a shard at all.
    with TestClient(create_app(shard_settings(0, tmp_path))) as shard:
        assert shard.get("/search", params={"q": "search"}).status_code == 404
        assert shard.put("/documents/doc-1", json={"text": "x"}).status_code == 404
        assert shard.get("/health").status_code == 200
        assert shard.get("/ready").status_code == 200


# ----------------------------------------------------------------------
# Search
# ----------------------------------------------------------------------


def test_search_returns_documents_from_several_shards(cluster: Cluster) -> None:
    index_corpus(cluster.client)

    payload = search(cluster.client, "search")

    assert payload["total"] == 3
    assert set(ranked_ids(cluster.client, "search")) == {"doc-1", "doc-2", "doc-3"}


def test_results_are_ordered_by_descending_score(cluster: Cluster) -> None:
    index_corpus(cluster.client)

    results = search(cluster.client, "search engines")["results"]
    assert isinstance(results, list)
    scores = [hit["score"] for hit in results]

    assert scores == sorted(scores, reverse=True)


def test_the_limit_applies_across_the_cluster(cluster: Cluster) -> None:
    index_corpus(cluster.client)

    payload = search(cluster.client, "search", limit=2)
    results = payload["results"]
    assert isinstance(results, list)

    assert payload["total"] == 3
    assert len(results) == 2


def test_an_empty_query_returns_no_results(cluster: Cluster) -> None:
    index_corpus(cluster.client)

    assert search(cluster.client, "")["total"] == 0
    assert search(cluster.client, "!!!")["total"] == 0


def test_a_query_matching_nothing_returns_no_results(cluster: Cluster) -> None:
    index_corpus(cluster.client)

    assert search(cluster.client, "astrophysics")["total"] == 0


def test_search_validation_is_preserved(cluster: Cluster) -> None:
    assert cluster.client.get("/search").status_code == 422
    assert cluster.client.get("/search", params={"q": "a", "limit": 0}).status_code == 422
    assert cluster.client.get("/search", params={"q": "a", "limit": 101}).status_code == 422


def test_repeated_searches_are_identical(cluster: Cluster) -> None:
    index_corpus(cluster.client)

    assert search(cluster.client, "search engines") == search(cluster.client, "search engines")


# ----------------------------------------------------------------------
# Mutation through the coordinator
# ----------------------------------------------------------------------


def test_index_update_search(cluster: Cluster) -> None:
    index_corpus(cluster.client)
    assert search(cluster.client, "pasta")["total"] == 1

    response = cluster.client.put(
        "/documents/doc-6", json={"text": "Replication keeps copies of a shard."}
    )

    assert response.status_code == 200
    assert response.json()["created"] is False
    assert search(cluster.client, "pasta")["total"] == 0
    assert ranked_ids(cluster.client, "replication") == ["doc-6"]


def test_index_delete_search(cluster: Cluster) -> None:
    index_corpus(cluster.client)

    assert cluster.client.delete("/documents/doc-1").status_code == 204

    assert set(ranked_ids(cluster.client, "search")) == {"doc-2", "doc-3"}
    assert cluster.client.get("/index/stats").json()["document_count"] == len(CORPUS) - 1


def test_deleting_an_unknown_document_returns_404(cluster: Cluster) -> None:
    assert cluster.client.delete("/documents/doc-404").status_code == 404


def test_a_blank_document_id_is_rejected(cluster: Cluster) -> None:
    assert cluster.client.put("/documents/%20", json={"text": "x"}).status_code == 400


# ----------------------------------------------------------------------
# Restart
# ----------------------------------------------------------------------


def test_documents_survive_a_full_cluster_restart(
    start_cluster: Callable[[ExitStack, Path], Cluster], tmp_path: Path
) -> None:
    with ExitStack() as stack:
        cluster = start_cluster(stack, tmp_path)
        index_corpus(cluster.client)
        before = search(cluster.client, "search engines")
        before_stats = cluster.client.get("/index/stats").json()

    # Every process is gone; only the shard databases remain.
    with ExitStack() as stack:
        restarted = start_cluster(stack, tmp_path)
        assert restarted.client.get("/ready").status_code == 200
        assert search(restarted.client, "search engines") == before
        assert restarted.client.get("/index/stats").json() == before_stats


def test_routing_is_stable_across_a_coordinator_restart(
    start_cluster: Callable[[ExitStack, Path], Cluster], tmp_path: Path
) -> None:
    with ExitStack() as stack:
        cluster = start_cluster(stack, tmp_path)
        index_corpus(cluster.client)
        placement = {
            document_id: cluster.client.put(
                f"/documents/{document_id}", json={"text": text}
            ).json()["shard_id"]
            for document_id, text in CORPUS.items()
        }

    with ExitStack() as stack:
        restarted = start_cluster(stack, tmp_path)
        # A fresh coordinator with no memory of the previous one resolves every
        # document to the same shard, which is what stable hashing buys.
        after = {
            document_id: restarted.client.put(
                f"/documents/{document_id}", json={"text": text}
            ).json()["shard_id"]
            for document_id, text in CORPUS.items()
        }

    assert after == placement


def test_a_restarted_shard_recovers_its_own_documents(
    start_cluster: Callable[[ExitStack, Path], Cluster],
    shard_settings: Callable[[int, Path], Settings],
    tmp_path: Path,
) -> None:
    with ExitStack() as stack:
        cluster = start_cluster(stack, tmp_path)
        index_corpus(cluster.client)
        stats_before = cluster.client.get("/index/stats").json()["shards"]

    # Restarting shard 1 alone: it rebuilds from its own database, and the
    # coordinator does nothing to help it.
    with TestClient(create_app(shard_settings(1, tmp_path))) as shard:
        assert shard.get("/ready").status_code == 200
        recovered = shard.get("/internal/stats").json()

    assert recovered["document_count"] == stats_before[1]["document_count"]
    assert recovered["unique_term_count"] == stats_before[1]["unique_term_count"]
