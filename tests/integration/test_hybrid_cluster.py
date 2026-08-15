"""Distributed hybrid search over a real cluster.

Every node is a separate application with its own database, index and vector
index, reached over real HTTP. The embedder is deterministic, so nothing here
loads a model.
"""

from contextlib import ExitStack
from pathlib import Path

import numpy as np

from app.search.document import Document
from app.search.engine import SearchEngine
from tests.conftest import Cluster, FakeEmbedder, InMemoryDocumentStore, launch_cluster

# Routing is deterministic: doc-12/doc-13 land on shard 0, doc-1/doc-5 on
# shard 1, the rest on shard 2.
CORPUS = {
    "doc-1": "car engine repair and vehicle maintenance",
    "doc-2": "search ranking and retrieval relevance",
    "doc-3": "bm25 index query relevance scoring",
    "doc-5": "cooking pasta recipe in the kitchen",
    "doc-12": "shard replica cluster node distributed replication",
    "doc-13": "motor vehicle automobile engine",
}


def engine_of(cluster: Cluster, name: str) -> SearchEngine:
    engine: SearchEngine = cluster.nodes[name].state.engine
    return engine


def index_corpus(cluster: Cluster) -> None:
    for document_id, text in CORPUS.items():
        response = cluster.client.put(f"/documents/{document_id}", json={"text": text})
        assert response.status_code in (200, 201), response.text


def hybrid(cluster: Cluster, query: str, **params: object) -> dict[str, object]:
    response = cluster.client.get("/search/hybrid", params={"q": query, **params})
    assert response.status_code == 200, response.text
    payload: dict[str, object] = response.json()
    return payload


def ranked(cluster: Cluster, query: str, **params: object) -> list[str]:
    results = hybrid(cluster, query, **params)["results"]
    assert isinstance(results, list)
    return [hit["document_id"] for hit in results]


# ----------------------------------------------------------------------
# Availability
# ----------------------------------------------------------------------


def test_hybrid_is_unavailable_without_semantic_search(cluster: Cluster) -> None:
    """503 and a reason, rather than silently returning BM25 as "hybrid"."""
    index_corpus(cluster)

    response = cluster.client.get("/search/hybrid", params={"q": "vehicle"})

    assert response.status_code == 503
    assert "hybrid search requires" in response.json()["detail"]
    # The two independent modes are untouched.
    assert cluster.client.get("/search", params={"q": "engine"}).status_code == 200


def test_hybrid_completes_over_the_cluster(semantic_cluster: Cluster) -> None:
    """Both fan-outs share one operation lock; this would hang if reacquired."""
    index_corpus(semantic_cluster)

    payload = hybrid(semantic_cluster, "vehicle motor repair", limit=5)

    assert payload["total"] > 0


# ----------------------------------------------------------------------
# Fusion across shards
# ----------------------------------------------------------------------


def test_both_signals_contribute_across_shards(semantic_cluster: Cluster) -> None:
    index_corpus(semantic_cluster)

    results = hybrid(semantic_cluster, "vehicle motor repair", limit=10)["results"]
    assert isinstance(results, list)

    # doc-1 is on shard 1 and doc-13 on shard 0, so the fused top must span them.
    assert {results[0]["document_id"], results[1]["document_id"]} == {"doc-1", "doc-13"}
    assert all(
        hit["lexical_rank"] is not None or hit["semantic_rank"] is not None for hit in results
    )


def test_a_document_appears_once_however_many_lists_found_it(
    semantic_cluster: Cluster,
) -> None:
    index_corpus(semantic_cluster)

    document_ids = ranked(semantic_cluster, "vehicle motor repair", limit=10)

    assert len(document_ids) == len(set(document_ids))


def test_results_are_ordered_by_fusion_score(semantic_cluster: Cluster) -> None:
    index_corpus(semantic_cluster)

    results = hybrid(semantic_cluster, "search relevance", limit=10)["results"]
    assert isinstance(results, list)
    scores = [hit["score"] for hit in results]

    assert scores == sorted(scores, reverse=True)


def test_identical_documents_come_out_in_document_id_order(
    semantic_cluster: Cluster,
) -> None:
    """Determinism survives all the way through fusion.

    Both input rankings already break their own ties on document id, so three
    identical documents arrive at fusion holding ranks 1, 2 and 3 in each list
    rather than tying. Their fusion scores therefore differ — but the resulting
    order is still exactly id-ascending, which is the property that matters and
    the one that would break if any stage depended on iteration order.
    """
    for document_id in ("doc-12", "doc-1", "doc-2"):
        semantic_cluster.client.put(f"/documents/{document_id}", json={"text": "identical text"})

    results = hybrid(semantic_cluster, "identical text", limit=10)["results"]
    assert isinstance(results, list)

    assert [hit["document_id"] for hit in results] == ["doc-1", "doc-12", "doc-2"]
    # Each holds the same rank in both lists, so each score is 2/(k+rank).
    for rank, hit in enumerate(results, start=1):
        assert hit["lexical_rank"] == rank
        assert hit["semantic_rank"] == rank
        assert hit["score"] == 2 / (60 + rank)


def test_hybrid_is_deterministic(semantic_cluster: Cluster) -> None:
    index_corpus(semantic_cluster)

    assert hybrid(semantic_cluster, "vehicle repair") == hybrid(semantic_cluster, "vehicle repair")


def test_distributed_hybrid_matches_a_single_node_corpus(semantic_cluster: Cluster) -> None:
    """Composing the two correct distributed paths gives the single-node answer."""
    index_corpus(semantic_cluster)

    single = SearchEngine(InMemoryDocumentStore(), embedder=FakeEmbedder())
    single.initialize()
    for document_id, text in CORPUS.items():
        single.index_document(Document(document_id=document_id, text=text))

    for query in ("vehicle motor", "search relevance", "recipe kitchen", "replication"):
        distributed = hybrid(semantic_cluster, query, limit=10)
        expected = single.hybrid_search(query, limit=10)

        assert [hit["document_id"] for hit in distributed["results"]] == [
            hit.document_id for hit in expected.results
        ]
        np.testing.assert_allclose(
            [hit["score"] for hit in distributed["results"]],
            [hit.score for hit in expected.results],
            rtol=1e-9,
            atol=1e-12,
        )


def test_an_empty_query_returns_nothing(semantic_cluster: Cluster) -> None:
    index_corpus(semantic_cluster)

    assert hybrid(semantic_cluster, "")["total"] == 0
    assert hybrid(semantic_cluster, "!!! ???")["total"] == 0


def test_validation_matches_the_other_search_endpoints(semantic_cluster: Cluster) -> None:
    assert semantic_cluster.client.get("/search/hybrid").status_code == 422
    assert (
        semantic_cluster.client.get("/search/hybrid", params={"q": "a", "limit": 0}).status_code
        == 422
    )


# ----------------------------------------------------------------------
# Explainability
# ----------------------------------------------------------------------


def test_ranks_are_always_present(semantic_cluster: Cluster) -> None:
    index_corpus(semantic_cluster)

    hit = hybrid(semantic_cluster, "vehicle motor repair")["results"][0]

    assert "lexical_rank" in hit
    assert "semantic_rank" in hit
    # The two ranks reconstruct the score exactly.
    assert hit["score"] == 1 / (60 + hit["lexical_rank"]) + 1 / (60 + hit["semantic_rank"])


def test_explain_adds_the_underlying_scores(semantic_cluster: Cluster) -> None:
    index_corpus(semantic_cluster)

    plain = hybrid(semantic_cluster, "vehicle motor repair")["results"][0]
    explained = hybrid(semantic_cluster, "vehicle motor repair", explain="true")["results"][0]

    assert plain["lexical_score"] is None
    assert explained["lexical_score"] is not None
    assert explained["semantic_score"] is not None
    assert explained["score"] == plain["score"]


# ----------------------------------------------------------------------
# Replication and failover
# ----------------------------------------------------------------------


def test_hybrid_survives_losing_a_primary(replicated_semantic_cluster: Cluster) -> None:
    for document_id, text in CORPUS.items():
        replicated_semantic_cluster.client.put(f"/documents/{document_id}", json={"text": text})
    before = hybrid(replicated_semantic_cluster, "vehicle motor repair", limit=10, explain="true")

    engine_of(replicated_semantic_cluster, "shard-1-primary").close()

    after = hybrid(replicated_semantic_cluster, "vehicle motor repair", limit=10, explain="true")

    # Identical ordering and identical ranks; scores within tolerance.
    assert [hit["document_id"] for hit in after["results"]] == [
        hit["document_id"] for hit in before["results"]
    ]
    assert [hit["lexical_rank"] for hit in after["results"]] == [
        hit["lexical_rank"] for hit in before["results"]
    ]
    assert [hit["semantic_rank"] for hit in after["results"]] == [
        hit["semantic_rank"] for hit in before["results"]
    ]
    np.testing.assert_allclose(
        [hit["score"] for hit in after["results"]],
        [hit["score"] for hit in before["results"]],
        rtol=1e-9,
        atol=1e-12,
    )


def test_an_out_of_sync_replica_cannot_rescue_hybrid(
    replicated_semantic_cluster: Cluster,
) -> None:
    for document_id, text in CORPUS.items():
        replicated_semantic_cluster.client.put(f"/documents/{document_id}", json={"text": text})

    engine_of(replicated_semantic_cluster, "shard-1-replica").mark_out_of_sync("forced")
    engine_of(replicated_semantic_cluster, "shard-1-primary").close()

    response = replicated_semantic_cluster.client.get("/search/hybrid", params={"q": "vehicle"})

    assert response.status_code == 503


def test_losing_every_copy_of_a_shard_fails_hybrid(
    replicated_semantic_cluster: Cluster,
) -> None:
    for document_id, text in CORPUS.items():
        replicated_semantic_cluster.client.put(f"/documents/{document_id}", json={"text": text})

    engine_of(replicated_semantic_cluster, "shard-1-primary").close()
    engine_of(replicated_semantic_cluster, "shard-1-replica").close()

    response = replicated_semantic_cluster.client.get("/search/hybrid", params={"q": "vehicle"})

    assert response.status_code == 503
    assert "1" in response.json()["detail"]


def test_a_retrieval_path_failing_fails_the_whole_request(
    semantic_cluster: Cluster,
) -> None:
    """ "Hybrid" asserts both signals took part, so one of them is not enough."""
    index_corpus(semantic_cluster)
    engine_of(semantic_cluster, "shard-1-primary").close()

    assert semantic_cluster.client.get("/search/hybrid", params={"q": "vehicle"}).status_code == 503
    # And both single-mode endpoints fail too, for the same shard reason.
    assert semantic_cluster.client.get("/search", params={"q": "vehicle"}).status_code == 503
    assert (
        semantic_cluster.client.get("/search/semantic", params={"q": "vehicle"}).status_code == 503
    )


# ----------------------------------------------------------------------
# Mutations and recovery
# ----------------------------------------------------------------------


def test_replacement_and_deletion_flow_through(semantic_cluster: Cluster) -> None:
    def hit_for(document_id: str, query: str) -> dict[str, object]:
        results = hybrid(semantic_cluster, query, limit=10, explain="true")["results"]
        assert isinstance(results, list)
        return next(hit for hit in results if hit["document_id"] == document_id)

    index_corpus(semantic_cluster)
    assert ranked(semantic_cluster, "recipe kitchen")[0] == "doc-5"
    before = hit_for("doc-5", "recipe kitchen")
    assert before["lexical_rank"] is not None

    semantic_cluster.client.put("/documents/doc-5", json={"text": "shard replica cluster"})

    # Both signals lost the old content: no lexical match at all, and the
    # embedding has moved off the old topic.
    after = hit_for("doc-5", "recipe kitchen")
    assert after["lexical_rank"] is None
    assert after["semantic_score"] < before["semantic_score"]
    assert ranked(semantic_cluster, "cluster replica")[0] in {"doc-5", "doc-12"}

    assert semantic_cluster.client.delete("/documents/doc-5").status_code == 204
    assert "doc-5" not in ranked(semantic_cluster, "cluster replica", limit=10)


def test_hybrid_is_restored_by_recovery(tmp_path: Path, embedder: FakeEmbedder) -> None:
    """Hybrid keeps no state of its own, so a restart restores it for free."""
    with ExitStack() as stack:
        running = launch_cluster(stack, tmp_path, embedder=embedder)
        for document_id, text in CORPUS.items():
            running.client.put(f"/documents/{document_id}", json={"text": text})
        before = running.client.get(
            "/search/hybrid", params={"q": "vehicle motor", "limit": 10}
        ).json()

    with ExitStack() as stack:
        restarted = launch_cluster(stack, tmp_path, embedder=FakeEmbedder())
        after = restarted.client.get(
            "/search/hybrid", params={"q": "vehicle motor", "limit": 10}
        ).json()

    assert [hit["document_id"] for hit in after["results"]] == [
        hit["document_id"] for hit in before["results"]
    ]
    np.testing.assert_allclose(
        [hit["score"] for hit in after["results"]],
        [hit["score"] for hit in before["results"]],
        rtol=1e-9,
        atol=1e-12,
    )
