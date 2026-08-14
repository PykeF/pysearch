"""Distributed semantic search over a real cluster.

Every node is a separate application with its own database and its own vector
index, reached over real HTTP. The embedder is deterministic, so nothing here
loads a model or touches the network.
"""

from contextlib import ExitStack
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from app.search.engine import SearchEngine
from tests.conftest import CLUSTER_SHARD_COUNT, Cluster, FakeEmbedder, launch_cluster

# Routing is deterministic: doc-12 and doc-13 land on shard 0, doc-1 and doc-5
# on shard 1, the rest on shard 2 — so every logical shard holds part of this.
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


def semantic(cluster: Cluster, query: str, **params: object) -> dict[str, object]:
    response = cluster.client.get("/search/semantic", params={"q": query, **params})
    assert response.status_code == 200, response.text
    payload: dict[str, object] = response.json()
    return payload


def ranked(cluster: Cluster, query: str, **params: object) -> list[str]:
    results = semantic(cluster, query, **params)["results"]
    assert isinstance(results, list)
    return [hit["document_id"] for hit in results]


# ----------------------------------------------------------------------
# Enablement
# ----------------------------------------------------------------------


def test_semantic_is_disabled_by_default(cluster: Cluster) -> None:
    """Existing lexical-only deployments must not acquire a model."""
    status = cluster.client.get("/cluster/status").json()

    assert status["semantic"]["enabled"] is False
    assert cluster.client.get("/search/semantic", params={"q": "car"}).status_code == 503
    # And lexical search is entirely unaffected.
    assert cluster.client.get("/search", params={"q": "car"}).status_code == 200


def test_an_enabled_cluster_reports_its_embedding_space(semantic_cluster: Cluster) -> None:
    status = semantic_cluster.client.get("/cluster/status").json()

    assert status["semantic"]["enabled"] is True
    assert "fake-topics@v1" in status["semantic"]["identity"]


def test_every_node_reports_its_vector_count(semantic_cluster: Cluster) -> None:
    index_corpus(semantic_cluster)

    counts = []
    for shard_id in range(CLUSTER_SHARD_COUNT):
        node = TestClient(semantic_cluster.nodes[f"shard-{shard_id}-primary"])
        status = node.get("/internal/node-status").json()
        assert status["semantic_enabled"] is True
        counts.append(status["vector_count"])

    assert sum(counts) == len(CORPUS)


# ----------------------------------------------------------------------
# Distributed search
# ----------------------------------------------------------------------


def test_semantic_search_spans_every_shard(semantic_cluster: Cluster) -> None:
    index_corpus(semantic_cluster)

    payload = semantic(semantic_cluster, "vehicle motor repair", limit=100)

    # Every document is a candidate under a similarity, so "total" is the corpus.
    assert payload["total"] == len(CORPUS)
    results = payload["results"]
    assert isinstance(results, list)
    assert len(results) == len(CORPUS)


def test_the_best_match_wins_wherever_it_lives(semantic_cluster: Cluster) -> None:
    index_corpus(semantic_cluster)

    # doc-13 is on shard 0, doc-1 on shard 1, doc-5 on shard 2.
    assert ranked(semantic_cluster, "automobile motor")[0] in {"doc-1", "doc-13"}
    assert ranked(semantic_cluster, "replication failover node")[0] == "doc-12"
    assert ranked(semantic_cluster, "recipe food kitchen")[0] == "doc-5"


def test_results_are_ordered_by_descending_similarity(semantic_cluster: Cluster) -> None:
    index_corpus(semantic_cluster)

    results = semantic(semantic_cluster, "query ranking", limit=100)["results"]
    assert isinstance(results, list)
    scores = [hit["score"] for hit in results]

    assert scores == sorted(scores, reverse=True)


def test_the_limit_applies_across_the_cluster(semantic_cluster: Cluster) -> None:
    index_corpus(semantic_cluster)

    results = semantic(semantic_cluster, "vehicle", limit=2)["results"]
    assert isinstance(results, list)

    assert len(results) == 2


def test_ties_across_shards_break_on_document_id(semantic_cluster: Cluster) -> None:
    # Identical text on shards 0, 1 and 2 means identical vectors, so ordering
    # is decided purely by the tie-break rule.
    for document_id in ("doc-12", "doc-1", "doc-2"):
        semantic_cluster.client.put(f"/documents/{document_id}", json={"text": "identical text"})

    results = semantic(semantic_cluster, "identical text", limit=10)["results"]
    assert isinstance(results, list)

    assert len({hit["score"] for hit in results}) == 1
    assert [hit["document_id"] for hit in results] == ["doc-1", "doc-12", "doc-2"]


def test_repeated_searches_are_identical(semantic_cluster: Cluster) -> None:
    index_corpus(semantic_cluster)

    assert semantic(semantic_cluster, "vehicle repair") == semantic(
        semantic_cluster, "vehicle repair"
    )


def test_distributed_results_match_a_single_node_corpus(
    semantic_cluster: Cluster, embedder: FakeEmbedder
) -> None:
    """The point of one serving copy per shard: the cluster ranks like one node."""
    from app.search.document import Document
    from tests.conftest import InMemoryDocumentStore

    index_corpus(semantic_cluster)

    single = SearchEngine(InMemoryDocumentStore(), embedder=FakeEmbedder())
    single.initialize()
    for document_id, text in CORPUS.items():
        single.index_document(Document(document_id=document_id, text=text))

    for query in ("vehicle motor", "query ranking", "recipe kitchen", "replication"):
        distributed = semantic(semantic_cluster, query, limit=100)
        expected = single.semantic_search(single.embed_query(query), limit=100)

        assert [hit["document_id"] for hit in distributed["results"]] == [
            hit.document_id for hit in expected.results
        ]
        np.testing.assert_allclose(
            [hit["score"] for hit in distributed["results"]],
            [hit.score for hit in expected.results],
            rtol=1e-6,
            atol=1e-6,
        )


# ----------------------------------------------------------------------
# Mutation
# ----------------------------------------------------------------------


def test_replacement_changes_the_semantic_ranking(semantic_cluster: Cluster) -> None:
    index_corpus(semantic_cluster)
    assert ranked(semantic_cluster, "recipe kitchen")[0] == "doc-5"

    semantic_cluster.client.put("/documents/doc-5", json={"text": "shard replica cluster failover"})

    assert ranked(semantic_cluster, "recipe kitchen")[0] != "doc-5"
    assert ranked(semantic_cluster, "replication node")[0] in {"doc-5", "doc-12"}


def test_deletion_removes_a_document_from_semantic_results(semantic_cluster: Cluster) -> None:
    index_corpus(semantic_cluster)

    assert semantic_cluster.client.delete("/documents/doc-5").status_code == 204

    assert "doc-5" not in ranked(semantic_cluster, "recipe kitchen", limit=100)
    assert semantic(semantic_cluster, "anything", limit=100)["total"] == len(CORPUS) - 1


# ----------------------------------------------------------------------
# Replication and failover
# ----------------------------------------------------------------------


def test_both_copies_build_their_own_vectors(replicated_semantic_cluster: Cluster) -> None:
    """Documents replicate; vectors are derived independently on each copy."""
    for document_id, text in CORPUS.items():
        replicated_semantic_cluster.client.put(f"/documents/{document_id}", json={"text": text})

    for shard_id in range(CLUSTER_SHARD_COUNT):
        primary = engine_of(replicated_semantic_cluster, f"shard-{shard_id}-primary")
        replica = engine_of(replicated_semantic_cluster, f"shard-{shard_id}-replica")
        assert primary.vector_count == replica.vector_count


def test_semantic_search_survives_losing_a_primary(
    replicated_semantic_cluster: Cluster,
) -> None:
    for document_id, text in CORPUS.items():
        replicated_semantic_cluster.client.put(f"/documents/{document_id}", json={"text": text})
    before = semantic(replicated_semantic_cluster, "vehicle motor repair", limit=100)

    engine_of(replicated_semantic_cluster, "shard-1-primary").close()

    after = semantic(replicated_semantic_cluster, "vehicle motor repair", limit=100)

    # Same ordering, and scores equal to within tolerance rather than bit-identical.
    assert [hit["document_id"] for hit in after["results"]] == [
        hit["document_id"] for hit in before["results"]
    ]
    np.testing.assert_allclose(
        [hit["score"] for hit in after["results"]],
        [hit["score"] for hit in before["results"]],
        rtol=1e-6,
        atol=1e-6,
    )


def test_an_out_of_sync_replica_cannot_rescue_semantic_search(
    replicated_semantic_cluster: Cluster,
) -> None:
    for document_id, text in CORPUS.items():
        replicated_semantic_cluster.client.put(f"/documents/{document_id}", json={"text": text})

    engine_of(replicated_semantic_cluster, "shard-1-replica").mark_out_of_sync("forced")
    engine_of(replicated_semantic_cluster, "shard-1-primary").close()

    response = replicated_semantic_cluster.client.get("/search/semantic", params={"q": "vehicle"})

    # Completeness wins: an unsynchronized copy must not stand in for a shard.
    assert response.status_code == 503


def test_losing_every_copy_of_a_shard_fails_semantic_search(
    replicated_semantic_cluster: Cluster,
) -> None:
    for document_id, text in CORPUS.items():
        replicated_semantic_cluster.client.put(f"/documents/{document_id}", json={"text": text})

    engine_of(replicated_semantic_cluster, "shard-1-primary").close()
    engine_of(replicated_semantic_cluster, "shard-1-replica").close()

    response = replicated_semantic_cluster.client.get("/search/semantic", params={"q": "vehicle"})

    assert response.status_code == 503
    assert "1" in response.json()["detail"]


def test_resynchronization_restores_semantic_equivalence(
    replicated_semantic_cluster: Cluster,
) -> None:
    for document_id, text in CORPUS.items():
        replicated_semantic_cluster.client.put(f"/documents/{document_id}", json={"text": text})
    replica = engine_of(replicated_semantic_cluster, "shard-1-replica")
    primary = engine_of(replicated_semantic_cluster, "shard-1-primary")
    replica.mark_out_of_sync("forced")

    response = TestClient(replicated_semantic_cluster.nodes["shard-1-replica"]).post(
        "/internal/replica/resync"
    )

    assert response.status_code == 200
    # Resync transfers documents; the vectors are rebuilt from them.
    assert replica.vector_count == primary.vector_count
    query = replica.embed_query("vehicle motor")
    left = replica.semantic_search(query, limit=10).results
    right = primary.semantic_search(primary.embed_query("vehicle motor"), limit=10).results
    assert [hit.document_id for hit in left] == [hit.document_id for hit in right]
    replica.validate()


# ----------------------------------------------------------------------
# Model compatibility
# ----------------------------------------------------------------------


def test_a_shard_refuses_a_query_from_a_different_model(semantic_cluster: Cluster) -> None:
    """Vectors from different models measure different spaces."""
    index_corpus(semantic_cluster)
    node = TestClient(semantic_cluster.nodes["shard-1-primary"])

    response = node.post(
        "/internal/search/semantic",
        json={
            "vector": [0.0, 0.0, 0.0, 0.0, 1.0],
            "limit": 5,
            "identity": {
                "implementation": "fake",
                "model_id": "some-other-model",
                "model_revision": "v9",
                "dimension": 5,
                "normalization": "l2",
            },
        },
    )

    assert response.status_code == 409
    assert "identity mismatch" in response.json()["detail"]


def test_a_shard_accepts_a_query_from_its_own_model(semantic_cluster: Cluster) -> None:
    index_corpus(semantic_cluster)
    node = TestClient(semantic_cluster.nodes["shard-1-primary"])
    identity = engine_of(semantic_cluster, "shard-1-primary").semantic_identity
    assert identity is not None

    response = node.post(
        "/internal/search/semantic",
        json={
            "vector": [1.0, 0.0, 0.0, 0.0, 0.0],
            "limit": 5,
            "identity": {
                "implementation": identity.implementation,
                "model_id": identity.model_id,
                "model_revision": identity.model_revision,
                "dimension": identity.dimension,
                "normalization": identity.normalization,
            },
        },
    )

    assert response.status_code == 200


# ----------------------------------------------------------------------
# Restart
# ----------------------------------------------------------------------


def test_semantic_state_is_rebuilt_after_a_full_restart(
    tmp_path: Path, embedder: FakeEmbedder
) -> None:
    with ExitStack() as stack:
        running = launch_cluster(stack, tmp_path, embedder=embedder)
        for document_id, text in CORPUS.items():
            running.client.put(f"/documents/{document_id}", json={"text": text})
        before = running.client.get(
            "/search/semantic", params={"q": "vehicle motor", "limit": 100}
        ).json()

    # Vectors are not persisted; every node re-embeds its documents on start.
    with ExitStack() as stack:
        restarted = launch_cluster(stack, tmp_path, embedder=FakeEmbedder())
        after = restarted.client.get(
            "/search/semantic", params={"q": "vehicle motor", "limit": 100}
        ).json()

    assert [hit["document_id"] for hit in after["results"]] == [
        hit["document_id"] for hit in before["results"]
    ]
    np.testing.assert_allclose(
        [hit["score"] for hit in after["results"]],
        [hit["score"] for hit in before["results"]],
        rtol=1e-6,
        atol=1e-6,
    )


def test_lexical_search_is_unchanged_when_semantic_is_enabled(
    semantic_cluster: Cluster, cluster: Cluster
) -> None:
    """Semantic is additive: BM25 answers identically with or without it."""
    index_corpus(semantic_cluster)
    index_corpus(cluster)

    for query in ("engine", "query relevance", "pasta"):
        with_semantic = semantic_cluster.client.get("/search", params={"q": query}).json()
        without = cluster.client.get("/search", params={"q": query}).json()
        assert with_semantic == without
