"""Distributed ranking must match an equivalent single-node corpus.

This is the correctness property that pays for the two-round fan-out. Shards
score with cluster-wide statistics, so ``idf`` and ``avgdl`` are the same values
a single node holding the whole corpus would use, and the only per-document
inputs left — ``tf`` and ``dl`` — are local by nature.

The contract asserted here is equivalent ranking and equivalent scores, not
bit-for-bit floating-point identity: ordering and tie-breaking are exact, and
scores are compared with a strict tolerance. Should the arithmetic happen to
land on identical bits, that is a property of the current implementation and
not something callers may rely on.
"""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.search.document import Document
from app.search.engine import SearchEngine
from app.storage.sqlite_store import IN_MEMORY, SqliteDocumentStore
from tests.conftest import Cluster

# Chosen so that the terms have genuinely different distributions: "search" is
# common, "bm25" is rare, and the documents differ in length. A term that is
# rare in the cluster but common on one shard is exactly the case where
# shard-local statistics would produce a different ranking.
CORPUS = {
    "doc-1": "search search search bm25 ranking",
    "doc-2": "search engines rank documents by relevance to a query about search",
    "doc-3": "distributed search",
    "doc-4": "sharding splits an index across nodes so that search can scale out",
    "doc-5": "an inverted index maps terms to documents",
    "doc-6": "bm25 saturates term frequency and normalises document length",
    "doc-12": "replication and failover keep a cluster available",
    "doc-13": "search",
}

QUERIES = (
    "search",
    "bm25",
    "search bm25",
    "index documents",
    "search search",
    "ranking function",
    "nonexistent",
)


@pytest.fixture
def single_node() -> Iterator[SearchEngine]:
    """One engine holding the entire corpus."""
    engine = SearchEngine(SqliteDocumentStore.open(IN_MEMORY))
    engine.initialize()
    for document_id, text in CORPUS.items():
        engine.index_document(Document(document_id=document_id, text=text))
    try:
        yield engine
    finally:
        engine.close()


@pytest.fixture
def loaded_cluster(cluster: Cluster) -> Cluster:
    """The same corpus, spread across three shards."""
    for document_id, text in CORPUS.items():
        response = cluster.client.put(f"/documents/{document_id}", json={"text": text})
        assert response.status_code == 201
    return cluster


def distributed_hits(client: TestClient, query: str) -> list[tuple[str, float]]:
    response = client.get("/search", params={"q": query, "limit": 100})
    assert response.status_code == 200, response.text
    payload = response.json()
    return [(hit["document_id"], hit["score"]) for hit in payload["results"]]


def single_node_hits(engine: SearchEngine, query: str) -> list[tuple[str, float]]:
    return [(hit.document_id, hit.score) for hit in engine.search(query, limit=100).results]


@pytest.mark.parametrize("query", QUERIES)
def test_distributed_ranking_matches_single_node(
    loaded_cluster: Cluster, single_node: SearchEngine, query: str
) -> None:
    distributed = distributed_hits(loaded_cluster.client, query)
    expected = single_node_hits(single_node, query)

    # Ordering and tie-breaking are exact.
    assert [document_id for document_id, _ in distributed] == [
        document_id for document_id, _ in expected
    ]
    # Scores are equal within a strict tolerance rather than to the last bit.
    for (_, actual_score), (_, expected_score) in zip(distributed, expected, strict=True):
        assert actual_score == pytest.approx(expected_score, rel=1e-12, abs=1e-12)


def test_the_totals_match_single_node(loaded_cluster: Cluster, single_node: SearchEngine) -> None:
    for query in QUERIES:
        response = loaded_cluster.client.get("/search", params={"q": query, "limit": 100})
        assert response.json()["total"] == single_node.search(query, limit=100).total


def test_cluster_statistics_match_the_single_node_corpus(
    loaded_cluster: Cluster, single_node: SearchEngine
) -> None:
    stats = loaded_cluster.client.get("/index/stats").json()
    expected = single_node.stats()

    assert stats["document_count"] == expected.document_count
    assert stats["average_document_length"] == pytest.approx(expected.average_document_length)


def test_shard_local_statistics_would_have_disagreed(loaded_cluster: Cluster) -> None:
    """Show the failure mode the global-statistics design exists to avoid.

    "bm25" appears in two documents cluster-wide but is the whole vocabulary
    story on its own shard, so a shard scoring with only its own statistics
    would compute a different idf, and the merged ranking would compare numbers
    from different scales. Asserting the shards genuinely differ is what makes
    the equivalence test above non-trivial.
    """
    stats = loaded_cluster.client.get("/index/stats").json()
    document_counts = [shard["document_count"] for shard in stats["shards"]]

    # The shards hold different numbers of documents and different vocabularies,
    # so their local N and avgdl are not interchangeable.
    assert len(set(document_counts)) > 1
    assert len({shard["average_document_length"] for shard in stats["shards"]}) > 1


# ----------------------------------------------------------------------
# Global top-k
# ----------------------------------------------------------------------


@pytest.mark.parametrize("limit", [1, 2, 3, 5, 8])
def test_global_top_k_matches_single_node_for_every_limit(
    loaded_cluster: Cluster, single_node: SearchEngine, limit: int
) -> None:
    """Local top-k per shard is sufficient for an exact global top-k.

    A document in the global top-k has fewer than k documents outranking it
    anywhere, so fewer than k outrank it on its own shard, so it is in that
    shard's local top-k and cannot be lost in the merge.
    """
    response = loaded_cluster.client.get("/search", params={"q": "search bm25", "limit": limit})
    distributed = [hit["document_id"] for hit in response.json()["results"]]
    expected = [hit.document_id for hit in single_node.search("search bm25", limit=limit).results]

    assert distributed == expected


def test_the_top_result_is_found_wherever_it_lives(
    loaded_cluster: Cluster, single_node: SearchEngine
) -> None:
    # doc-12 is the only document on shard 0; a query that should rank it first
    # proves the merge does not favour any particular shard.
    assert distributed_hits(loaded_cluster.client, "replication")[0][0] == "doc-12"
    assert distributed_hits(loaded_cluster.client, "failover")[0][0] == "doc-12"
    # And a query whose best answer sits on shard 2.
    assert distributed_hits(loaded_cluster.client, "sharding")[0][0] == "doc-4"


def test_several_top_results_from_one_shard_are_all_returned(
    loaded_cluster: Cluster, single_node: SearchEngine
) -> None:
    # doc-2, doc-3, doc-4 and doc-6 all live on shard 2, so a query matching
    # them exercises a shard returning more than one candidate.
    distributed = [
        document_id for document_id, _ in distributed_hits(loaded_cluster.client, "search")
    ]
    expected = [document_id for document_id, _ in single_node_hits(single_node, "search")]

    assert distributed == expected
    assert len(distributed) > 3


def test_ties_across_shards_break_on_document_id(cluster: Cluster) -> None:
    # Identical text, deliberately placed on shards 1, 2 and 0.
    for document_id in ("doc-1", "doc-2", "doc-12"):
        cluster.client.put(f"/documents/{document_id}", json={"text": "identical text"})

    results = cluster.client.get("/search", params={"q": "identical"}).json()["results"]

    assert len({hit["score"] for hit in results}) == 1
    assert [hit["document_id"] for hit in results] == ["doc-1", "doc-12", "doc-2"]
