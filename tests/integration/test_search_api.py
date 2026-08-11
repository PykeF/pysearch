"""API integration tests for search, covering the full index-to-results path."""

from typing import Any

from fastapi.testclient import TestClient

CORPUS = {
    "doc-1": "Distributed systems make search scalable.",
    "doc-2": "Search engines rank documents by relevance.",
    "doc-3": "BM25 is a ranking function used by search engines.",
    "doc-4": "Cooking pasta requires boiling water.",
}


def index_corpus(client: TestClient) -> None:
    for document_id, text in CORPUS.items():
        client.put(f"/documents/{document_id}", json={"text": text})


def search(client: TestClient, query: str, **params: Any) -> dict[str, Any]:
    response = client.get("/search", params={"q": query, **params})
    assert response.status_code == 200
    payload: dict[str, Any] = response.json()
    return payload


def test_index_then_search(client: TestClient) -> None:
    index_corpus(client)

    payload = search(client, "search")

    assert payload["query"] == "search"
    assert payload["total"] == 3
    assert {hit["document_id"] for hit in payload["results"]} == {"doc-1", "doc-2", "doc-3"}


def test_results_are_ordered_by_descending_score(client: TestClient) -> None:
    index_corpus(client)

    scores = [hit["score"] for hit in search(client, "search engines")["results"]]

    assert scores == sorted(scores, reverse=True)
    assert all(score > 0 for score in scores)


def test_results_include_the_document_text(client: TestClient) -> None:
    index_corpus(client)

    hit = search(client, "pasta")["results"][0]

    assert hit["document_id"] == "doc-4"
    assert hit["text"] == CORPUS["doc-4"]


def test_index_then_update_then_search(client: TestClient) -> None:
    index_corpus(client)
    assert search(client, "pasta")["total"] == 1

    client.put("/documents/doc-4", json={"text": "Sharding distributes an index across nodes."})

    assert search(client, "pasta")["total"] == 0
    assert search(client, "sharding")["results"][0]["document_id"] == "doc-4"


def test_index_then_delete_then_search(client: TestClient) -> None:
    index_corpus(client)
    assert search(client, "search")["total"] == 3

    assert client.delete("/documents/doc-1").status_code == 204

    payload = search(client, "search")
    assert payload["total"] == 2
    assert {hit["document_id"] for hit in payload["results"]} == {"doc-2", "doc-3"}


def test_deletion_updates_reported_statistics(client: TestClient) -> None:
    index_corpus(client)
    before = client.get("/index/stats").json()

    client.delete("/documents/doc-4")
    after = client.get("/index/stats").json()

    assert after["document_count"] == before["document_count"] - 1
    assert after["unique_term_count"] < before["unique_term_count"]


def test_limit_truncates_results_without_changing_the_total(client: TestClient) -> None:
    index_corpus(client)

    payload = search(client, "search", limit=2)

    assert payload["total"] == 3
    assert len(payload["results"]) == 2


def test_the_default_limit_is_ten(client: TestClient) -> None:
    for n in range(12):
        client.put(f"/documents/doc-{n:02d}", json={"text": "search"})

    payload = search(client, "search")

    assert payload["total"] == 12
    assert len(payload["results"]) == 10


def test_a_query_matching_nothing_returns_an_empty_result_set(client: TestClient) -> None:
    index_corpus(client)

    payload = search(client, "astrophysics")

    assert payload["total"] == 0
    assert payload["results"] == []


def test_an_empty_query_returns_no_results(client: TestClient) -> None:
    index_corpus(client)

    payload = search(client, "")

    assert payload["total"] == 0
    assert payload["results"] == []


def test_a_punctuation_only_query_returns_no_results(client: TestClient) -> None:
    index_corpus(client)

    assert search(client, "!!! ???")["total"] == 0


def test_searching_an_empty_index_returns_no_results(client: TestClient) -> None:
    assert search(client, "search")["total"] == 0


def test_a_missing_query_parameter_is_rejected(client: TestClient) -> None:
    assert client.get("/search").status_code == 422


def test_out_of_range_limits_are_rejected(client: TestClient) -> None:
    assert client.get("/search", params={"q": "search", "limit": 0}).status_code == 422
    assert client.get("/search", params={"q": "search", "limit": 101}).status_code == 422


def test_a_non_numeric_limit_is_rejected(client: TestClient) -> None:
    assert client.get("/search", params={"q": "search", "limit": "many"}).status_code == 422


def test_queries_are_case_and_punctuation_insensitive(client: TestClient) -> None:
    index_corpus(client)

    assert search(client, "SEARCH!")["total"] == search(client, "search")["total"] == 3


def test_ties_are_broken_by_document_id(client: TestClient) -> None:
    for document_id in ("doc-c", "doc-a", "doc-b"):
        client.put(f"/documents/{document_id}", json={"text": "identical text"})

    results = search(client, "identical")["results"]

    assert [hit["document_id"] for hit in results] == ["doc-a", "doc-b", "doc-c"]


def test_repeated_searches_return_identical_payloads(client: TestClient) -> None:
    index_corpus(client)

    assert search(client, "search engines") == search(client, "search engines")
