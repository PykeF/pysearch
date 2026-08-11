"""API integration tests for indexing and deleting documents."""

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


def test_indexing_a_new_document_returns_201(client: TestClient) -> None:
    response = client.put("/documents/doc-1", json={"text": "distributed search"})

    assert response.status_code == 201
    assert response.json() == {"document_id": "doc-1", "created": True}


def test_reindexing_an_existing_document_returns_200(client: TestClient) -> None:
    client.put("/documents/doc-1", json={"text": "distributed search"})

    response = client.put("/documents/doc-1", json={"text": "vector ranking"})

    assert response.status_code == 200
    assert response.json() == {"document_id": "doc-1", "created": False}


def test_a_blank_document_id_is_rejected(client: TestClient) -> None:
    response = client.put("/documents/%20", json={"text": "text"})

    assert response.status_code == 400
    assert "document_id" in response.json()["detail"]


def test_a_missing_body_is_rejected(client: TestClient) -> None:
    assert client.put("/documents/doc-1").status_code == 422


def test_a_body_without_text_is_rejected(client: TestClient) -> None:
    assert client.put("/documents/doc-1", json={}).status_code == 422


def test_an_empty_document_is_accepted(client: TestClient) -> None:
    response = client.put("/documents/doc-1", json={"text": ""})

    assert response.status_code == 201
    assert client.get("/index/stats").json()["document_count"] == 1


def test_deleting_a_document_returns_204(client: TestClient) -> None:
    client.put("/documents/doc-1", json={"text": "distributed search"})

    response = client.delete("/documents/doc-1")

    assert response.status_code == 204
    assert response.content == b""


def test_deleting_an_unknown_document_returns_404(client: TestClient) -> None:
    response = client.delete("/documents/doc-404")

    assert response.status_code == 404
    assert "doc-404" in response.json()["detail"]


def test_index_stats_reflect_the_indexed_corpus(client: TestClient) -> None:
    client.put("/documents/doc-1", json={"text": "distributed search"})
    client.put("/documents/doc-2", json={"text": "distributed systems scale"})

    stats = client.get("/index/stats").json()

    assert stats == {
        "document_count": 2,
        "unique_term_count": 4,
        "average_document_length": 2.5,
    }


def test_index_stats_on_an_empty_index(client: TestClient) -> None:
    assert client.get("/index/stats").json() == {
        "document_count": 0,
        "unique_term_count": 0,
        "average_document_length": 0.0,
    }


def test_each_application_gets_its_own_index(client: TestClient, settings: Settings) -> None:
    # Guards the isolation the client fixture depends on: index state must not
    # leak between applications through module-level globals.
    client.put("/documents/doc-1", json={"text": "distributed search"})

    other = TestClient(create_app(settings))

    assert other.get("/index/stats").json()["document_count"] == 0
