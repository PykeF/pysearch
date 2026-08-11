"""API integration tests for the health endpoint."""

from fastapi.testclient import TestClient


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_is_documented_in_the_openapi_schema(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()

    assert "/health" in schema["paths"]
    assert "get" in schema["paths"]["/health"]


def test_unknown_route_returns_404(client: TestClient) -> None:
    assert client.get("/does-not-exist").status_code == 404
