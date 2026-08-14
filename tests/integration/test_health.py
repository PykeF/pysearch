"""API integration tests for the liveness and readiness endpoints."""

import pytest
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


def test_ready_returns_200_after_startup_recovery(client: TestClient) -> None:
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready", "detail": "ready"}


def test_ready_returns_503_when_the_engine_is_degraded(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = client.app.state.engine  # type: ignore[attr-defined]

    def explode(*args: object, **kwargs: object) -> None:
        raise RuntimeError("derived state update failed")

    monkeypatch.setattr(engine._index, "add_document", explode)
    assert client.put("/documents/doc-1", json={"text": "search"}).status_code == 503

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
    assert "degraded" in response.json()["detail"]


def test_health_still_reports_ok_while_degraded(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Liveness answers "should this process be restarted", and a degraded engine
    # is exactly the case where a restart would help — but that is the
    # orchestrator's decision to make from /ready, not a reason to fail /health.
    engine = client.app.state.engine  # type: ignore[attr-defined]

    def explode(*args: object, **kwargs: object) -> None:
        raise RuntimeError("derived state update failed")

    monkeypatch.setattr(engine._index, "add_document", explode)
    assert client.put("/documents/doc-1", json={"text": "search"}).status_code == 503

    assert client.get("/health").status_code == 200


def test_a_degraded_engine_refuses_search_and_mutations(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = client.app.state.engine  # type: ignore[attr-defined]

    def explode(*args: object, **kwargs: object) -> None:
        raise RuntimeError("derived state update failed")

    monkeypatch.setattr(engine._index, "add_document", explode)
    assert client.put("/documents/doc-1", json={"text": "search"}).status_code == 503

    assert client.get("/search", params={"q": "search"}).status_code == 503
    assert client.get("/index/stats").status_code == 503
    assert client.put("/documents/doc-2", json={"text": "search"}).status_code == 503
    assert client.delete("/documents/doc-1").status_code == 503
