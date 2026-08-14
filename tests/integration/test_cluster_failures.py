"""Failure behaviour of a real cluster.

Failures are injected structurally — a client that raises, a shard whose engine
is broken, a URL with nothing behind it — never by waiting. Nothing in this
module sleeps.
"""

import asyncio
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from pathlib import Path

import httpx2
import pytest
from fastapi.testclient import TestClient

from app.cluster.client import HttpShardClient, ShardClient
from app.cluster.errors import ShardTimeoutError, ShardUnavailableError
from app.core.config import Settings
from app.main import create_app
from app.search.document import Document
from app.search.index import CorpusStats
from tests.conftest import CLUSTER_SHARD_COUNT, make_coordinator_settings, make_shard_settings


class BrokenShardClient:
    """A shard that always fails, standing in for one that is down."""

    def __init__(self, error: Exception) -> None:
        self._error = error

    async def put_document(self, document: Document) -> bool:
        raise self._error

    async def delete_document(self, document_id: str) -> None:
        raise self._error

    async def search(self, query: str, limit: int, corpus_stats: CorpusStats) -> object:
        raise self._error

    async def corpus_stats(self, terms: Sequence[str]) -> CorpusStats:
        raise self._error

    async def index_stats(self) -> object:
        raise self._error

    async def is_ready(self) -> bool:
        return False


def cluster_with_broken_shard(
    stack: ExitStack, directory: Path, broken_shard: int, error: Exception
) -> TestClient:
    """A cluster where one shard is unreachable and the others are healthy."""
    clients: list[ShardClient] = []
    for shard_id in range(CLUSTER_SHARD_COUNT):
        if shard_id == broken_shard:
            clients.append(BrokenShardClient(error))
            continue
        settings = make_shard_settings(shard_id, directory)
        shard_app = create_app(settings)
        stack.enter_context(TestClient(shard_app))
        base_url = f"http://shard-{shard_id}"
        clients.append(
            HttpShardClient(
                base_url,
                httpx2.AsyncClient(
                    transport=httpx2.ASGITransport(app=shard_app), base_url=base_url
                ),
            )
        )

    coordinator = TestClient(create_app(make_coordinator_settings(), shard_clients=clients))
    stack.enter_context(coordinator)
    return coordinator


# ----------------------------------------------------------------------
# Search: fail the whole query
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [ShardUnavailableError("shard is down"), ShardTimeoutError("shard timed out")],
    ids=["unavailable", "timeout"],
)
def test_a_failed_shard_fails_the_search(tmp_path: Path, error: Exception) -> None:
    with ExitStack() as stack:
        client = cluster_with_broken_shard(stack, tmp_path, broken_shard=2, error=error)

        response = client.get("/search", params={"q": "search"})

        assert response.status_code == 503
        # Never a 200 with a short result list: an incomplete answer is not
        # returned as though it were complete.
        assert "2" in response.json()["detail"]


def test_a_failed_shard_fails_cluster_statistics(tmp_path: Path) -> None:
    with ExitStack() as stack:
        client = cluster_with_broken_shard(
            stack, tmp_path, broken_shard=0, error=ShardUnavailableError("down")
        )

        assert client.get("/index/stats").status_code == 503


def test_a_failed_shard_makes_the_cluster_unready(tmp_path: Path) -> None:
    with ExitStack() as stack:
        client = cluster_with_broken_shard(
            stack, tmp_path, broken_shard=1, error=ShardUnavailableError("down")
        )

        response = client.get("/ready")

        assert response.status_code == 503
        assert response.json()["status"] == "not_ready"
        assert "1" in response.json()["detail"]


def test_the_coordinator_stays_alive_when_a_shard_is_down(tmp_path: Path) -> None:
    # Liveness must not depend on peers: restarting the coordinator would not
    # fix a broken shard.
    with ExitStack() as stack:
        client = cluster_with_broken_shard(
            stack, tmp_path, broken_shard=1, error=ShardUnavailableError("down")
        )

        assert client.get("/health").status_code == 200


# ----------------------------------------------------------------------
# Writes: no rerouting
# ----------------------------------------------------------------------


def test_a_write_to_a_failed_owner_fails(tmp_path: Path) -> None:
    with ExitStack() as stack:
        client = cluster_with_broken_shard(
            stack, tmp_path, broken_shard=1, error=ShardUnavailableError("down")
        )

        # doc-1 is owned by shard 1, which is down.
        assert client.put("/documents/doc-1", json={"text": "search"}).status_code == 503
        # doc-12 is owned by shard 0, which is healthy: the failure is confined
        # to the documents the broken shard owns.
        assert client.put("/documents/doc-12", json={"text": "search"}).status_code == 201


def test_a_delete_on_a_failed_owner_fails(tmp_path: Path) -> None:
    with ExitStack() as stack:
        client = cluster_with_broken_shard(
            stack, tmp_path, broken_shard=1, error=ShardTimeoutError("timed out")
        )

        assert client.delete("/documents/doc-1").status_code == 503


# ----------------------------------------------------------------------
# Transport translation
# ----------------------------------------------------------------------


def test_an_unreachable_address_becomes_a_shard_unavailable_error() -> None:
    # Port 9 is the discard port: a real connection attempt that fails fast,
    # so the transport error path runs for real rather than being simulated.
    client = HttpShardClient(
        "http://127.0.0.1:9",
        httpx2.AsyncClient(timeout=httpx2.Timeout(0.25, connect=0.25)),
    )

    with pytest.raises(ShardUnavailableError):
        asyncio.run(client.corpus_stats(["search"]))


def test_a_degraded_shard_is_treated_as_unavailable(
    shard_settings: Callable[[int, Path], Settings],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A shard answering 503 is a failed shard, not a partial result."""
    clients: list[ShardClient] = []
    with ExitStack() as stack:
        for shard_id in range(CLUSTER_SHARD_COUNT):
            shard_app = create_app(shard_settings(shard_id, tmp_path))
            stack.enter_context(TestClient(shard_app))
            base_url = f"http://shard-{shard_id}"
            clients.append(
                HttpShardClient(
                    base_url,
                    httpx2.AsyncClient(
                        transport=httpx2.ASGITransport(app=shard_app), base_url=base_url
                    ),
                )
            )
            if shard_id == 2:
                degraded_app = shard_app

        coordinator = TestClient(create_app(make_coordinator_settings(), shard_clients=clients))
        stack.enter_context(coordinator)

        # Break shard 2's derived state so its next write degrades it.
        engine = degraded_app.state.engine

        def explode(*args: object, **kwargs: object) -> None:
            raise RuntimeError("derived state update failed")

        monkeypatch.setattr(engine._index, "add_document", explode)
        assert coordinator.put("/documents/doc-2", json={"text": "search"}).status_code == 503

        # From the coordinator's position a degraded shard is simply one that
        # cannot take part, so the cluster reports itself unready and refuses.
        assert coordinator.get("/ready").status_code == 503
        assert coordinator.get("/search", params={"q": "search"}).status_code == 503
