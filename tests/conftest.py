"""Shared test fixtures.

Every fixture that touches storage uses pytest's ``tmp_path``, so tests never
depend on developer-machine state and never share a database.
"""

from collections.abc import Callable, Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import httpx2
import pytest
from fastapi.testclient import TestClient

from app.cluster.client import HttpShardClient
from app.core.config import Settings
from app.main import create_app
from app.search.document import Document
from app.search.engine import SearchEngine
from app.storage.base import DocumentStore
from app.storage.sqlite_store import SqliteDocumentStore


class InMemoryDocumentStore:
    """A dictionary-backed :class:`DocumentStore` for tests that need no durability.

    Its existence is also the proof that the engine depends on the protocol
    rather than on SQLite.
    """

    def __init__(self) -> None:
        self._documents: dict[str, Document] = {}

    def put(self, document: Document) -> bool:
        created = document.document_id not in self._documents
        self._documents[document.document_id] = document
        return created

    def get(self, document_id: str) -> Document | None:
        return self._documents.get(document_id)

    def delete(self, document_id: str) -> bool:
        return self._documents.pop(document_id, None) is not None

    def iter_documents(self) -> Iterator[Document]:
        for document_id in sorted(self._documents):
            yield self._documents[document_id]

    def count(self) -> int:
        return len(self._documents)

    def close(self) -> None:
        return None


@pytest.fixture
def storage_path(tmp_path: Path) -> Path:
    """A database path unique to the running test."""
    return tmp_path / "pysearch.db"


@pytest.fixture
def settings(storage_path: Path) -> Settings:
    """Fully explicit settings, so tests never depend on the ambient environment."""
    return Settings(
        app_name="pysearch-test",
        environment="test",
        log_level="WARNING",
        storage_path=storage_path,
    )


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    """An HTTP client bound to an isolated application with its own database.

    Entering the context manager runs the lifespan hook, which is what opens
    storage and rebuilds the index.
    """
    with TestClient(create_app(settings)) as test_client:
        yield test_client


@pytest.fixture
def sqlite_store(storage_path: Path) -> Iterator[SqliteDocumentStore]:
    """A file-backed store, closed when the test finishes."""
    store = SqliteDocumentStore.open(storage_path)
    try:
        yield store
    finally:
        store.close()


@pytest.fixture(params=["sqlite", "in_memory"])
def document_store(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[DocumentStore]:
    """Each store implementation in turn, for the storage contract tests."""
    store: DocumentStore
    if request.param == "sqlite":
        store = SqliteDocumentStore.open(tmp_path / "contract.db")
    else:
        store = InMemoryDocumentStore()
    try:
        yield store
    finally:
        store.close()


@pytest.fixture
def engine() -> Iterator[SearchEngine]:
    """An initialized engine over an in-memory store, for behaviour tests."""
    search_engine = SearchEngine(InMemoryDocumentStore())
    search_engine.initialize()
    try:
        yield search_engine
    finally:
        search_engine.close()


# ----------------------------------------------------------------------
# Cluster harness
# ----------------------------------------------------------------------

CLUSTER_SHARD_COUNT = 3


@dataclass
class Cluster:
    """A running coordinator and the shard applications behind it."""

    client: TestClient
    shard_paths: tuple[Path, ...]


def make_shard_settings(shard_id: int, directory: Path) -> Settings:
    """Settings for one shard node, with its own database."""
    return Settings(
        app_name=f"pysearch-shard-{shard_id}",
        environment="test",
        log_level="WARNING",
        node_role="shard",
        shard_id=shard_id,
        shard_count=CLUSTER_SHARD_COUNT,
        storage_path=directory / f"shard-{shard_id}.db",
    )


def make_coordinator_settings() -> Settings:
    """Settings for a coordinator over the shard topology."""
    return Settings(
        app_name="pysearch-coordinator",
        environment="test",
        log_level="WARNING",
        node_role="coordinator",
        shard_count=CLUSTER_SHARD_COUNT,
        shard_urls=",".join(f"http://shard-{n}" for n in range(CLUSTER_SHARD_COUNT)),
    )


def launch_cluster(stack: ExitStack, directory: Path) -> Cluster:
    """Bring up shard applications and a coordinator wired to them.

    Each shard is a real application with its own engine and its own SQLite
    file, reached over real HTTP through an ASGI transport: routing,
    serialization, status codes and error translation all execute, without
    sockets, ports or timing.
    """
    shard_clients: list[HttpShardClient] = []
    shard_paths: list[Path] = []

    for shard_id in range(CLUSTER_SHARD_COUNT):
        settings = make_shard_settings(shard_id, directory)
        shard_paths.append(settings.storage_path)
        shard_app = create_app(settings)
        # Entering the TestClient runs the shard's lifespan, which opens its
        # database and rebuilds its index.
        stack.enter_context(TestClient(shard_app))

        base_url = f"http://shard-{shard_id}"
        http = httpx2.AsyncClient(transport=httpx2.ASGITransport(app=shard_app), base_url=base_url)
        shard_clients.append(HttpShardClient(base_url, http))

    coordinator = TestClient(create_app(make_coordinator_settings(), shard_clients=shard_clients))
    stack.enter_context(coordinator)
    return Cluster(client=coordinator, shard_paths=tuple(shard_paths))


@pytest.fixture
def cluster(tmp_path: Path) -> Iterator[Cluster]:
    """A three-shard cluster with an empty corpus."""
    with ExitStack() as stack:
        yield launch_cluster(stack, tmp_path)


@pytest.fixture
def start_cluster() -> Callable[[ExitStack, Path], Cluster]:
    """The cluster launcher, for tests that start and stop clusters themselves."""
    return launch_cluster


@pytest.fixture
def shard_settings() -> Callable[[int, Path], Settings]:
    """Settings builder for a standalone shard node."""
    return make_shard_settings
