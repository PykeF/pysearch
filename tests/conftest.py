"""Shared test fixtures.

Every fixture that touches storage uses pytest's ``tmp_path``, so tests never
depend on developer-machine state and never share a database.
"""

from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.cluster.client import HttpShardClient
from app.cluster.replication import HttpNodeLink
from app.cluster.topology import build_topology
from app.core.config import Settings
from app.main import create_app
from app.search.analysis import analyze
from app.search.document import Document
from app.search.engine import SearchEngine
from app.semantic.embedder import L2_NORMALIZATION, SemanticIdentity, normalize_rows
from app.storage.base import DocumentStore
from app.storage.sqlite_store import SqliteDocumentStore


class InMemoryDocumentStore:
    """A dictionary-backed :class:`DocumentStore` for tests that need no durability.

    Its existence is also the proof that the engine depends on the protocol
    rather than on SQLite.
    """

    def __init__(self) -> None:
        self._documents: dict[str, Document] = {}
        self._generation = 0

    def put(self, document: Document, generation: int) -> bool:
        created = document.document_id not in self._documents
        self._documents[document.document_id] = document
        self._generation = generation
        return created

    def get(self, document_id: str) -> Document | None:
        return self._documents.get(document_id)

    def delete(self, document_id: str, generation: int) -> bool:
        existed = self._documents.pop(document_id, None) is not None
        self._generation = generation
        return existed

    def replace_all(self, documents: Iterable[Document], generation: int) -> None:
        self._documents = {document.document_id: document for document in documents}
        self._generation = generation

    def generation(self) -> int:
        return self._generation

    def iter_documents(self) -> Iterator[Document]:
        for document_id in sorted(self._documents):
            yield self._documents[document_id]

    def count(self) -> int:
        return len(self._documents)

    def close(self) -> None:
        return None


class FakeEmbedder:
    """A deterministic embedder that needs no model and no network.

    Text is projected onto a handful of fixed "topic" words: each dimension
    counts how often that topic's words appear, and the result is normalized.
    That is not real semantics, but it is *stable* semantics — documents about
    the same topic land near each other and unrelated ones do not — which is
    exactly what the vector plumbing needs in order to be tested without loading
    a transformer into three hundred tests.
    """

    TOPICS: tuple[tuple[str, ...], ...] = (
        ("car", "cars", "automobile", "engine", "vehicle", "motor", "repair", "maintenance"),
        ("search", "query", "ranking", "retrieval", "index", "relevance", "bm25"),
        ("cook", "cooking", "pasta", "recipe", "food", "kitchen", "boiling", "water"),
        ("shard", "replica", "cluster", "node", "distributed", "replication", "failover"),
    )

    def __init__(self, model_id: str = "fake-topics", revision: str = "v1") -> None:
        self._identity = SemanticIdentity(
            implementation="fake",
            model_id=model_id,
            model_revision=revision,
            dimension=len(self.TOPICS) + 1,
            normalization=L2_NORMALIZATION,
        )

    @property
    def identity(self) -> SemanticIdentity:
        return self._identity

    def embed_documents(self, texts: Sequence[str]) -> Any:
        if not texts:
            return np.zeros((0, self._identity.dimension), dtype=np.float32)
        rows = np.stack([self._project(text) for text in texts])
        return normalize_rows(rows)

    def embed_query(self, text: str) -> Any:
        return self.embed_documents([text])[0]

    def _project(self, text: str) -> Any:
        counts = np.zeros(self._identity.dimension, dtype=np.float32)
        tokens = analyze(text)
        for token in tokens:
            for position, topic in enumerate(self.TOPICS):
                if token in topic:
                    counts[position] += 1.0
        # A constant last dimension keeps a text with no topic word from
        # embedding to the zero vector, which has no direction at all.
        counts[-1] = 0.25
        return counts


@pytest.fixture
def embedder() -> FakeEmbedder:
    """A deterministic embedder for every test that needs the semantic path."""
    return FakeEmbedder()


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


@pytest.fixture
def semantic_engine(embedder: FakeEmbedder) -> Iterator[SearchEngine]:
    """An initialized engine with semantic retrieval enabled."""
    search_engine = SearchEngine(InMemoryDocumentStore(), embedder=embedder)
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
    """A running coordinator and the physical shard nodes behind it."""

    client: TestClient
    shard_paths: tuple[Path, ...]
    nodes: dict[str, FastAPI]


def make_shard_settings(
    shard_id: int,
    directory: Path,
    replica_role: str = "primary",
    replicas: Sequence[str] = (),
    primary_url: str = "",
) -> Settings:
    """Settings for one physical shard node, with its own database."""
    suffix = "primary" if replica_role == "primary" else "replica"
    return Settings(
        app_name=f"pysearch-shard-{shard_id}-{suffix}",
        node_id=f"shard-{shard_id}-{suffix}",
        environment="test",
        log_level="WARNING",
        node_role="shard",
        shard_id=shard_id,
        shard_count=CLUSTER_SHARD_COUNT,
        replica_role=replica_role,  # type: ignore[arg-type]
        replica_urls=",".join(replicas),
        primary_url=primary_url,
        storage_path=directory / f"shard-{shard_id}-{suffix}.db",
    )


def make_coordinator_settings(replication_factor: int = 1) -> Settings:
    """Settings for a coordinator over the shard topology."""
    replica_urls = ""
    if replication_factor > 1:
        replica_urls = ";".join(
            f"http://shard-{shard}-replica" for shard in range(CLUSTER_SHARD_COUNT)
        )
    return Settings(
        app_name="pysearch-coordinator",
        environment="test",
        log_level="WARNING",
        node_role="coordinator",
        shard_count=CLUSTER_SHARD_COUNT,
        shard_urls=",".join(f"http://shard-{n}-primary" for n in range(CLUSTER_SHARD_COUNT)),
        replica_urls=replica_urls,
    )


def asgi_client(app: FastAPI, base_url: str) -> httpx2.AsyncClient:
    """An async client that reaches an in-process application over real HTTP."""
    return httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), base_url=base_url)


def launch_cluster(
    stack: ExitStack,
    directory: Path,
    replication_factor: int = 1,
    embedder: object | None = None,
) -> Cluster:
    """Bring up shard nodes and a coordinator wired to them.

    Each physical node is a real application with its own engine and its own
    SQLite file. Nodes reach each other over real HTTP through ASGI transports —
    ``TestClient`` is itself an httpx client, so a primary's synchronous
    replication call runs the replica's full request stack — which means
    routing, replication, serialization, status codes and error translation all
    execute, without sockets, ports or timing.
    """
    primaries: list[HttpShardClient] = []
    replicas: list[list[HttpShardClient]] = []
    shard_paths: list[Path] = []
    nodes: dict[str, FastAPI] = {}
    apps_by_url: dict[str, FastAPI] = {}

    def link_to(url: str) -> HttpNodeLink:
        """A synchronous peer link into whichever in-process app owns that URL."""
        return HttpNodeLink(url, TestClient(apps_by_url[url]))

    for shard_id in range(CLUSTER_SHARD_COUNT):
        primary_url = f"http://shard-{shard_id}-primary"
        replica_url = f"http://shard-{shard_id}-replica"
        replicated = replication_factor > 1

        primary_settings = make_shard_settings(
            shard_id,
            directory,
            replica_role="primary",
            replicas=[replica_url] if replicated else [],
        )
        shard_paths.append(primary_settings.storage_path)
        primary_app = create_app(primary_settings, node_links=link_to, embedder=embedder)
        apps_by_url[primary_url] = primary_app
        nodes[f"shard-{shard_id}-primary"] = primary_app

        replica_app: FastAPI | None = None
        if replicated:
            replica_settings = make_shard_settings(
                shard_id, directory, replica_role="replica", primary_url=primary_url
            )
            shard_paths.append(replica_settings.storage_path)
            replica_app = create_app(replica_settings, node_links=link_to, embedder=embedder)
            apps_by_url[replica_url] = replica_app
            nodes[f"shard-{shard_id}-replica"] = replica_app

        # The primary starts first: a replica verifies itself against its
        # primary during its own startup, so the primary has to be answering by
        # then or the replica will refuse to serve.
        stack.enter_context(TestClient(primary_app))
        if replica_app is not None:
            stack.enter_context(TestClient(replica_app))

        primaries.append(HttpShardClient(primary_url, asgi_client(primary_app, primary_url)))
        if replica_app is not None:
            replicas.append([HttpShardClient(replica_url, asgi_client(replica_app, replica_url))])

    topology = build_topology(primaries=primaries, replicas=replicas)
    coordinator = TestClient(
        create_app(
            make_coordinator_settings(replication_factor),
            topology=topology,
            embedder=embedder,
        )
    )
    stack.enter_context(coordinator)
    return Cluster(client=coordinator, shard_paths=tuple(shard_paths), nodes=nodes)


@pytest.fixture
def cluster(tmp_path: Path) -> Iterator[Cluster]:
    """A three-shard cluster with one copy per shard and an empty corpus."""
    with ExitStack() as stack:
        yield launch_cluster(stack, tmp_path)


@pytest.fixture
def semantic_cluster(tmp_path: Path, embedder: FakeEmbedder) -> Iterator[Cluster]:
    """A three-shard cluster with semantic retrieval enabled on every node."""
    with ExitStack() as stack:
        yield launch_cluster(stack, tmp_path, embedder=embedder)


@pytest.fixture
def replicated_semantic_cluster(tmp_path: Path, embedder: FakeEmbedder) -> Iterator[Cluster]:
    """A replicated cluster with semantic retrieval enabled on every copy."""
    with ExitStack() as stack:
        yield launch_cluster(stack, tmp_path, replication_factor=2, embedder=embedder)


@pytest.fixture
def replicated_cluster(tmp_path: Path) -> Iterator[Cluster]:
    """A three-shard cluster with a primary and a replica per logical shard."""
    with ExitStack() as stack:
        yield launch_cluster(stack, tmp_path, replication_factor=2)


@pytest.fixture
def start_cluster() -> Callable[..., Cluster]:
    """The cluster launcher, for tests that start and stop clusters themselves."""
    return launch_cluster


@pytest.fixture
def shard_settings() -> Callable[[int, Path], Settings]:
    """Settings builder for a standalone shard node."""
    return make_shard_settings
