"""Shared test fixtures.

Every fixture that touches storage uses pytest's ``tmp_path``, so tests never
depend on developer-machine state and never share a database.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

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
