"""Tests for the engine's durability contract: write ordering, failure, degradation."""

from collections.abc import Iterable, Iterator

import pytest

from app.search.document import Document
from app.search.engine import SearchEngine
from app.search.errors import DocumentNotFoundError, EngineNotReadyError, IndexInvariantError
from app.storage.base import DocumentStore
from app.storage.errors import StorageError


class RecordingStore:
    """A store that records the order of calls and can be made to fail."""

    def __init__(self) -> None:
        self._documents: dict[str, Document] = {}
        self._generation = 0
        self.calls: list[str] = []
        self.fail_on_put = False
        self.fail_on_delete = False

    def put(self, document: Document, generation: int) -> bool:
        self.calls.append("put")
        if self.fail_on_put:
            raise StorageError("storage is unavailable")
        created = document.document_id not in self._documents
        self._documents[document.document_id] = document
        self._generation = generation
        return created

    def get(self, document_id: str) -> Document | None:
        return self._documents.get(document_id)

    def delete(self, document_id: str, generation: int) -> bool:
        self.calls.append("delete")
        if self.fail_on_delete:
            raise StorageError("storage is unavailable")
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
        self.calls.append("close")


@pytest.fixture
def store() -> RecordingStore:
    return RecordingStore()


@pytest.fixture
def persistent_engine(store: RecordingStore) -> SearchEngine:
    engine = SearchEngine(store)
    engine.initialize()
    return engine


# ----------------------------------------------------------------------
# Lifecycle
# ----------------------------------------------------------------------


def test_an_uninitialized_engine_refuses_every_request(store: RecordingStore) -> None:
    engine = SearchEngine(store)

    assert engine.status().ready is False
    with pytest.raises(EngineNotReadyError):
        engine.search("search", limit=10)
    with pytest.raises(EngineNotReadyError):
        engine.stats()
    with pytest.raises(EngineNotReadyError):
        engine.index_document(Document(document_id="doc-1", text="search"))
    with pytest.raises(EngineNotReadyError):
        engine.delete_document("doc-1")


def test_initialize_reports_what_it_rebuilt(store: RecordingStore) -> None:
    store.put(Document(document_id="doc-1", text="distributed search"), generation=1)
    store.put(Document(document_id="doc-2", text="ranking"), generation=2)

    report = SearchEngine(store).initialize()

    assert report.document_count == 2
    assert report.duration_seconds >= 0.0


def test_initialize_rebuilds_derived_state_from_storage(store: RecordingStore) -> None:
    store.put(Document(document_id="doc-1", text="distributed search"), generation=1)

    engine = SearchEngine(store)
    engine.initialize()

    assert engine.stats().document_count == 1
    assert engine.search("search", limit=10).results[0].document_id == "doc-1"
    engine.validate()


def test_an_initialized_engine_is_ready(persistent_engine: SearchEngine) -> None:
    status = persistent_engine.status()

    assert status.ready is True
    assert status.detail == "ready"


def test_close_stops_the_engine_serving(persistent_engine: SearchEngine) -> None:
    persistent_engine.close()

    assert persistent_engine.status().ready is False
    with pytest.raises(EngineNotReadyError):
        persistent_engine.search("search", limit=10)


# ----------------------------------------------------------------------
# Write ordering
# ----------------------------------------------------------------------


def test_storage_is_written_before_derived_state(persistent_engine: SearchEngine) -> None:
    store: RecordingStore = persistent_engine._store  # type: ignore[assignment]
    store.calls.clear()

    persistent_engine.index_document(Document(document_id="doc-1", text="search"))

    # The durable write is the first thing that happens, and the derived state
    # reflects it only afterwards.
    assert store.calls == ["put"]
    assert store.get("doc-1") is not None
    assert persistent_engine.stats().document_count == 1


def test_deletion_hits_storage_before_derived_state(persistent_engine: SearchEngine) -> None:
    persistent_engine.index_document(Document(document_id="doc-1", text="search"))
    store: RecordingStore = persistent_engine._store  # type: ignore[assignment]
    store.calls.clear()

    persistent_engine.delete_document("doc-1")

    assert store.calls == ["delete"]
    assert store.get("doc-1") is None
    assert persistent_engine.stats().document_count == 0


def test_deleting_an_unknown_document_leaves_derived_state_untouched(
    persistent_engine: SearchEngine,
) -> None:
    persistent_engine.index_document(Document(document_id="doc-1", text="search"))

    with pytest.raises(DocumentNotFoundError):
        persistent_engine.delete_document("doc-404")

    assert persistent_engine.stats().document_count == 1
    assert persistent_engine.status().ready is True
    persistent_engine.validate()


# ----------------------------------------------------------------------
# Storage failure: nothing changes anywhere
# ----------------------------------------------------------------------


def test_a_failed_write_reports_failure_and_changes_nothing(
    persistent_engine: SearchEngine, store: RecordingStore
) -> None:
    store.fail_on_put = True

    with pytest.raises(StorageError):
        persistent_engine.index_document(Document(document_id="doc-1", text="search"))

    assert store.count() == 0
    assert persistent_engine.stats().document_count == 0
    # A storage failure before the commit is not a degradation: derived state
    # still matches the authoritative corpus.
    assert persistent_engine.status().ready is True
    persistent_engine.validate()


def test_a_failed_delete_reports_failure_and_changes_nothing(
    persistent_engine: SearchEngine, store: RecordingStore
) -> None:
    persistent_engine.index_document(Document(document_id="doc-1", text="search"))
    store.fail_on_delete = True

    with pytest.raises(StorageError):
        persistent_engine.delete_document("doc-1")

    assert store.count() == 1
    assert persistent_engine.stats().document_count == 1
    assert persistent_engine.status().ready is True
    persistent_engine.validate()


# ----------------------------------------------------------------------
# Degradation: the commit landed but derived state did not follow
# ----------------------------------------------------------------------


def _break_index_writes(engine: SearchEngine, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the in-memory index fail, simulating a post-commit derived-state failure."""

    def explode(*args: object, **kwargs: object) -> None:
        raise RuntimeError("derived state update failed")

    monkeypatch.setattr(engine._index, "add_document", explode)
    monkeypatch.setattr(engine._index, "remove_document", explode)


def test_a_post_commit_failure_degrades_the_engine(
    persistent_engine: SearchEngine, store: RecordingStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    _break_index_writes(persistent_engine, monkeypatch)

    with pytest.raises(EngineNotReadyError):
        persistent_engine.index_document(Document(document_id="doc-1", text="search"))

    status = persistent_engine.status()
    assert status.ready is False
    assert "degraded" in status.detail
    # No compensating rollback: the durable write really happened.
    assert store.count() == 1


def test_a_degraded_engine_refuses_mutations_search_and_stats(
    persistent_engine: SearchEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    _break_index_writes(persistent_engine, monkeypatch)
    with pytest.raises(EngineNotReadyError):
        persistent_engine.index_document(Document(document_id="doc-1", text="search"))

    with pytest.raises(EngineNotReadyError):
        persistent_engine.index_document(Document(document_id="doc-2", text="search"))
    with pytest.raises(EngineNotReadyError):
        persistent_engine.delete_document("doc-1")
    with pytest.raises(EngineNotReadyError):
        persistent_engine.search("search", limit=10)
    with pytest.raises(EngineNotReadyError):
        persistent_engine.stats()


def test_a_post_commit_delete_failure_degrades_the_engine(
    persistent_engine: SearchEngine, store: RecordingStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    persistent_engine.index_document(Document(document_id="doc-1", text="search"))
    _break_index_writes(persistent_engine, monkeypatch)

    with pytest.raises(EngineNotReadyError):
        persistent_engine.delete_document("doc-1")

    assert persistent_engine.status().ready is False
    # The durable delete stands; storage remains authoritative.
    assert store.count() == 0


def test_reinitialization_repairs_a_degraded_engine(
    persistent_engine: SearchEngine, store: RecordingStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    _break_index_writes(persistent_engine, monkeypatch)
    with pytest.raises(EngineNotReadyError):
        persistent_engine.index_document(Document(document_id="doc-1", text="search"))
    assert persistent_engine.status().ready is False

    monkeypatch.undo()
    persistent_engine.initialize()

    # The write that degraded the engine was durable, so recovery finds it.
    assert persistent_engine.status().ready is True
    assert persistent_engine.stats().document_count == 1
    assert persistent_engine.search("search", limit=10).results[0].document_id == "doc-1"
    persistent_engine.validate()


def test_validate_still_works_while_degraded(
    persistent_engine: SearchEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    _break_index_writes(persistent_engine, monkeypatch)
    with pytest.raises(EngineNotReadyError):
        persistent_engine.index_document(Document(document_id="doc-1", text="search"))

    # Diagnosis must stay available precisely when something has gone wrong;
    # here it reports the drift the degradation was raised for.
    with pytest.raises(IndexInvariantError, match="cache and the index hold different documents"):
        persistent_engine.validate()


# ----------------------------------------------------------------------
# Cross-layer invariants
# ----------------------------------------------------------------------


def test_validate_detects_a_cache_that_disagrees_with_storage(
    persistent_engine: SearchEngine, store: RecordingStore
) -> None:
    persistent_engine.index_document(Document(document_id="doc-1", text="search"))
    # Mutate storage behind the engine's back.
    store.put(Document(document_id="doc-1", text="something else entirely"), generation=1)

    with pytest.raises(IndexInvariantError, match="differs from storage"):
        persistent_engine.validate()


def test_validate_detects_a_document_missing_from_the_cache(
    persistent_engine: SearchEngine, store: RecordingStore
) -> None:
    store.put(Document(document_id="ghost", text="never indexed"), generation=0)

    with pytest.raises(IndexInvariantError, match="storage holds"):
        persistent_engine.validate()


def test_the_engine_depends_only_on_the_store_protocol(store: RecordingStore) -> None:
    # RecordingStore never imports or inherits anything from the SQLite module.
    checked: DocumentStore = store
    engine = SearchEngine(checked)
    engine.initialize()

    assert engine.status().ready is True
