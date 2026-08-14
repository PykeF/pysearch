"""Contract tests for the document store.

Every test here runs against each implementation in turn via the
``document_store`` fixture, so the SQLite store and the in-memory test double
are held to the same behaviour. Durability tests, which only mean something for
a real file, are separate at the bottom.
"""

from pathlib import Path

import pytest

from app.search.document import Document
from app.storage.base import DocumentStore
from app.storage.errors import StorageError, StorageInitializationError
from app.storage.sqlite_store import SqliteDocumentStore


def document(document_id: str, text: str = "distributed search") -> Document:
    return Document(document_id=document_id, text=text)


# ----------------------------------------------------------------------
# Contract
# ----------------------------------------------------------------------


def test_put_then_get_returns_the_document(document_store: DocumentStore) -> None:
    assert document_store.put(document("doc-1"), generation=1) is True

    stored = document_store.get("doc-1")

    assert stored == document("doc-1")


def test_get_returns_none_for_an_unknown_document(document_store: DocumentStore) -> None:
    assert document_store.get("doc-404") is None


def test_put_reports_creation_then_replacement(document_store: DocumentStore) -> None:
    assert document_store.put(document("doc-1", "first"), generation=1) is True
    assert document_store.put(document("doc-1", "second"), generation=1) is False


def test_replacement_overwrites_the_text(document_store: DocumentStore) -> None:
    document_store.put(document("doc-1", "first"), generation=1)
    document_store.put(document("doc-1", "second"), generation=1)

    stored = document_store.get("doc-1")

    assert stored is not None
    assert stored.text == "second"
    assert document_store.count() == 1


def test_delete_reports_whether_the_document_existed(document_store: DocumentStore) -> None:
    document_store.put(document("doc-1"), generation=1)

    assert document_store.delete("doc-1", generation=2) is True
    assert document_store.delete("doc-1", generation=2) is False


def test_delete_removes_the_document(document_store: DocumentStore) -> None:
    document_store.put(document("doc-1"), generation=1)
    document_store.delete("doc-1", generation=2)

    assert document_store.get("doc-1") is None
    assert document_store.count() == 0


def test_count_tracks_stored_documents(document_store: DocumentStore) -> None:
    assert document_store.count() == 0

    document_store.put(document("doc-1"), generation=1)
    document_store.put(document("doc-2"), generation=1)

    assert document_store.count() == 2


def test_iteration_yields_every_document_in_a_stable_order(
    document_store: DocumentStore,
) -> None:
    for document_id in ("doc-c", "doc-a", "doc-b"):
        document_store.put(document(document_id, f"text for {document_id}"), generation=1)

    first_pass = list(document_store.iter_documents())
    second_pass = list(document_store.iter_documents())

    assert [stored.document_id for stored in first_pass] == ["doc-a", "doc-b", "doc-c"]
    assert first_pass == second_pass


def test_iteration_over_an_empty_store_yields_nothing(document_store: DocumentStore) -> None:
    assert list(document_store.iter_documents()) == []


def test_empty_text_is_stored_verbatim(document_store: DocumentStore) -> None:
    document_store.put(document("doc-1", ""), generation=1)

    stored = document_store.get("doc-1")

    assert stored is not None
    assert stored.text == ""


def test_unicode_text_survives_a_round_trip(document_store: DocumentStore) -> None:
    document_store.put(document("doc-1", "分布式搜索 — naïve café ß"), generation=1)

    stored = document_store.get("doc-1")

    assert stored is not None
    assert stored.text == "分布式搜索 — naïve café ß"


# ----------------------------------------------------------------------
# Durability, which only the SQLite store claims
# ----------------------------------------------------------------------


def test_documents_survive_closing_and_reopening(storage_path: Path) -> None:
    store = SqliteDocumentStore.open(storage_path)
    store.put(document("doc-1", "distributed search"), generation=1)
    store.put(document("doc-2", "ranking function"), generation=1)
    store.close()

    reopened = SqliteDocumentStore.open(storage_path)
    try:
        assert reopened.count() == 2
        stored = reopened.get("doc-1")
        assert stored is not None
        assert stored.text == "distributed search"
    finally:
        reopened.close()


def test_deletions_survive_reopening(storage_path: Path) -> None:
    store = SqliteDocumentStore.open(storage_path)
    store.put(document("doc-1"), generation=1)
    store.put(document("doc-2"), generation=1)
    store.delete("doc-1", generation=2)
    store.close()

    reopened = SqliteDocumentStore.open(storage_path)
    try:
        assert reopened.get("doc-1") is None
        assert reopened.count() == 1
    finally:
        reopened.close()


def test_replacements_survive_reopening(storage_path: Path) -> None:
    store = SqliteDocumentStore.open(storage_path)
    store.put(document("doc-1", "before"), generation=1)
    store.put(document("doc-1", "after"), generation=1)
    store.close()

    reopened = SqliteDocumentStore.open(storage_path)
    try:
        stored = reopened.get("doc-1")
        assert stored is not None
        assert stored.text == "after"
        assert reopened.count() == 1
    finally:
        reopened.close()


def test_opening_creates_missing_parent_directories(tmp_path: Path) -> None:
    nested = tmp_path / "data" / "nested" / "pysearch.db"

    store = SqliteDocumentStore.open(nested)
    try:
        assert nested.exists()
    finally:
        store.close()


def test_opening_an_unusable_path_raises_initialization_error(tmp_path: Path) -> None:
    # A directory cannot be opened as a database file.
    directory = tmp_path / "not-a-database"
    directory.mkdir()

    with pytest.raises(StorageInitializationError):
        SqliteDocumentStore.open(directory)


def test_using_a_closed_store_raises_a_storage_error(storage_path: Path) -> None:
    store = SqliteDocumentStore.open(storage_path)
    store.close()

    with pytest.raises(StorageError):
        store.put(document("doc-1"), generation=1)


def test_storage_errors_do_not_leak_sql(storage_path: Path) -> None:
    store = SqliteDocumentStore.open(storage_path)
    store.close()

    with pytest.raises(StorageError) as raised:
        store.get("doc-1")

    assert "SELECT" not in str(raised.value)
