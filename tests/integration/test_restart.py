"""Restart and recovery tests.

These are the tests that justify Phase 2: state written before a restart must
still be there afterwards, and the derived index rebuilt from storage must be
indistinguishable from one built by indexing the same documents normally.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.search.document import Document
from app.search.engine import SearchEngine, SearchResults
from app.storage.sqlite_store import SqliteDocumentStore

CORPUS = {
    "doc-1": "Distributed systems make search scalable across many machines.",
    "doc-2": "Search engines rank documents by relevance to a query.",
    "doc-3": "BM25 is the ranking function used by most lexical search engines.",
    "doc-4": "Cooking pasta well requires salted boiling water.",
}

QUERIES = ("search", "search engines", "ranking function", "pasta", "distributed")


def start_engine(storage_path: Path) -> SearchEngine:
    """Open the database and bring an engine up against it, as startup does."""
    engine = SearchEngine(SqliteDocumentStore.open(storage_path))
    engine.initialize()
    return engine


def index_corpus(engine: SearchEngine, corpus: dict[str, str] | None = None) -> None:
    for document_id, text in (corpus or CORPUS).items():
        engine.index_document(Document(document_id=document_id, text=text))


def all_results(engine: SearchEngine) -> dict[str, SearchResults]:
    return {query: engine.search(query, limit=10) for query in QUERIES}


# ----------------------------------------------------------------------
# Recovery through the engine
# ----------------------------------------------------------------------


def test_documents_and_results_survive_a_restart(storage_path: Path) -> None:
    engine = start_engine(storage_path)
    index_corpus(engine)
    before_stats = engine.stats()
    before_results = all_results(engine)
    engine.close()

    restarted = start_engine(storage_path)
    try:
        assert restarted.stats() == before_stats
        assert all_results(restarted) == before_results
        restarted.validate()
    finally:
        restarted.close()


def test_an_update_survives_a_restart(storage_path: Path) -> None:
    engine = start_engine(storage_path)
    index_corpus(engine)
    engine.index_document(
        Document(document_id="doc-4", text="Sharding splits an index across nodes.")
    )
    engine.close()

    restarted = start_engine(storage_path)
    try:
        assert restarted.search("pasta", limit=10).total == 0
        assert restarted.search("sharding", limit=10).results[0].document_id == "doc-4"
        assert restarted.stats().document_count == 4
        restarted.validate()
    finally:
        restarted.close()


def test_a_deletion_survives_a_restart(storage_path: Path) -> None:
    engine = start_engine(storage_path)
    index_corpus(engine)
    engine.delete_document("doc-1")
    engine.close()

    restarted = start_engine(storage_path)
    try:
        assert restarted.stats().document_count == 3
        assert {hit.document_id for hit in restarted.search("search", limit=10).results} == {
            "doc-2",
            "doc-3",
        }
        assert restarted.search("distributed", limit=10).total == 0
        restarted.validate()
    finally:
        restarted.close()


def test_recovery_from_an_empty_database(storage_path: Path) -> None:
    engine = start_engine(storage_path)
    try:
        stats = engine.stats()
        assert stats.document_count == 0
        assert stats.unique_term_count == 0
        assert stats.average_document_length == 0.0
        assert engine.search("search", limit=10).total == 0
        engine.validate()
    finally:
        engine.close()


def test_repeated_restarts_are_stable(storage_path: Path) -> None:
    engine = start_engine(storage_path)
    index_corpus(engine)
    expected = all_results(engine)
    engine.close()

    for _ in range(3):
        restarted = start_engine(storage_path)
        try:
            assert all_results(restarted) == expected
            restarted.validate()
        finally:
            restarted.close()


# ----------------------------------------------------------------------
# Rebuild equivalence: recovered state must equal freshly indexed state
# ----------------------------------------------------------------------


@pytest.fixture
def rebuilt_and_fresh(storage_path: Path, tmp_path: Path) -> tuple[SearchEngine, SearchEngine]:
    """One engine recovered from disk, one built by indexing the same corpus."""
    seed = start_engine(storage_path)
    index_corpus(seed)
    seed.close()
    rebuilt = start_engine(storage_path)

    fresh = start_engine(tmp_path / "fresh.db")
    index_corpus(fresh)

    return rebuilt, fresh


def test_rebuilt_statistics_match_fresh_indexing(
    rebuilt_and_fresh: tuple[SearchEngine, SearchEngine],
) -> None:
    rebuilt, fresh = rebuilt_and_fresh
    try:
        assert rebuilt.stats() == fresh.stats()
    finally:
        rebuilt.close()
        fresh.close()


def test_rebuilt_posting_lists_match_fresh_indexing(
    rebuilt_and_fresh: tuple[SearchEngine, SearchEngine],
) -> None:
    rebuilt, fresh = rebuilt_and_fresh
    try:
        # Reaching into the index is deliberate here: the specification asks for
        # posting lists to match logically, not merely for results to agree.
        rebuilt_index = rebuilt._index
        fresh_index = fresh._index

        terms = {term for text in CORPUS.values() for term in text.lower().split()}
        for term in sorted(terms):
            cleaned = term.strip(".,")
            assert dict(rebuilt_index.posting_map(cleaned)) == dict(
                fresh_index.posting_map(cleaned)
            )
            assert rebuilt_index.document_frequency(cleaned) == fresh_index.document_frequency(
                cleaned
            )
    finally:
        rebuilt.close()
        fresh.close()


def test_rebuilt_search_results_match_fresh_indexing(
    rebuilt_and_fresh: tuple[SearchEngine, SearchEngine],
) -> None:
    rebuilt, fresh = rebuilt_and_fresh
    try:
        # Scores, ordering and totals, not just the set of matching ids.
        assert all_results(rebuilt) == all_results(fresh)
    finally:
        rebuilt.close()
        fresh.close()


# ----------------------------------------------------------------------
# Recovery through the HTTP application
# ----------------------------------------------------------------------


def test_a_restarted_application_serves_the_same_results(settings: Settings) -> None:
    with TestClient(create_app(settings)) as client:
        for document_id, text in CORPUS.items():
            client.put(f"/documents/{document_id}", json={"text": text})
        before = client.get("/search", params={"q": "search engines"}).json()
        before_stats = client.get("/index/stats").json()

    # A second application over the same database is a restart.
    with TestClient(create_app(settings)) as restarted:
        assert restarted.get("/ready").status_code == 200
        assert restarted.get("/search", params={"q": "search engines"}).json() == before
        assert restarted.get("/index/stats").json() == before_stats


def test_an_update_through_the_api_survives_a_restart(settings: Settings) -> None:
    with TestClient(create_app(settings)) as client:
        client.put("/documents/doc-1", json={"text": "Cooking pasta requires water."})
        client.put("/documents/doc-1", json={"text": "Sharding splits an index."})

    with TestClient(create_app(settings)) as restarted:
        assert restarted.get("/search", params={"q": "pasta"}).json()["total"] == 0
        assert restarted.get("/search", params={"q": "sharding"}).json()["total"] == 1
        assert restarted.get("/index/stats").json()["document_count"] == 1


def test_a_deletion_through_the_api_survives_a_restart(settings: Settings) -> None:
    with TestClient(create_app(settings)) as client:
        client.put("/documents/doc-1", json={"text": "distributed search"})
        client.put("/documents/doc-2", json={"text": "distributed systems"})
        assert client.delete("/documents/doc-1").status_code == 204

    with TestClient(create_app(settings)) as restarted:
        assert restarted.get("/index/stats").json()["document_count"] == 1
        results = restarted.get("/search", params={"q": "distributed"}).json()
        assert [hit["document_id"] for hit in results["results"]] == ["doc-2"]


def test_the_database_file_is_created_where_configured(settings: Settings) -> None:
    assert not settings.storage_path.exists()

    with TestClient(create_app(settings)) as client:
        client.put("/documents/doc-1", json={"text": "distributed search"})

    assert settings.storage_path.exists()
