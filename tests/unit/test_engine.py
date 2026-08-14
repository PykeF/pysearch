"""Tests for the search engine service."""

import threading

import pytest

from app.search.document import Document
from app.search.engine import SearchEngine
from app.search.errors import DocumentNotFoundError, InvalidDocumentError


def index(engine: SearchEngine, documents: dict[str, str]) -> None:
    for document_id, text in documents.items():
        engine.index_document(Document(document_id=document_id, text=text))


def ranked_ids(engine: SearchEngine, query: str, limit: int = 10) -> list[str]:
    return [hit.document_id for hit in engine.search(query, limit).results]


# ----------------------------------------------------------------------
# Document model
# ----------------------------------------------------------------------


def test_a_blank_document_id_is_rejected() -> None:
    with pytest.raises(InvalidDocumentError):
        Document(document_id="   ", text="text")

    with pytest.raises(InvalidDocumentError):
        Document(document_id="", text="text")


def test_an_empty_document_is_valid_but_never_matches(engine: SearchEngine) -> None:
    index(engine, {"empty": "", "real": "search"})

    assert engine.stats().document_count == 2
    assert ranked_ids(engine, "search") == ["real"]
    engine.validate()


# ----------------------------------------------------------------------
# Indexing and replacement
# ----------------------------------------------------------------------


def test_indexing_reports_creation_then_replacement(engine: SearchEngine) -> None:
    assert engine.index_document(Document(document_id="doc-1", text="search")) is True
    assert engine.index_document(Document(document_id="doc-1", text="ranking")) is False
    assert engine.stats().document_count == 1


def test_replacement_makes_old_terms_unsearchable(engine: SearchEngine) -> None:
    index(engine, {"doc-1": "distributed search engine"})
    assert ranked_ids(engine, "engine") == ["doc-1"]

    index(engine, {"doc-1": "vector ranking"})

    assert ranked_ids(engine, "engine") == []
    assert ranked_ids(engine, "ranking") == ["doc-1"]
    engine.validate()


def test_replacement_updates_the_returned_text(engine: SearchEngine) -> None:
    index(engine, {"doc-1": "first text about search"})
    index(engine, {"doc-1": "second text about search"})

    results = engine.search("search", limit=10).results

    assert results[0].text == "second text about search"


# ----------------------------------------------------------------------
# Deletion
# ----------------------------------------------------------------------


def test_deletion_removes_a_document_from_results(engine: SearchEngine) -> None:
    index(engine, {"doc-1": "search", "doc-2": "search"})
    engine.delete_document("doc-1")

    assert ranked_ids(engine, "search") == ["doc-2"]
    assert engine.stats().document_count == 1
    engine.validate()


def test_deleting_an_unknown_document_raises(engine: SearchEngine) -> None:
    with pytest.raises(DocumentNotFoundError):
        engine.delete_document("doc-404")


def test_deleting_every_document_leaves_an_empty_index(engine: SearchEngine) -> None:
    index(engine, {"doc-1": "search", "doc-2": "ranking"})
    engine.delete_document("doc-1")
    engine.delete_document("doc-2")

    stats = engine.stats()

    assert stats.document_count == 0
    assert stats.unique_term_count == 0
    assert stats.average_document_length == 0.0
    assert ranked_ids(engine, "search") == []
    engine.validate()


# ----------------------------------------------------------------------
# Querying
# ----------------------------------------------------------------------


def test_query_terms_are_analysed_like_documents(engine: SearchEngine) -> None:
    index(engine, {"doc-1": "Distributed Search"})

    assert ranked_ids(engine, "SEARCH!") == ["doc-1"]


def test_an_empty_query_returns_no_results(engine: SearchEngine) -> None:
    index(engine, {"doc-1": "search"})

    outcome = engine.search("", limit=10)

    assert outcome.total == 0
    assert outcome.results == ()


def test_a_punctuation_only_query_returns_no_results(engine: SearchEngine) -> None:
    index(engine, {"doc-1": "search"})

    assert engine.search("!!! ???", limit=10).total == 0


def test_a_query_matching_nothing_returns_no_results(engine: SearchEngine) -> None:
    index(engine, {"doc-1": "search"})

    assert engine.search("astrophysics", limit=10).total == 0


def test_searching_an_empty_index_returns_no_results(engine: SearchEngine) -> None:
    assert engine.search("search", limit=10).total == 0


def test_total_counts_all_matches_while_results_respect_the_limit(engine: SearchEngine) -> None:
    index(engine, {f"doc-{n}": "search" for n in range(5)})

    outcome = engine.search("search", limit=2)

    assert outcome.total == 5
    assert len(outcome.results) == 2


def test_results_are_ordered_by_descending_score(engine: SearchEngine) -> None:
    index(
        engine,
        {
            "weak": "search followed by plenty of unrelated filler words here",
            "strong": "search search search",
        },
    )

    scores = [hit.score for hit in engine.search("search", limit=10).results]

    assert scores == sorted(scores, reverse=True)
    assert ranked_ids(engine, "search")[0] == "strong"


def test_ties_are_broken_by_ascending_document_id(engine: SearchEngine) -> None:
    # Identical text means identical scores, so ordering is decided entirely by
    # the tie-break rule rather than by insertion or dictionary order.
    index(engine, {"doc-c": "search", "doc-a": "search", "doc-b": "search"})

    outcome = engine.search("search", limit=10)
    scores = {hit.score for hit in outcome.results}

    assert len(scores) == 1
    assert [hit.document_id for hit in outcome.results] == ["doc-a", "doc-b", "doc-c"]


def test_ordering_is_stable_across_repeated_searches(engine: SearchEngine) -> None:
    index(engine, {"doc-c": "search", "doc-a": "search", "doc-b": "search engine"})

    assert ranked_ids(engine, "search engine") == ranked_ids(engine, "search engine")


def test_results_carry_the_document_text(engine: SearchEngine) -> None:
    index(engine, {"doc-1": "distributed search"})

    assert engine.search("search", limit=10).results[0].text == "distributed search"


# ----------------------------------------------------------------------
# Concurrency
# ----------------------------------------------------------------------


def test_concurrent_writes_leave_the_index_consistent(engine: SearchEngine) -> None:
    """Run bounded concurrent traffic and assert the invariants afterwards.

    The assertions are on the final state, not on the scheduling, so this is
    deterministic: whatever order the threads happen to run in, the totals and
    the structural invariants must hold.
    """
    documents_per_thread = 40
    thread_count = 4

    def writer(thread_id: int) -> None:
        for n in range(documents_per_thread):
            document_id = f"doc-{thread_id}-{n}"
            engine.index_document(Document(document_id=document_id, text=f"term{n} shared"))

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(thread_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    engine.validate()

    stats = engine.stats()
    expected_documents = documents_per_thread * thread_count
    assert stats.document_count == expected_documents
    # Every document analyses to exactly two tokens.
    assert stats.total_token_count == expected_documents * 2
    assert stats.average_document_length == 2.0
    assert engine.search("shared", limit=1).total == expected_documents


def test_concurrent_writes_deletes_and_searches_leave_the_index_consistent(
    engine: SearchEngine,
) -> None:
    document_count = 60
    index(engine, {f"doc-{n}": f"term{n} shared" for n in range(document_count)})

    def deleter() -> None:
        for n in range(0, document_count, 2):
            engine.delete_document(f"doc-{n}")

    def searcher() -> None:
        for _ in range(40):
            engine.search("shared", limit=5)

    def writer() -> None:
        for n in range(document_count, document_count + 20):
            engine.index_document(Document(document_id=f"doc-{n}", text=f"term{n} shared"))

    threads = [
        threading.Thread(target=deleter),
        threading.Thread(target=searcher),
        threading.Thread(target=writer),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    engine.validate()

    stats = engine.stats()
    expected_documents = document_count - (document_count // 2) + 20
    assert stats.document_count == expected_documents
    assert stats.total_token_count == expected_documents * 2
    assert engine.search("shared", limit=1).total == expected_documents
