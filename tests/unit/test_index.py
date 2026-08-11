"""Tests for the inverted index, its statistics, and its invariants."""

import pytest

from app.search.errors import IndexInvariantError
from app.search.index import InvertedIndex, Posting


@pytest.fixture
def index() -> InvertedIndex:
    return InvertedIndex()


def postings_as_dict(index: InvertedIndex, term: str) -> dict[str, int]:
    return {posting.document_id: posting.term_frequency for posting in index.postings(term)}


# ----------------------------------------------------------------------
# Insertion
# ----------------------------------------------------------------------


def test_empty_index_has_zeroed_statistics(index: InvertedIndex) -> None:
    assert index.document_count == 0
    assert index.unique_term_count == 0
    assert index.total_token_count == 0
    assert index.average_document_length == 0.0


def test_single_document_populates_postings(index: InvertedIndex) -> None:
    index.add_document("doc-1", ["search", "engine"])

    assert index.document_count == 1
    assert index.unique_term_count == 2
    assert postings_as_dict(index, "search") == {"doc-1": 1}
    assert index.document_length("doc-1") == 2
    assert index.average_document_length == 2.0


def test_repeated_terms_become_term_frequency(index: InvertedIndex) -> None:
    index.add_document("doc-1", ["search", "search", "search", "engine"])

    assert index.term_frequency("search", "doc-1") == 3
    assert index.document_frequency("search") == 1
    assert index.document_length("doc-1") == 4


def test_multiple_documents_share_posting_lists(index: InvertedIndex) -> None:
    index.add_document("doc-1", ["search", "engine"])
    index.add_document("doc-2", ["search", "index", "index"])

    assert postings_as_dict(index, "search") == {"doc-1": 1, "doc-2": 1}
    assert postings_as_dict(index, "index") == {"doc-2": 2}
    assert index.document_frequency("search") == 2
    assert index.document_frequency("engine") == 1
    assert index.document_count == 2
    assert index.total_token_count == 5
    assert index.average_document_length == 2.5


def test_postings_expose_explicit_posting_values(index: InvertedIndex) -> None:
    index.add_document("doc-1", ["search", "search"])

    assert list(index.postings("search")) == [Posting(document_id="doc-1", term_frequency=2)]


def test_unknown_terms_and_documents_report_empty(index: InvertedIndex) -> None:
    index.add_document("doc-1", ["search"])

    assert index.document_frequency("absent") == 0
    assert index.term_frequency("absent", "doc-1") == 0
    assert index.document_length("doc-404") == 0
    assert list(index.postings("absent")) == []


def test_empty_document_is_indexed_with_zero_length(index: InvertedIndex) -> None:
    index.add_document("doc-1", ["search"])
    index.add_document("doc-2", [])

    assert index.document_count == 2
    assert index.contains("doc-2")
    assert index.document_length("doc-2") == 0
    assert index.average_document_length == 0.5
    index.validate()


# ----------------------------------------------------------------------
# Replacement
# ----------------------------------------------------------------------


def test_replacement_removes_terms_that_disappeared(index: InvertedIndex) -> None:
    index.add_document("doc-1", ["search", "engine"])
    index.add_document("doc-1", ["search", "ranking"])

    assert index.document_frequency("engine") == 0
    assert list(index.postings("engine")) == []
    assert index.unique_term_count == 2


def test_replacement_adds_new_terms(index: InvertedIndex) -> None:
    index.add_document("doc-1", ["search"])
    index.add_document("doc-1", ["search", "ranking"])

    assert postings_as_dict(index, "ranking") == {"doc-1": 1}


def test_replacement_keeps_statistics_correct(index: InvertedIndex) -> None:
    index.add_document("doc-1", ["a", "b", "c"])
    index.add_document("doc-2", ["a"])
    index.add_document("doc-1", ["a"])

    assert index.document_count == 2
    assert index.total_token_count == 2
    assert index.average_document_length == 1.0
    assert index.term_frequency("a", "doc-1") == 1
    index.validate()


def test_replacement_updates_term_frequency(index: InvertedIndex) -> None:
    index.add_document("doc-1", ["search"])
    index.add_document("doc-1", ["search", "search", "search"])

    assert index.term_frequency("search", "doc-1") == 3
    assert index.document_length("doc-1") == 3


# ----------------------------------------------------------------------
# Deletion
# ----------------------------------------------------------------------


def test_deletion_removes_the_document(index: InvertedIndex) -> None:
    index.add_document("doc-1", ["search"])
    index.remove_document("doc-1")

    assert not index.contains("doc-1")
    assert index.document_count == 0
    assert index.document_length("doc-1") == 0


def test_deletion_drops_terms_left_with_no_postings(index: InvertedIndex) -> None:
    index.add_document("doc-1", ["search", "unique"])
    index.add_document("doc-2", ["search"])
    index.remove_document("doc-1")

    assert index.unique_term_count == 1
    assert index.document_frequency("unique") == 0
    assert postings_as_dict(index, "search") == {"doc-2": 1}


def test_deletion_updates_document_frequency(index: InvertedIndex) -> None:
    index.add_document("doc-1", ["search"])
    index.add_document("doc-2", ["search"])
    index.remove_document("doc-2")

    assert index.document_frequency("search") == 1


def test_deletion_updates_corpus_statistics(index: InvertedIndex) -> None:
    index.add_document("doc-1", ["a", "b", "c", "d"])
    index.add_document("doc-2", ["a", "b"])
    index.remove_document("doc-1")

    assert index.document_count == 1
    assert index.total_token_count == 2
    assert index.average_document_length == 2.0
    index.validate()


def test_deleting_every_document_returns_to_the_empty_state(index: InvertedIndex) -> None:
    index.add_document("doc-1", ["a", "b"])
    index.add_document("doc-2", ["b"])
    index.remove_document("doc-1")
    index.remove_document("doc-2")

    assert index.document_count == 0
    assert index.unique_term_count == 0
    assert index.total_token_count == 0
    assert index.average_document_length == 0.0
    index.validate()


def test_deleting_a_missing_document_raises(index: InvertedIndex) -> None:
    with pytest.raises(KeyError):
        index.remove_document("doc-404")


# ----------------------------------------------------------------------
# Invariants
# ----------------------------------------------------------------------


def test_validate_accepts_a_consistent_index(index: InvertedIndex) -> None:
    index.add_document("doc-1", ["search", "search", "engine"])
    index.add_document("doc-2", ["engine"])
    index.add_document("doc-3", [])

    index.validate()


def test_validate_detects_a_corrupted_total_length(index: InvertedIndex) -> None:
    index.add_document("doc-1", ["search"])
    index._total_length += 1

    with pytest.raises(IndexInvariantError, match="total length"):
        index.validate()


def test_validate_detects_an_empty_posting_list(index: InvertedIndex) -> None:
    index.add_document("doc-1", ["search"])
    index._postings["ghost"] = {}

    with pytest.raises(IndexInvariantError, match="empty posting list"):
        index.validate()


def test_validate_detects_a_posting_for_a_missing_document(index: InvertedIndex) -> None:
    index.add_document("doc-1", ["search"])
    index._postings["search"]["doc-ghost"] = 1

    with pytest.raises(IndexInvariantError, match="unknown document"):
        index.validate()


def test_validate_detects_a_document_term_mapping_mismatch(index: InvertedIndex) -> None:
    index.add_document("doc-1", ["search"])
    index._document_terms["doc-1"] = frozenset({"search", "phantom"})

    with pytest.raises(IndexInvariantError, match="no posting"):
        index.validate()


def test_validate_detects_a_length_that_disagrees_with_term_frequencies(
    index: InvertedIndex,
) -> None:
    index.add_document("doc-1", ["search"])
    index._document_lengths["doc-1"] = 5
    index._total_length = 5

    with pytest.raises(IndexInvariantError, match="term frequencies sum"):
        index.validate()
