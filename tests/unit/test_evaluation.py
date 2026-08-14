"""Tests for the retrieval evaluation harness.

The metrics themselves are checked against hand-computed values. What the real
model scores is a measurement, not an assertion — that lives in the script, so a
model change shows up as a different number rather than a failing test.
"""

import pytest

from scripts.evaluate_retrieval import CORPUS, QUERIES, K, recall_at_k, reciprocal_rank


def test_recall_counts_relevant_documents_in_the_top_k() -> None:
    ranked = ["a", "b", "c", "d", "e", "f"]

    assert recall_at_k(ranked, frozenset({"a", "b"}), 5) == pytest.approx(1.0)
    assert recall_at_k(ranked, frozenset({"a", "f"}), 5) == pytest.approx(0.5)
    assert recall_at_k(ranked, frozenset({"f"}), 5) == pytest.approx(0.0)


def test_recall_of_an_empty_relevant_set_is_zero() -> None:
    assert recall_at_k(["a"], frozenset(), 5) == 0.0


def test_reciprocal_rank_is_one_over_the_first_relevant_position() -> None:
    ranked = ["a", "b", "c"]

    assert reciprocal_rank(ranked, frozenset({"a"})) == pytest.approx(1.0)
    assert reciprocal_rank(ranked, frozenset({"b"})) == pytest.approx(0.5)
    assert reciprocal_rank(ranked, frozenset({"c"})) == pytest.approx(1 / 3)


def test_reciprocal_rank_is_zero_when_nothing_relevant_is_found() -> None:
    assert reciprocal_rank(["a", "b"], frozenset({"z"})) == 0.0


def test_every_labelled_query_points_at_documents_that_exist() -> None:
    """A typo in a label would quietly deflate the measured scores."""
    for labelled in QUERIES:
        assert labelled.relevant, f"{labelled.query!r} has no relevant documents"
        unknown = labelled.relevant - set(CORPUS)
        assert not unknown, f"{labelled.query!r} refers to unknown documents {unknown}"


def test_the_evaluation_set_covers_both_kinds_of_query() -> None:
    # The comparison is only informative if it contains queries that share
    # wording with their answer and queries that do not.
    notes = " ".join(labelled.note for labelled in QUERIES)

    assert "paraphrase" in notes
    assert "identifier" in notes or "exact" in notes
    assert K > 0
