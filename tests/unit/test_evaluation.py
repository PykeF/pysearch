"""Tests for the retrieval evaluation harness.

The metrics are checked against hand-computed values, and the labelled set is
checked for the integrity properties that make its numbers meaningful. What the
real model *scores* is a measurement, not an assertion — that lives in the
script, so a model or parameter change shows up as a different number rather
than a failing test.
"""

import pytest

from scripts.evaluate_retrieval import recall_at_k, reciprocal_rank
from scripts.evaluation_data import (
    CATEGORIES,
    CORPUS,
    DEVELOPMENT_QUERIES,
    EVALUATION_QUERIES,
)

# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------


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


# ----------------------------------------------------------------------
# Dataset integrity
# ----------------------------------------------------------------------


def test_every_labelled_query_points_at_documents_that_exist() -> None:
    """A typo in a label would quietly deflate every mode's score."""
    for labelled in (*DEVELOPMENT_QUERIES, *EVALUATION_QUERIES):
        assert labelled.relevant, f"{labelled.query!r} has no relevant documents"
        unknown = labelled.relevant - set(CORPUS)
        assert not unknown, f"{labelled.query!r} refers to unknown documents {unknown}"


def test_the_development_and_evaluation_queries_do_not_overlap() -> None:
    """The split is the point: parameters are chosen on one set, reported on the other."""
    development = {labelled.query for labelled in DEVELOPMENT_QUERIES}
    evaluation = {labelled.query for labelled in EVALUATION_QUERIES}

    assert not development & evaluation


def test_every_query_declares_a_known_category() -> None:
    for labelled in (*DEVELOPMENT_QUERIES, *EVALUATION_QUERIES):
        assert labelled.category in CATEGORIES, labelled.query


def test_the_evaluation_set_covers_every_category() -> None:
    # An aggregate over only one kind of query would say nothing about where
    # the two signals differ.
    covered = {labelled.category for labelled in EVALUATION_QUERIES}

    assert covered == set(CATEGORIES)


def test_each_category_has_enough_queries_to_report() -> None:
    for category in CATEGORIES:
        group = [q for q in EVALUATION_QUERIES if q.category == category]
        assert len(group) >= 4, f"{category} has only {len(group)} queries"


def test_the_corpus_contains_deliberate_near_duplicates() -> None:
    """The distractors are what create tension between the two signals.

    Several documents differ only by an identifier, so meaning cannot separate
    them and exact matching has to.
    """
    connection_resets = [key for key, text in CORPUS.items() if "ERR_CONN_RESET" in text]
    battery = [key for key, text in CORPUS.items() if "battery replacement procedure" in text]

    assert len(connection_resets) >= 3
    assert len(battery) >= 3


def test_the_corpus_is_large_enough_for_ranking_to_be_contested() -> None:
    assert len(CORPUS) >= 60
    assert len(EVALUATION_QUERIES) >= 18
    assert len(DEVELOPMENT_QUERIES) >= 10
