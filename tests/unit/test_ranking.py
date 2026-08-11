"""Tests for BM25 scoring.

The expected values in this module were computed independently from the
documented formula, and are written here as fixed numbers with their derivation
in a comment. They are deliberately not recomputed from the implementation,
since a test that re-runs the code under test cannot detect an error in it.
"""

import pytest

from app.search.index import InvertedIndex
from app.search.ranking import BM25Params, BM25Scorer


@pytest.fixture
def scorer() -> BM25Scorer:
    return BM25Scorer()


def build_index(documents: dict[str, list[str]]) -> InvertedIndex:
    index = InvertedIndex()
    for document_id, tokens in documents.items():
        index.add_document(document_id, tokens)
    return index


# ----------------------------------------------------------------------
# Hand-computed values
# ----------------------------------------------------------------------


def test_score_matches_an_independently_computed_value(scorer: BM25Scorer) -> None:
    # N=2, df=1, tf=1, dl=1, avgdl=1.0
    #   idf = ln(1 + (2 - 1 + 0.5) / (1 + 0.5)) = ln(2)
    #   tf part = (1 * 2.2) / (1 + 1.2 * (0.25 + 0.75 * 1.0)) = 2.2 / 2.2 = 1
    #   score   = ln(2)
    index = build_index({"doc-1": ["cat"], "doc-2": ["dog"]})

    scores = scorer.score_query(index, ["cat"])

    assert scores["doc-1"] == pytest.approx(0.6931471805599453, abs=1e-12)
    assert "doc-2" not in scores


def test_term_frequency_saturation_matches_hand_computation(scorer: BM25Scorer) -> None:
    # N=2, df=2, avgdl=2.0, so idf = ln(1 + 0.5 / 2.5) = ln(1.2)
    #   doc-a: tf=2, dl=2 -> ln(1.2) * 4.4 / 3.2 = 0.2506921405916876
    #   doc-b: tf=1, dl=2 -> ln(1.2) * 2.2 / 2.2 = 0.1823215567939546
    index = build_index({"doc-a": ["cat", "cat"], "doc-b": ["cat", "dog"]})

    scores = scorer.score_query(index, ["cat"])

    assert scores["doc-a"] == pytest.approx(0.2506921405916876, abs=1e-12)
    assert scores["doc-b"] == pytest.approx(0.1823215567939546, abs=1e-12)


def test_length_normalization_matches_hand_computation(scorer: BM25Scorer) -> None:
    # N=2, df=2, avgdl=2.5, tf=1 in both documents.
    #   short (dl=1): ln(1.2) * 2.2 / (1 + 1.2 * (0.25 + 0.75 * 0.4)) = 0.24163097888355428
    #   long  (dl=4): ln(1.2) * 2.2 / (1 + 1.2 * (0.25 + 0.75 * 1.6)) = 0.1463895711484307
    index = build_index({"short": ["cat"], "long": ["cat", "dog", "bird", "fish"]})

    scores = scorer.score_query(index, ["cat"])

    assert scores["short"] == pytest.approx(0.24163097888355428, abs=1e-12)
    assert scores["long"] == pytest.approx(0.1463895711484307, abs=1e-12)


def test_inverse_document_frequency_matches_hand_computation(scorer: BM25Scorer) -> None:
    # N=3: rare term (df=1) -> ln(1 + 2.5 / 1.5); ubiquitous term (df=3) -> ln(1 + 0.5 / 3.5)
    assert scorer.inverse_document_frequency(1, 3) == pytest.approx(0.9808292530117263, abs=1e-12)
    assert scorer.inverse_document_frequency(3, 3) == pytest.approx(0.13353139262452257, abs=1e-12)


# ----------------------------------------------------------------------
# Ranking properties
# ----------------------------------------------------------------------


def test_a_matching_document_outranks_a_non_matching_one(scorer: BM25Scorer) -> None:
    index = build_index({"hit": ["search", "engine"], "miss": ["cooking", "recipe"]})

    scores = scorer.score_query(index, ["search"])

    assert "hit" in scores
    assert "miss" not in scores


def test_more_occurrences_of_a_query_term_score_higher(scorer: BM25Scorer) -> None:
    index = build_index({"few": ["search", "x", "y"], "many": ["search", "search", "search"]})

    scores = scorer.score_query(index, ["search"])

    assert scores["many"] > scores["few"]


def test_a_shorter_document_scores_higher_at_equal_term_frequency(scorer: BM25Scorer) -> None:
    index = build_index({"short": ["search"], "long": ["search", "a", "b", "c", "d", "e"]})

    scores = scorer.score_query(index, ["search"])

    assert scores["short"] > scores["long"]


def test_a_rarer_term_contributes_more_than_a_common_one(scorer: BM25Scorer) -> None:
    # "common" appears in all three documents, "rare" in only one; both appear
    # once in doc-1, which has the same length as the others.
    index = build_index(
        {
            "doc-1": ["common", "rare"],
            "doc-2": ["common", "filler"],
            "doc-3": ["common", "filler"],
        }
    )

    rare_scores = scorer.score_query(index, ["rare"])
    common_scores = scorer.score_query(index, ["common"])

    assert rare_scores["doc-1"] > common_scores["doc-1"]


def test_idf_stays_positive_for_a_term_in_every_document(scorer: BM25Scorer) -> None:
    # The reason for the ln(1 + ...) formulation: the classic form would be
    # negative here, letting a common term push a document below one that
    # does not contain it at all.
    assert scorer.inverse_document_frequency(10, 10) > 0.0


# ----------------------------------------------------------------------
# Edge cases
# ----------------------------------------------------------------------


def test_unknown_terms_contribute_nothing(scorer: BM25Scorer) -> None:
    index = build_index({"doc-1": ["search"]})

    with_unknown = scorer.score_query(index, ["search", "nonexistent"])
    without_unknown = scorer.score_query(index, ["search"])

    assert with_unknown == without_unknown


def test_a_query_of_only_unknown_terms_matches_nothing(scorer: BM25Scorer) -> None:
    index = build_index({"doc-1": ["search"]})

    assert scorer.score_query(index, ["nonexistent"]) == {}


def test_repeated_query_terms_accumulate(scorer: BM25Scorer) -> None:
    index = build_index({"doc-1": ["search"], "doc-2": ["other"]})

    once = scorer.score_query(index, ["search"])
    twice = scorer.score_query(index, ["search", "search"])

    assert twice["doc-1"] == pytest.approx(2 * once["doc-1"], abs=1e-12)


def test_multiple_query_terms_sum_their_contributions(scorer: BM25Scorer) -> None:
    index = build_index({"doc-1": ["search", "engine"], "doc-2": ["search", "other"]})

    both = scorer.score_query(index, ["search", "engine"])
    search_only = scorer.score_query(index, ["search"])

    assert both["doc-1"] > search_only["doc-1"]
    assert both["doc-2"] == pytest.approx(search_only["doc-2"], abs=1e-12)


def test_an_empty_index_produces_no_scores(scorer: BM25Scorer) -> None:
    assert scorer.score_query(InvertedIndex(), ["search"]) == {}


def test_a_corpus_of_only_empty_documents_produces_no_scores(scorer: BM25Scorer) -> None:
    # avgdl is 0 here; the guard keeps it out of the denominator.
    index = build_index({"doc-1": [], "doc-2": []})

    assert scorer.score_query(index, ["search"]) == {}


def test_scoring_is_reproducible(scorer: BM25Scorer) -> None:
    index = build_index({"doc-1": ["a", "b"], "doc-2": ["a", "a", "c"]})

    assert scorer.score_query(index, ["a", "b", "c"]) == scorer.score_query(index, ["a", "b", "c"])


# ----------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------


def test_default_parameters_are_the_documented_values(scorer: BM25Scorer) -> None:
    assert scorer.params == BM25Params(k1=1.2, b=0.75)


def test_b_zero_disables_length_normalization() -> None:
    scorer = BM25Scorer(BM25Params(k1=1.2, b=0.0))
    index = build_index({"short": ["search"], "long": ["search", "a", "b", "c", "d"]})

    scores = scorer.score_query(index, ["search"])

    assert scores["short"] == pytest.approx(scores["long"], abs=1e-12)


def test_higher_k1_slows_term_frequency_saturation() -> None:
    index = build_index({"doc-1": ["search", "search", "search", "search"]})

    low = BM25Scorer(BM25Params(k1=0.5, b=0.75)).score_query(index, ["search"])
    high = BM25Scorer(BM25Params(k1=2.0, b=0.75)).score_query(index, ["search"])

    assert high["doc-1"] > low["doc-1"]
