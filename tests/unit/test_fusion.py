"""Tests for Reciprocal Rank Fusion.

Fusion is a pure function over two ranked lists, so these tests need no engine,
no cluster and no model. Expected values are computed by hand from the formula
and written as literals — a test that recomputes the implementation cannot
detect an error in it.
"""

import pytest

from app.hybrid.fusion import FusionConfig, reciprocal_rank_fusion
from app.search.results import SearchResult, SearchResults


def ranked(*entries: tuple[str, float], total: int | None = None) -> SearchResults:
    """A ranked list, in the order given."""
    results = tuple(
        SearchResult(document_id=document_id, score=score, text=f"text of {document_id}")
        for document_id, score in entries
    )
    return SearchResults(total=total if total is not None else len(results), results=results)


EMPTY = SearchResults(total=0, results=())
K10 = FusionConfig(rrf_k=10)


# ----------------------------------------------------------------------
# The formula
# ----------------------------------------------------------------------


def test_a_document_in_both_lists_sums_its_contributions() -> None:
    # k=10, lexical rank 1 and semantic rank 1: 1/11 + 1/11
    fused = reciprocal_rank_fusion(ranked(("doc-a", 5.0)), ranked(("doc-a", 0.9)), 10, K10)

    assert len(fused.results) == 1
    assert fused.results[0].score == pytest.approx(0.18181818181818182, abs=1e-12)


def test_a_document_in_one_list_contributes_once() -> None:
    # 1/11 only — there is deliberately no penalty term for the missing list.
    fused = reciprocal_rank_fusion(ranked(("doc-a", 5.0)), EMPTY, 10, K10)

    assert fused.results[0].score == pytest.approx(0.09090909090909091, abs=1e-12)
    assert fused.results[0].lexical_rank == 1
    assert fused.results[0].semantic_rank is None


def test_ranks_are_one_based() -> None:
    # Second place must contribute 1/(k+2), not 1/(k+1).
    fused = reciprocal_rank_fusion(ranked(("doc-a", 9.0), ("doc-b", 1.0)), EMPTY, 10, K10)

    by_id = {hit.document_id: hit for hit in fused.results}
    assert by_id["doc-a"].lexical_rank == 1
    assert by_id["doc-b"].lexical_rank == 2
    assert by_id["doc-b"].score == pytest.approx(1 / 12, abs=1e-12)


def test_the_default_k_is_used_when_no_config_is_given() -> None:
    fused = reciprocal_rank_fusion(ranked(("doc-a", 5.0)), EMPTY, 10)

    assert fused.results[0].score == pytest.approx(1 / (FusionConfig().rrf_k + 1), abs=1e-12)


def test_k_trades_top_placement_against_appearing_in_both_lists() -> None:
    """This is what the parameter actually controls.

    "top-of-one" is first in the lexical list and absent from the semantic one,
    scoring ``1/(k+1)``. "middle-of-both" is fifth in each, scoring ``2/(k+5)``.
    Small k rewards the strong single placement; large k flattens the ranks
    until two mediocre contributions outweigh one excellent one.
    """
    lexical = ranked(
        ("top-of-one", 9.0),
        ("filler-a", 8.0),
        ("filler-b", 7.0),
        ("filler-c", 6.0),
        ("middle-of-both", 5.0),
    )
    semantic = ranked(
        ("filler-d", 0.9),
        ("filler-e", 0.8),
        ("filler-f", 0.7),
        ("filler-g", 0.6),
        ("middle-of-both", 0.5),
    )

    def order(fused: object) -> list[str]:
        return [hit.document_id for hit in fused.results]  # type: ignore[attr-defined]

    small = order(reciprocal_rank_fusion(lexical, semantic, 10, FusionConfig(rrf_k=1)))
    large = order(reciprocal_rank_fusion(lexical, semantic, 10, FusionConfig(rrf_k=1000)))

    # The claim is about these two relative to each other; the fillers only
    # exist to push "middle-of-both" down to rank five in both lists.
    # k=1:    1/2 = 0.500 beats 2/6 = 0.333
    assert small.index("top-of-one") < small.index("middle-of-both")
    # k=1000: 2/1005 = 0.00199 beats 1/1001 = 0.00100
    assert large.index("middle-of-both") < large.index("top-of-one")


# ----------------------------------------------------------------------
# Ordering and deduplication
# ----------------------------------------------------------------------


def test_results_are_ordered_by_descending_fusion_score() -> None:
    lexical = ranked(("doc-a", 9.0), ("doc-b", 8.0), ("doc-c", 7.0))
    semantic = ranked(("doc-c", 0.9), ("doc-b", 0.8))

    fused = reciprocal_rank_fusion(lexical, semantic, 10, K10)
    scores = [hit.score for hit in fused.results]

    assert scores == sorted(scores, reverse=True)


def test_a_document_in_both_lists_appears_once() -> None:
    lexical = ranked(("doc-a", 9.0), ("doc-b", 8.0))
    semantic = ranked(("doc-a", 0.9), ("doc-b", 0.8))

    fused = reciprocal_rank_fusion(lexical, semantic, 10, K10)

    assert [hit.document_id for hit in fused.results] == ["doc-a", "doc-b"]
    assert fused.total == 2


def test_ties_break_on_ascending_document_id() -> None:
    # Identical positions in both lists give identical fusion scores.
    lexical = ranked(("doc-c", 5.0))
    semantic = ranked(("doc-a", 0.5))

    fused = reciprocal_rank_fusion(lexical, semantic, 10, K10)

    assert len({hit.score for hit in fused.results}) == 1
    assert [hit.document_id for hit in fused.results] == ["doc-a", "doc-c"]


def test_the_underlying_scores_are_not_a_hidden_tie_break() -> None:
    """A much larger BM25 score must not jump a tie."""
    lexical = ranked(("doc-z", 999.0))
    semantic = ranked(("doc-a", 0.01))

    fused = reciprocal_rank_fusion(lexical, semantic, 10, K10)

    assert [hit.document_id for hit in fused.results] == ["doc-a", "doc-z"]


def test_ordering_is_independent_of_input_order() -> None:
    lexical = ranked(("doc-b", 5.0), ("doc-a", 4.0))
    semantic = ranked(("doc-a", 0.9), ("doc-b", 0.8))

    first = reciprocal_rank_fusion(lexical, semantic, 10, K10)
    second = reciprocal_rank_fusion(lexical, semantic, 10, K10)

    assert first == second


# ----------------------------------------------------------------------
# Provenance
# ----------------------------------------------------------------------


def test_both_ranks_and_scores_are_carried_through() -> None:
    lexical = ranked(("doc-a", 3.72), ("doc-b", 1.5))
    semantic = ranked(("doc-b", 0.81), ("doc-a", 0.44))

    by_id = {
        hit.document_id: hit for hit in reciprocal_rank_fusion(lexical, semantic, 10, K10).results
    }

    assert (by_id["doc-a"].lexical_rank, by_id["doc-a"].semantic_rank) == (1, 2)
    assert (by_id["doc-b"].lexical_rank, by_id["doc-b"].semantic_rank) == (2, 1)
    assert by_id["doc-a"].lexical_score == pytest.approx(3.72)
    assert by_id["doc-b"].semantic_score == pytest.approx(0.81)


def test_a_missing_list_leaves_its_rank_and_score_unset() -> None:
    fused = reciprocal_rank_fusion(EMPTY, ranked(("doc-a", 0.5)), 10, K10)

    assert fused.results[0].lexical_rank is None
    assert fused.results[0].lexical_score is None
    assert fused.results[0].semantic_rank == 1


def test_result_text_comes_from_the_retrieved_candidates() -> None:
    fused = reciprocal_rank_fusion(ranked(("doc-a", 1.0)), EMPTY, 10, K10)

    assert fused.results[0].text == "text of doc-a"


# ----------------------------------------------------------------------
# Totals and truncation
# ----------------------------------------------------------------------


def test_total_is_the_candidate_union_size() -> None:
    lexical = ranked(("doc-a", 3.0), ("doc-b", 2.0), total=97)
    semantic = ranked(("doc-b", 0.9), ("doc-c", 0.8), total=500)

    fused = reciprocal_rank_fusion(lexical, semantic, 10, K10)

    # Three distinct documents entered fusion. Neither input total is reused:
    # they mean different things again.
    assert fused.total == 3


def test_the_limit_truncates_the_ranking_but_not_the_total() -> None:
    lexical = ranked(*[(f"doc-{n}", 10.0 - n) for n in range(6)])

    fused = reciprocal_rank_fusion(lexical, EMPTY, 2, K10)

    assert len(fused.results) == 2
    assert fused.total == 6


def test_fusion_of_two_empty_lists_is_empty() -> None:
    fused = reciprocal_rank_fusion(EMPTY, EMPTY, 10, K10)

    assert fused.total == 0
    assert fused.results == ()


def test_an_empty_lexical_list_is_not_an_error() -> None:
    """BM25 matching nothing is the ordinary outcome for a paraphrase."""
    fused = reciprocal_rank_fusion(EMPTY, ranked(("doc-a", 0.9), ("doc-b", 0.8)), 10, K10)

    assert [hit.document_id for hit in fused.results] == ["doc-a", "doc-b"]
    assert all(hit.lexical_rank is None for hit in fused.results)


def test_candidate_truncation_can_change_the_outcome() -> None:
    """The limitation, asserted rather than only documented.

    A document just outside both candidate lists is absent from the fused
    ranking even though its combined score would have placed it — so hybrid
    results are the exact RRF of the *retrieved* lists, not of the corpus.
    """
    full_lexical = ranked(("doc-x", 5.0), ("doc-a", 4.0), ("doc-b", 3.0))
    full_semantic = ranked(("doc-y", 0.9), ("doc-a", 0.8), ("doc-b", 0.7))

    complete = reciprocal_rank_fusion(full_lexical, full_semantic, 10, K10)
    # The same query with each path truncated to its top result.
    truncated = reciprocal_rank_fusion(ranked(("doc-x", 5.0)), ranked(("doc-y", 0.9)), 10, K10)

    assert complete.results[0].document_id == "doc-a"
    assert "doc-a" not in {hit.document_id for hit in truncated.results}


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------


def test_candidate_depth_scales_with_the_limit_up_to_a_ceiling() -> None:
    config = FusionConfig(candidate_multiplier=5, max_candidate_depth=100)

    assert config.candidate_depth(10) == 50
    assert config.candidate_depth(1) == 5
    assert config.candidate_depth(50) == 100


@pytest.mark.parametrize(
    ("field", "value"),
    [("rrf_k", 0), ("candidate_multiplier", 0)],
)
def test_invalid_configuration_is_rejected(field: str, value: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        FusionConfig(**{field: value})


def test_a_limit_below_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        reciprocal_rank_fusion(EMPTY, EMPTY, 0, K10)
