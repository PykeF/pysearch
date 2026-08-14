"""Tests for the exact vector index.

Vectors here are hand-built unit vectors along known axes, so every similarity
is a number that can be reasoned about rather than trusted. No model is loaded.
"""

import numpy as np
import pytest

from app.semantic.vector_index import ExactVectorIndex, ScoredDocument, VectorIndexError

DIMENSION = 3


def unit(*components: float) -> np.ndarray:
    """A unit vector along the given components."""
    vector = np.array(components, dtype=np.float32)
    return vector / np.linalg.norm(vector)


X = unit(1, 0, 0)
Y = unit(0, 1, 0)
Z = unit(0, 0, 1)
XY = unit(1, 1, 0)


@pytest.fixture
def index() -> ExactVectorIndex:
    return ExactVectorIndex(DIMENSION)


# ----------------------------------------------------------------------
# Construction and basics
# ----------------------------------------------------------------------


def test_a_new_index_is_empty(index: ExactVectorIndex) -> None:
    assert index.count == 0
    assert index.dimension == DIMENSION
    assert index.document_ids() == frozenset()
    assert index.search(X, limit=5) == []


@pytest.mark.parametrize("dimension", [0, -1])
def test_an_invalid_dimension_is_rejected(dimension: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        ExactVectorIndex(dimension)


def test_add_then_search_finds_the_document(index: ExactVectorIndex) -> None:
    index.add("doc-1", X)

    assert index.count == 1
    assert index.contains("doc-1")
    assert index.search(X, limit=5) == [ScoredDocument(document_id="doc-1", score=1.0)]


def test_similarity_is_the_cosine_of_the_angle(index: ExactVectorIndex) -> None:
    index.add("same", X)
    index.add("orthogonal", Y)
    index.add("halfway", XY)

    hits = {hit.document_id: hit.score for hit in index.search(X, limit=5)}

    assert hits["same"] == pytest.approx(1.0)
    assert hits["orthogonal"] == pytest.approx(0.0)
    # 45 degrees between X and XY.
    assert hits["halfway"] == pytest.approx(np.sqrt(0.5), abs=1e-6)


def test_opposite_vectors_score_negative(index: ExactVectorIndex) -> None:
    index.add("opposite", unit(-1, 0, 0))

    assert index.search(X, limit=1)[0].score == pytest.approx(-1.0)


# ----------------------------------------------------------------------
# Ordering
# ----------------------------------------------------------------------


def test_results_are_ordered_by_descending_similarity(index: ExactVectorIndex) -> None:
    index.add("far", Y)
    index.add("near", X)
    index.add("middling", XY)

    assert [hit.document_id for hit in index.search(X, limit=5)] == ["near", "middling", "far"]


def test_ties_break_on_ascending_document_id(index: ExactVectorIndex) -> None:
    # Identical vectors mean identical scores, so ordering is decided entirely
    # by the tie-break rule rather than by insertion order or row layout.
    for document_id in ("doc-c", "doc-a", "doc-b"):
        index.add(document_id, X)

    assert [hit.document_id for hit in index.search(X, limit=5)] == ["doc-a", "doc-b", "doc-c"]


def test_the_limit_truncates_the_ranking(index: ExactVectorIndex) -> None:
    for n in range(5):
        index.add(f"doc-{n}", X)

    assert len(index.search(X, limit=2)) == 2


def test_a_limit_below_one_is_rejected(index: ExactVectorIndex) -> None:
    index.add("doc-1", X)

    with pytest.raises(ValueError, match="at least 1"):
        index.search(X, limit=0)


def test_repeated_searches_are_identical(index: ExactVectorIndex) -> None:
    for n in range(6):
        index.add(f"doc-{n}", unit(1, n, 0))

    assert index.search(X, limit=4) == index.search(X, limit=4)


# ----------------------------------------------------------------------
# Replacement and deletion
# ----------------------------------------------------------------------


def test_adding_the_same_id_replaces_its_vector(index: ExactVectorIndex) -> None:
    index.add("doc-1", X)
    index.add("doc-1", Z)

    assert index.count == 1
    # The old direction is gone, not merely outranked.
    assert index.search(X, limit=1)[0].score == pytest.approx(0.0)
    assert index.search(Z, limit=1)[0].score == pytest.approx(1.0)


def test_remove_reports_whether_a_vector_existed(index: ExactVectorIndex) -> None:
    index.add("doc-1", X)

    assert index.remove("doc-1") is True
    assert index.remove("doc-1") is False


def test_a_removed_document_never_appears_again(index: ExactVectorIndex) -> None:
    index.add("doc-1", X)
    index.add("doc-2", X)
    index.remove("doc-1")

    assert [hit.document_id for hit in index.search(X, limit=5)] == ["doc-2"]
    assert not index.contains("doc-1")
    assert index.count == 1


def test_removing_from_the_middle_keeps_the_rest_searchable(index: ExactVectorIndex) -> None:
    # Removal swaps the last row into the hole, so this exercises that path.
    for n in range(5):
        index.add(f"doc-{n}", unit(1, n + 1, 0))
    index.remove("doc-2")

    found = {hit.document_id for hit in index.search(X, limit=10)}

    assert found == {"doc-0", "doc-1", "doc-3", "doc-4"}
    index.validate()


def test_removing_everything_returns_to_the_empty_state(index: ExactVectorIndex) -> None:
    index.add("doc-1", X)
    index.add("doc-2", Y)
    index.remove("doc-1")
    index.remove("doc-2")

    assert index.count == 0
    assert index.search(X, limit=5) == []
    index.validate()


def test_clear_drops_every_vector(index: ExactVectorIndex) -> None:
    index.add("doc-1", X)
    index.clear()

    assert index.count == 0
    assert index.dimension == DIMENSION


# ----------------------------------------------------------------------
# Growth
# ----------------------------------------------------------------------


def test_the_index_grows_beyond_its_initial_capacity(index: ExactVectorIndex) -> None:
    for n in range(200):
        index.add(f"doc-{n:03d}", unit(1, n + 1, 0))

    assert index.count == 200
    assert index.search(X, limit=1)[0].document_id == "doc-000"
    index.validate()


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------


def test_a_vector_of_the_wrong_width_is_rejected(index: ExactVectorIndex) -> None:
    with pytest.raises(VectorIndexError, match="dimension"):
        index.add("doc-1", np.array([1.0, 0.0], dtype=np.float32))


def test_a_two_dimensional_array_is_rejected(index: ExactVectorIndex) -> None:
    with pytest.raises(VectorIndexError, match="1-D"):
        index.add("doc-1", np.zeros((2, DIMENSION), dtype=np.float32))


def test_non_finite_values_are_rejected(index: ExactVectorIndex) -> None:
    with pytest.raises(VectorIndexError, match="non-finite"):
        index.add("doc-1", np.array([np.nan, 0.0, 0.0], dtype=np.float32))


def test_a_query_of_the_wrong_width_is_rejected(index: ExactVectorIndex) -> None:
    index.add("doc-1", X)

    with pytest.raises(VectorIndexError, match="dimension"):
        index.search(np.array([1.0, 0.0], dtype=np.float32), limit=1)


def test_validate_accepts_a_consistent_index(index: ExactVectorIndex) -> None:
    index.add("doc-1", X)
    index.add("doc-2", Y)

    index.validate()


def test_validate_detects_an_unnormalized_vector(index: ExactVectorIndex) -> None:
    index.add("doc-1", X)
    # Reach past the entry point that would have rejected this, to prove the
    # invariant check is a real second line of defence.
    index._matrix[0] = np.array([3.0, 0.0, 0.0], dtype=np.float32)

    with pytest.raises(VectorIndexError, match="expected 1"):
        index.validate()


def test_validate_detects_a_position_map_that_disagrees(index: ExactVectorIndex) -> None:
    index.add("doc-1", X)
    index._positions["doc-1"] = 7

    with pytest.raises(VectorIndexError, match="indexed elsewhere"):
        index.validate()


def test_a_zero_vector_is_allowed(index: ExactVectorIndex) -> None:
    # Text that embeds to nothing has no direction; it simply never resembles a
    # query, which is more honest than rejecting the document.
    index.add("empty", np.zeros(DIMENSION, dtype=np.float32))

    index.validate()
    assert index.search(X, limit=1)[0].score == pytest.approx(0.0)
