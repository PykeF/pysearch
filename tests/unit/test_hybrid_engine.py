"""Hybrid retrieval on a single node.

The engine runs both retrievals under one lock and fuses the result. These tests
use the deterministic fake embedder, so no model is loaded.
"""

import pytest

from app.hybrid.fusion import FusionConfig
from app.search.document import Document
from app.search.engine import SearchEngine
from app.search.errors import EngineNotReadyError
from app.semantic.errors import SemanticDisabledError

CAR = "car engine repair and vehicle maintenance"
SEARCH = "search ranking and retrieval relevance"
COOKING = "cooking pasta recipe in the kitchen"
SHARDING = "shard replica cluster node distributed replication"


def index(engine: SearchEngine, documents: dict[str, str]) -> None:
    for document_id, text in documents.items():
        engine.index_document(Document(document_id=document_id, text=text))


def ranked(engine: SearchEngine, query: str, limit: int = 10) -> list[str]:
    return [hit.document_id for hit in engine.hybrid_search(query, limit).results]


# ----------------------------------------------------------------------
# Availability
# ----------------------------------------------------------------------


def test_hybrid_requires_semantic_search(engine: SearchEngine) -> None:
    """No silent degradation to BM25 under the name "hybrid"."""
    index(engine, {"doc-1": CAR})

    with pytest.raises(SemanticDisabledError, match="hybrid search requires"):
        engine.hybrid_search("vehicle", limit=5)


def test_a_non_ready_engine_refuses_hybrid_search(semantic_engine: SearchEngine) -> None:
    index(semantic_engine, {"doc-1": CAR})
    semantic_engine.mark_out_of_sync("forced for the test")

    with pytest.raises(EngineNotReadyError):
        semantic_engine.hybrid_search("vehicle", limit=5)


def test_hybrid_completes_without_deadlocking(semantic_engine: SearchEngine) -> None:
    """Both retrievals share one non-reentrant lock; this would hang if reacquired."""
    index(semantic_engine, {"doc-1": CAR, "doc-2": SEARCH})

    outcome = semantic_engine.hybrid_search("vehicle repair", limit=5)

    assert outcome.total > 0


# ----------------------------------------------------------------------
# Fusion behaviour
# ----------------------------------------------------------------------


def test_both_signals_contribute(semantic_engine: SearchEngine) -> None:
    index(semantic_engine, {"doc-1": CAR, "doc-2": SEARCH, "doc-3": COOKING})

    hits = semantic_engine.hybrid_search("vehicle motor repair", limit=5).results
    top = hits[0]

    assert top.document_id == "doc-1"
    # It ranked in both lists, which is what its fusion score is made of.
    assert top.lexical_rank is not None
    assert top.semantic_rank is not None
    assert top.score == pytest.approx(
        1 / (60 + top.lexical_rank) + 1 / (60 + top.semantic_rank), abs=1e-12
    )


def test_a_paraphrase_still_ranks_when_bm25_finds_nothing(
    semantic_engine: SearchEngine,
) -> None:
    """The Phase 5 finding, now flowing through fusion."""
    index(semantic_engine, {"doc-1": CAR, "doc-2": SEARCH})

    # "automobile" appears in no document, so BM25 contributes nothing at all.
    assert semantic_engine.search("automobile", limit=10).total == 0

    hits = semantic_engine.hybrid_search("automobile", limit=5).results

    assert hits[0].document_id == "doc-1"
    assert hits[0].lexical_rank is None
    assert hits[0].semantic_rank == 1


def test_total_is_the_candidate_union(semantic_engine: SearchEngine) -> None:
    index(semantic_engine, {"doc-1": CAR, "doc-2": SEARCH, "doc-3": COOKING})

    outcome = semantic_engine.hybrid_search("vehicle", limit=1)

    # Semantic ranks the whole corpus, so all three are candidates even though
    # only one is returned.
    assert outcome.total == 3
    assert len(outcome.results) == 1


def test_results_carry_document_text(semantic_engine: SearchEngine) -> None:
    index(semantic_engine, {"doc-1": CAR})

    assert semantic_engine.hybrid_search("vehicle", limit=1).results[0].text == CAR


def test_results_are_ordered_by_fusion_score(semantic_engine: SearchEngine) -> None:
    index(semantic_engine, {"doc-1": CAR, "doc-2": SEARCH, "doc-3": COOKING, "doc-4": SHARDING})

    scores = [hit.score for hit in semantic_engine.hybrid_search("vehicle", limit=10).results]

    assert scores == sorted(scores, reverse=True)


def test_hybrid_is_deterministic(semantic_engine: SearchEngine) -> None:
    index(semantic_engine, {"doc-1": CAR, "doc-2": SEARCH, "doc-3": COOKING})

    assert semantic_engine.hybrid_search("vehicle repair", limit=5) == (
        semantic_engine.hybrid_search("vehicle repair", limit=5)
    )


def test_an_empty_query_returns_nothing(semantic_engine: SearchEngine) -> None:
    """Answered without asking the model what an empty string means."""
    index(semantic_engine, {"doc-1": CAR})

    for query in ("", "   ", "!!! ???"):
        outcome = semantic_engine.hybrid_search(query, limit=5)
        assert outcome.total == 0
        assert outcome.results == ()


def test_the_configuration_changes_the_ranking(semantic_engine: SearchEngine) -> None:
    index(semantic_engine, {"doc-1": CAR, "doc-2": SEARCH, "doc-3": COOKING})

    default = semantic_engine.hybrid_search("vehicle", limit=3)
    small_k = semantic_engine.hybrid_search("vehicle", limit=3, config=FusionConfig(rrf_k=1))

    # Same documents, different fusion scores.
    assert [hit.score for hit in default.results] != [hit.score for hit in small_k.results]


def test_candidate_depth_bounds_what_fusion_sees(semantic_engine: SearchEngine) -> None:
    index(semantic_engine, {f"doc-{n}": SEARCH for n in range(10)})

    shallow = semantic_engine.hybrid_search(
        "search", limit=1, config=FusionConfig(candidate_multiplier=1)
    )

    # depth = 1 * limit = 1 from each path, so at most two candidates fuse.
    assert shallow.total <= 2


# ----------------------------------------------------------------------
# Mutations flow through automatically
# ----------------------------------------------------------------------


def test_replacement_changes_hybrid_ranking(semantic_engine: SearchEngine) -> None:
    def hit_for(document_id: str, query: str):  # type: ignore[no-untyped-def]
        hits = semantic_engine.hybrid_search(query, limit=10).results
        return next(hit for hit in hits if hit.document_id == document_id)

    index(semantic_engine, {"doc-1": CAR, "doc-2": SEARCH})
    assert ranked(semantic_engine, "vehicle repair")[0] == "doc-1"
    before = hit_for("doc-1", "vehicle repair")
    assert before.lexical_rank is not None

    index(semantic_engine, {"doc-1": COOKING})

    after = hit_for("doc-1", "vehicle repair")
    # Both signals lost the old content: BM25 no longer matches the document at
    # all, and its embedding has moved away from the old topic.
    assert after.lexical_rank is None
    assert after.semantic_score < before.semantic_score
    assert ranked(semantic_engine, "recipe kitchen")[0] == "doc-1"


def test_deletion_removes_a_document_from_hybrid(semantic_engine: SearchEngine) -> None:
    index(semantic_engine, {"doc-1": CAR, "doc-2": SEARCH})
    semantic_engine.delete_document("doc-1")

    assert "doc-1" not in ranked(semantic_engine, "vehicle repair")


def test_hybrid_needs_no_state_of_its_own_after_a_rebuild(
    semantic_engine: SearchEngine,
) -> None:
    """Fusion is computed per query, so recovery restores it for free."""
    index(semantic_engine, {"doc-1": CAR, "doc-2": SEARCH, "doc-3": COOKING})
    before = semantic_engine.hybrid_search("vehicle repair", limit=5)

    semantic_engine.initialize()

    assert semantic_engine.hybrid_search("vehicle repair", limit=5) == before
