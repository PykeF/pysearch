"""Semantic state inside the engine.

The whole point of these tests is that semantic state is *derived*: it follows
the authoritative documents through every mutation, is rebuilt from them at
startup, and never survives a document it belonged to.
"""

import numpy as np
import pytest

from app.search.document import Document
from app.search.engine import NodeState, ReplicationOutcome, SearchEngine
from app.search.errors import EngineNotReadyError
from app.semantic.errors import EmbeddingError, SemanticDisabledError
from app.storage.sqlite_store import SqliteDocumentStore
from tests.conftest import FakeEmbedder, InMemoryDocumentStore

CAR = "car engine repair and vehicle maintenance"
SEARCH = "search ranking and retrieval relevance"
COOKING = "cooking pasta recipe in the kitchen"


def ranked(engine: SearchEngine, query: str, limit: int = 10) -> list[str]:
    return [
        hit.document_id for hit in engine.semantic_search(engine.embed_query(query), limit).results
    ]


def index(engine: SearchEngine, documents: dict[str, str]) -> None:
    for document_id, text in documents.items():
        engine.index_document(Document(document_id=document_id, text=text))


# ----------------------------------------------------------------------
# Enablement
# ----------------------------------------------------------------------


def test_semantic_is_off_without_an_embedder(engine: SearchEngine) -> None:
    assert engine.semantic_enabled is False
    assert engine.semantic_identity is None
    assert engine.vector_count is None

    with pytest.raises(SemanticDisabledError):
        engine.embed_query("anything")


def test_a_lexical_only_engine_still_indexes_and_searches(engine: SearchEngine) -> None:
    # Semantic being absent must cost the lexical path nothing.
    index(engine, {"doc-1": CAR})

    assert engine.search("engine", limit=10).total == 1
    engine.validate()


def test_semantic_reports_its_identity(semantic_engine: SearchEngine) -> None:
    identity = semantic_engine.semantic_identity

    assert semantic_engine.semantic_enabled is True
    assert identity is not None
    assert identity.normalization == "l2"
    assert "fake-topics@v1" in identity.fingerprint


# ----------------------------------------------------------------------
# Indexing
# ----------------------------------------------------------------------


def test_indexing_a_document_gives_it_a_vector(semantic_engine: SearchEngine) -> None:
    index(semantic_engine, {"doc-1": CAR})

    assert semantic_engine.vector_count == 1
    assert ranked(semantic_engine, "automobile repair") == ["doc-1"]
    semantic_engine.validate()


def test_semantic_search_ranks_by_topic(semantic_engine: SearchEngine) -> None:
    index(semantic_engine, {"doc-car": CAR, "doc-search": SEARCH, "doc-cook": COOKING})

    assert ranked(semantic_engine, "vehicle motor repair")[0] == "doc-car"
    assert ranked(semantic_engine, "query relevance ranking")[0] == "doc-search"
    assert ranked(semantic_engine, "recipe food")[0] == "doc-cook"


def test_the_total_counts_documents_searched(semantic_engine: SearchEngine) -> None:
    index(semantic_engine, {"doc-1": CAR, "doc-2": SEARCH, "doc-3": COOKING})

    outcome = semantic_engine.semantic_search(semantic_engine.embed_query("car"), limit=2)

    # Every document has a similarity, so "total" is the corpus, not a match count.
    assert outcome.total == 3
    assert len(outcome.results) == 2


def test_searching_an_empty_corpus_returns_nothing(semantic_engine: SearchEngine) -> None:
    outcome = semantic_engine.semantic_search(semantic_engine.embed_query("car"), limit=10)

    assert outcome.total == 0
    assert outcome.results == ()


def test_results_carry_the_document_text(semantic_engine: SearchEngine) -> None:
    index(semantic_engine, {"doc-1": CAR})

    hit = semantic_engine.semantic_search(semantic_engine.embed_query("car"), limit=1).results[0]

    assert hit.text == CAR


# ----------------------------------------------------------------------
# Replacement and deletion
# ----------------------------------------------------------------------


def test_replacement_replaces_the_vector(semantic_engine: SearchEngine) -> None:
    """A superseded document must stop being findable under its old meaning."""

    def score_of(document_id: str, query: str) -> float:
        hits = semantic_engine.semantic_search(semantic_engine.embed_query(query), limit=10)
        return next(hit.score for hit in hits.results if hit.document_id == document_id)

    index(semantic_engine, {"doc-1": CAR, "doc-2": SEARCH})
    assert ranked(semantic_engine, "vehicle repair")[0] == "doc-1"
    before = score_of("doc-1", "vehicle repair")

    index(semantic_engine, {"doc-1": COOKING})

    # The old vector is gone rather than merely outranked: the document's
    # similarity to its former topic collapses.
    after = score_of("doc-1", "vehicle repair")
    assert before > 0.5
    assert after < 0.2
    assert ranked(semantic_engine, "recipe kitchen")[0] == "doc-1"
    assert semantic_engine.vector_count == 2
    semantic_engine.validate()


def test_deletion_removes_the_vector(semantic_engine: SearchEngine) -> None:
    index(semantic_engine, {"doc-1": CAR, "doc-2": SEARCH})
    semantic_engine.delete_document("doc-1")

    assert semantic_engine.vector_count == 1
    assert "doc-1" not in ranked(semantic_engine, "vehicle repair")
    semantic_engine.validate()


def test_deleting_everything_empties_the_vector_index(semantic_engine: SearchEngine) -> None:
    index(semantic_engine, {"doc-1": CAR, "doc-2": SEARCH})
    semantic_engine.delete_document("doc-1")
    semantic_engine.delete_document("doc-2")

    assert semantic_engine.vector_count == 0
    semantic_engine.validate()


# ----------------------------------------------------------------------
# Embedding as a mutation precondition
# ----------------------------------------------------------------------


class BrokenEmbedder(FakeEmbedder):
    """An embedder that fails on demand, to test the write path's ordering."""

    def __init__(self) -> None:
        super().__init__()
        self.fail = False

    def embed_documents(self, texts, /):  # type: ignore[no-untyped-def]
        if self.fail:
            raise EmbeddingError("the model is unavailable")
        return super().embed_documents(texts)


def test_an_embedding_failure_changes_nothing(storage_path) -> None:  # type: ignore[no-untyped-def]
    """Embedding runs before the durable write, so a failure leaves no trace."""
    embedder = BrokenEmbedder()
    engine = SearchEngine(SqliteDocumentStore.open(storage_path), embedder=embedder)
    engine.initialize()
    index(engine, {"doc-1": CAR})
    embedder.fail = True

    with pytest.raises(EmbeddingError):
        engine.index_document(Document(document_id="doc-2", text=SEARCH))

    # No document, no generation consumed, no degradation: the mutation simply
    # did not happen.
    assert engine.stats().document_count == 1
    assert engine.generation == 1
    assert engine.status().state is NodeState.READY
    engine.validate()
    engine.close()


def test_a_replicated_embedding_failure_does_not_advance_the_generation() -> None:
    """A replica must never look synchronized when it could not embed."""
    embedder = BrokenEmbedder()
    replica = SearchEngine(InMemoryDocumentStore(), embedder=embedder)
    replica.initialize()
    replica.apply_replicated_put(Document(document_id="doc-1", text=CAR), 1)
    embedder.fail = True

    with pytest.raises(EmbeddingError):
        replica.apply_replicated_put(Document(document_id="doc-2", text=SEARCH), 2)

    assert replica.generation == 1
    assert replica.stats().document_count == 1
    replica.validate()


def test_a_duplicate_replication_costs_no_embedding() -> None:
    """The generation is checked first, so redelivery does not run the model."""
    embedder = BrokenEmbedder()
    replica = SearchEngine(InMemoryDocumentStore(), embedder=embedder)
    replica.initialize()
    replica.apply_replicated_put(Document(document_id="doc-1", text=CAR), 1)
    embedder.fail = True

    # Would raise if the replica embedded before classifying the generation.
    assert replica.apply_replicated_put(Document(document_id="doc-1", text=CAR), 1) is (
        ReplicationOutcome.DUPLICATE
    )


def test_a_gap_costs_no_embedding() -> None:
    embedder = BrokenEmbedder()
    replica = SearchEngine(InMemoryDocumentStore(), embedder=embedder)
    replica.initialize()
    embedder.fail = True

    assert replica.apply_replicated_put(Document(document_id="doc-1", text=CAR), 5) is (
        ReplicationOutcome.GAP
    )
    assert replica.status().state is NodeState.OUT_OF_SYNC


# ----------------------------------------------------------------------
# Replication and recovery
# ----------------------------------------------------------------------


def test_replicated_writes_build_semantic_state(semantic_engine: SearchEngine) -> None:
    semantic_engine.apply_replicated_put(Document(document_id="doc-1", text=CAR), 1)

    assert semantic_engine.vector_count == 1
    assert ranked(semantic_engine, "vehicle repair") == ["doc-1"]
    semantic_engine.validate()


def test_replicated_deletes_remove_semantic_state(semantic_engine: SearchEngine) -> None:
    semantic_engine.apply_replicated_put(Document(document_id="doc-1", text=CAR), 1)
    semantic_engine.apply_replicated_delete("doc-1", 2)

    assert semantic_engine.vector_count == 0
    semantic_engine.validate()


def test_startup_rebuilds_vectors_from_documents(storage_path) -> None:  # type: ignore[no-untyped-def]
    """Vectors are derived: nothing is persisted, everything is re-embedded."""
    embedder = FakeEmbedder()
    engine = SearchEngine(SqliteDocumentStore.open(storage_path), embedder=embedder)
    engine.initialize()
    index(engine, {"doc-1": CAR, "doc-2": SEARCH, "doc-3": COOKING})
    before = ranked(engine, "vehicle repair")
    engine.close()

    restarted = SearchEngine(SqliteDocumentStore.open(storage_path), embedder=FakeEmbedder())
    try:
        report = restarted.initialize()

        assert report.vector_count == 3
        assert report.semantic_duration_seconds >= 0.0
        assert ranked(restarted, "vehicle repair") == before
        restarted.validate()
    finally:
        restarted.close()


def test_resynchronization_rebuilds_vectors(semantic_engine: SearchEngine) -> None:
    source = SearchEngine(InMemoryDocumentStore(), embedder=FakeEmbedder())
    source.initialize()
    index(source, {"doc-1": CAR, "doc-2": SEARCH})

    documents, generation = source.export_snapshot()
    semantic_engine.resynchronize(documents, generation)

    assert semantic_engine.vector_count == 2
    assert ranked(semantic_engine, "vehicle repair") == ranked(source, "vehicle repair")
    semantic_engine.validate()


def test_two_copies_with_the_same_documents_rank_identically() -> None:
    """Equivalence, not bit-identity: same ordering, scores within tolerance."""
    corpus = {"doc-1": CAR, "doc-2": SEARCH, "doc-3": COOKING}
    first = SearchEngine(InMemoryDocumentStore(), embedder=FakeEmbedder())
    first.initialize()
    second = SearchEngine(InMemoryDocumentStore(), embedder=FakeEmbedder())
    second.initialize()
    index(first, corpus)
    # The second copy reaches the same state by replication instead, which is
    # how a replica actually gets there.
    for generation, (document_id, text) in enumerate(corpus.items(), start=1):
        second.apply_replicated_put(Document(document_id=document_id, text=text), generation)

    for query in ("vehicle repair", "query ranking", "recipe kitchen"):
        left = first.semantic_search(first.embed_query(query), limit=10).results
        right = second.semantic_search(second.embed_query(query), limit=10).results

        assert [hit.document_id for hit in left] == [hit.document_id for hit in right]
        np.testing.assert_allclose(
            [hit.score for hit in left], [hit.score for hit in right], rtol=1e-6, atol=1e-6
        )


# ----------------------------------------------------------------------
# Gating
# ----------------------------------------------------------------------


def test_a_non_ready_engine_refuses_semantic_search(semantic_engine: SearchEngine) -> None:
    semantic_engine.mark_out_of_sync("forced for the test")

    with pytest.raises(EngineNotReadyError):
        semantic_engine.semantic_search(np.zeros(5, dtype=np.float32), limit=5)


def test_validate_detects_a_vector_index_that_lost_a_document(
    semantic_engine: SearchEngine,
) -> None:
    index(semantic_engine, {"doc-1": CAR})
    semantic_engine._vectors.remove("doc-1")

    with pytest.raises(Exception, match="different documents"):
        semantic_engine.validate()
