"""Generation sequencing: the proof that a replica really is synchronized.

Generations are a contiguous per-shard mutation sequence, and every rule here
exists to protect one inference: *equal generation means equal state*. If a
replica could skip a number, it could sit at the primary's generation while
missing a mutation, and every downstream decision built on that comparison —
whether to serve, whether to resynchronize — would be wrong.
"""

from collections.abc import Sequence
from pathlib import Path

import pytest

from app.search.document import Document
from app.search.engine import NodeState, ReplicationOutcome, SearchEngine
from app.search.errors import DocumentNotFoundError, EngineNotReadyError
from app.storage.sqlite_store import IN_MEMORY, SqliteDocumentStore


def new_engine() -> SearchEngine:
    engine = SearchEngine(SqliteDocumentStore.open(IN_MEMORY))
    engine.initialize()
    return engine


def document(n: int, text: str | None = None) -> Document:
    return Document(document_id=f"doc-{n}", text=text if text is not None else f"text {n}")


def corpus(engine: SearchEngine) -> dict[str, str]:
    """The logical content of a copy, independent of how it got there."""
    documents, _ = engine.export_snapshot()
    return {stored.document_id: stored.text for stored in documents}


def stored_corpus(engine: SearchEngine) -> dict[str, str]:
    """The durable content, read straight from storage.

    Needed where the copy is deliberately refusing to serve: every engine-level
    read is gated on READY, which is the behaviour under test, so the assertion
    goes to the authoritative copy instead.
    """
    return {stored.document_id: stored.text for stored in engine._store.iter_documents()}


@pytest.fixture
def replica() -> SearchEngine:
    return new_engine()


# ----------------------------------------------------------------------
# The six sequencing proofs
# ----------------------------------------------------------------------


def test_a_replica_at_generation_4_rejects_generation_6(replica: SearchEngine) -> None:
    """The gap case: refusing is the whole point."""
    for n in range(1, 5):
        assert replica.apply_replicated_put(document(n), n) is ReplicationOutcome.APPLIED
    assert replica.generation == 4

    outcome = replica.apply_replicated_put(document(6), 6)

    assert outcome is ReplicationOutcome.GAP
    # Nothing was applied: taking the newest mutation while missing an earlier
    # one would leave a corpus that never existed anywhere.
    assert replica.generation == 4
    assert "doc-6" not in stored_corpus(replica)
    assert replica.status().state is NodeState.OUT_OF_SYNC


def test_generation_5_then_6_is_accepted(replica: SearchEngine) -> None:
    """The contiguous case."""
    for n in range(1, 5):
        replica.apply_replicated_put(document(n), n)

    assert replica.apply_replicated_put(document(5), 5) is ReplicationOutcome.APPLIED
    assert replica.apply_replicated_put(document(6), 6) is ReplicationOutcome.APPLIED

    assert replica.generation == 6
    assert replica.status().ready is True
    assert {"doc-5", "doc-6"} <= set(corpus(replica))


def test_a_repeated_generation_5_is_idempotent(replica: SearchEngine) -> None:
    """The redelivery case: a retried mutation must not be an error."""
    for n in range(1, 6):
        replica.apply_replicated_put(document(n), n)
    before = corpus(replica)

    outcome = replica.apply_replicated_put(document(5), 5)

    assert outcome is ReplicationOutcome.DUPLICATE
    assert replica.generation == 5
    assert corpus(replica) == before
    assert replica.status().ready is True


def test_a_replica_made_out_of_sync_by_a_gap_refuses_to_serve(
    replica: SearchEngine,
) -> None:
    """An out-of-sync copy must be invisible to search, not merely flagged."""
    replica.apply_replicated_put(document(1), 1)
    replica.apply_replicated_put(document(3), 3)

    assert replica.status().state is NodeState.OUT_OF_SYNC
    assert replica.status().ready is False

    # Every path the coordinator would use to select or query a copy refuses,
    # so an unsynchronized replica cannot contribute statistics or results.
    for call in (
        lambda: replica.search("text", limit=10),
        lambda: replica.corpus_stats(["text"]),
        lambda: replica.stats(),
        lambda: replica.export_snapshot(),
    ):
        with pytest.raises(EngineNotReadyError):
            call()


def test_full_resynchronization_repairs_a_gap(replica: SearchEngine) -> None:
    """Recovery: OUT_OF_SYNC -> resync -> READY, with the gap actually filled."""
    primary = new_engine()
    for n in range(1, 7):
        primary.index_document(document(n))

    replica.apply_replicated_put(document(1), 1)
    replica.apply_replicated_put(document(3), 3)
    assert replica.status().state is NodeState.OUT_OF_SYNC

    documents, generation = primary.export_snapshot()
    replica.resynchronize(documents, generation)

    assert replica.status().state is NodeState.READY
    assert replica.generation == primary.generation == 6
    # The mutation the replica had missed is present after recovery.
    assert "doc-2" in corpus(replica)
    replica.validate()


def test_equal_generations_after_recovery_mean_equal_state(replica: SearchEngine) -> None:
    """The inference the whole mechanism exists to license."""
    primary = new_engine()
    for n in range(1, 5):
        primary.index_document(document(n))
    primary.index_document(document(2, text="replaced text"))
    primary.delete_document("doc-3")

    # The replica took a different path to get there: it missed mutations,
    # went out of sync, and was repaired.
    replica.apply_replicated_put(document(1), 1)
    replica.apply_replicated_put(document(9), 9)
    documents, generation = primary.export_snapshot()
    replica.resynchronize(documents, generation)

    assert replica.generation == primary.generation
    assert corpus(replica) == corpus(primary)
    assert replica.stats() == primary.stats()


# ----------------------------------------------------------------------
# Sequencing details
# ----------------------------------------------------------------------


def test_a_primary_advances_by_exactly_one_per_mutation() -> None:
    engine = new_engine()

    engine.index_document(document(1))
    assert engine.generation == 1
    engine.index_document(document(2))
    assert engine.generation == 2
    engine.index_document(document(1, text="replacement"))
    assert engine.generation == 3
    engine.delete_document("doc-1")
    assert engine.generation == 4


def test_a_delete_that_matches_nothing_consumes_no_generation() -> None:
    engine = new_engine()
    engine.index_document(document(1))

    with pytest.raises(DocumentNotFoundError):
        engine.delete_document("doc-404")

    # Generations number applied mutations, and a no-op is not one; consuming a
    # number here would make every replica look like it had missed something.
    assert engine.generation == 1
    engine.validate()


def test_replicated_deletes_are_applied_and_advance_the_generation(
    replica: SearchEngine,
) -> None:
    replica.apply_replicated_put(document(1), 1)

    assert replica.apply_replicated_delete("doc-1", 2) is ReplicationOutcome.APPLIED

    assert replica.generation == 2
    assert corpus(replica) == {}
    replica.validate()


def test_a_replicated_delete_of_an_absent_document_still_advances(
    replica: SearchEngine,
) -> None:
    """A replica may legitimately not hold what the primary is deleting.

    Answering with an error would strand the primary, so the deletion is a
    no-op that still takes the generation forward and keeps the copies aligned.
    """
    replica.apply_replicated_put(document(1), 1)

    assert replica.apply_replicated_delete("doc-404", 2) is ReplicationOutcome.APPLIED

    assert replica.generation == 2
    assert corpus(replica) == {"doc-1": "text 1"}
    replica.validate()


def test_a_repeated_delete_is_idempotent(replica: SearchEngine) -> None:
    replica.apply_replicated_put(document(1), 1)
    replica.apply_replicated_delete("doc-1", 2)

    assert replica.apply_replicated_delete("doc-1", 2) is ReplicationOutcome.DUPLICATE
    assert replica.generation == 2


def test_a_delete_with_a_gap_is_refused(replica: SearchEngine) -> None:
    replica.apply_replicated_put(document(1), 1)

    assert replica.apply_replicated_delete("doc-1", 5) is ReplicationOutcome.GAP

    assert replica.generation == 1
    assert replica.status().state is NodeState.OUT_OF_SYNC
    assert stored_corpus(replica) == {"doc-1": "text 1"}


def test_the_generation_survives_a_restart(storage_path: Path) -> None:
    path = storage_path

    engine = SearchEngine(SqliteDocumentStore.open(path))
    engine.initialize()
    engine.index_document(document(1))
    engine.index_document(document(2))
    engine.close()

    restarted = SearchEngine(SqliteDocumentStore.open(path))
    try:
        restarted.initialize()
        # Persisted with the documents, in the same transaction, so a copy can
        # never come back holding data at one generation and claiming another.
        assert restarted.generation == 2
        restarted.validate()
    finally:
        restarted.close()


def test_resynchronization_is_atomic_in_storage(replica: SearchEngine) -> None:
    replica.apply_replicated_put(document(1), 1)

    snapshot: Sequence[Document] = [document(7), document(8)]
    replica.resynchronize(snapshot, 12)

    assert corpus(replica) == {"doc-7": "text 7", "doc-8": "text 8"}
    assert replica.generation == 12
    replica.validate()
