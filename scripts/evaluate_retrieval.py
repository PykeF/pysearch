"""Measure lexical and semantic retrieval side by side.

    uv run --extra semantic python scripts/evaluate_retrieval.py

Two independent retrieval systems answer the same labelled queries, and the
results are reported separately. They are deliberately **not** combined: how to
fuse two rankings is the next phase's question, and answering it here would hide
the thing this script exists to show — that the two fail in different places.

The corpus is synthetic and written for this project, so it can be committed
without licensing questions. It is far too small to say anything about retrieval
quality in general; the point is the measurement discipline, not the number.

Metrics
-------

Recall@k   fraction of relevant documents that appear in the top k
MRR        mean reciprocal rank of the first relevant document
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from app.search.document import Document
from app.search.engine import SearchEngine, SearchResults
from app.semantic.embedder import Model2VecEmbedder
from app.storage.sqlite_store import IN_MEMORY, SqliteDocumentStore

CORPUS: dict[str, str] = {
    # Vehicles, described without the words a user is likely to type.
    "veh-1": "automobile repair and servicing for engines and gearboxes",
    "veh-2": "how to keep a motor vehicle running smoothly through winter",
    "veh-3": "brake pads, tyre pressure and other roadworthiness checks",
    "veh-4": "replacing worn windscreen wipers before the rainy season",
    # Retrieval, described in its own vocabulary.
    "ir-1": "BM25 ranks documents using term frequency and inverse document frequency",
    "ir-2": "an inverted index maps each term to the documents that contain it",
    "ir-3": "vector similarity retrieves passages that mean the same thing",
    "ir-4": "tokenization splits text into the units an index actually stores",
    "ir-5": "stop words carry little signal but occupy most of a posting list",
    # Distributed systems.
    "sys-1": "sharding splits a dataset so that each node stores only part of it",
    "sys-2": "replication keeps a second durable copy so a lost node loses nothing",
    "sys-3": "a coordinator fans a query out to every shard and merges the answers",
    "sys-4": "a write is acknowledged only once every copy has committed it durably",
    "sys-5": "leader election lets a cluster agree on which node may accept writes",
    # Cooking, as an unrelated topic.
    "cook-1": "simmer the sauce gently while the pasta finishes cooking",
    "cook-2": "a sharp knife and a hot pan matter more than an expensive recipe",
    "cook-3": "let the dough rest before shaping it into loaves",
    # Near-identical documents distinguished only by an identifier. This is the
    # case exact matching is built for and embeddings are worst at: the
    # surrounding text is deliberately almost the same in all three.
    "err-1": "error code E2231 indicates a failed checksum on the write path",
    "err-2": "error code E4410 indicates a failed checksum on the write path",
    "err-3": "error code E7782 indicates a failed checksum on the write path",
    # Endpoint documentation, likewise near-identical apart from the path.
    "api-1": "the coordinator exposes /cluster/status for operators",
    "api-2": "the coordinator exposes /index/stats for operators",
    "api-3": "the coordinator exposes /search/semantic for operators",
}


@dataclass(frozen=True, slots=True)
class LabelledQuery:
    """A query and the documents a human would call relevant."""

    query: str
    relevant: frozenset[str]
    note: str


QUERIES: tuple[LabelledQuery, ...] = (
    # Paraphrases: the query shares few or no words with the answer.
    LabelledQuery(
        "car maintenance",
        frozenset({"veh-1", "veh-2", "veh-3", "veh-4"}),
        "paraphrase: 'car' appears in no document",
    ),
    LabelledQuery(
        "fixing a broken engine",
        frozenset({"veh-1", "veh-2"}),
        "paraphrase, partial overlap",
    ),
    LabelledQuery(
        "searching by meaning rather than keywords",
        frozenset({"ir-3"}),
        "conceptual, little lexical overlap",
    ),
    LabelledQuery(
        "surviving the loss of a machine",
        frozenset({"sys-2"}),
        "paraphrase of replication",
    ),
    LabelledQuery(
        "splitting data across machines",
        frozenset({"sys-1"}),
        "paraphrase of sharding",
    ),
    LabelledQuery(
        "making dinner",
        frozenset({"cook-1", "cook-2", "cook-3"}),
        "paraphrase of cooking",
    ),
    # Exact vocabulary: the query uses the document's own words.
    LabelledQuery(
        "how does BM25 work",
        frozenset({"ir-1"}),
        "exact keyword",
    ),
    LabelledQuery(
        "inverted index",
        frozenset({"ir-2"}),
        "exact phrase",
    ),
    LabelledQuery(
        "stop words posting list",
        frozenset({"ir-5"}),
        "exact terms",
    ),
    # Identifiers: one rare token decides the answer, and everything around it
    # is near-identical, so meaning cannot separate the candidates.
    LabelledQuery(
        "E2231",
        frozenset({"err-1"}),
        "identifier among near-identical documents",
    ),
    LabelledQuery(
        "E7782 checksum",
        frozenset({"err-3"}),
        "identifier among near-identical documents",
    ),
    LabelledQuery(
        "/index/stats",
        frozenset({"api-2"}),
        "path among near-identical documents",
    ),
    LabelledQuery(
        "/search/semantic endpoint",
        frozenset({"api-3"}),
        "path among near-identical documents",
    ),
)

K = 5


def recall_at_k(ranked: Sequence[str], relevant: frozenset[str], k: int) -> float:
    """Fraction of the relevant documents that appear in the top ``k``."""
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranked: Sequence[str], relevant: frozenset[str]) -> float:
    """One over the position of the first relevant document, or zero."""
    for position, document_id in enumerate(ranked, start=1):
        if document_id in relevant:
            return 1.0 / position
    return 0.0


@dataclass(frozen=True, slots=True)
class Evaluation:
    """What one retrieval mode scored across the labelled queries."""

    recall: float
    mrr: float
    ranks: list[float]
    candidates: list[int]


def evaluate(search: Callable[[str, int], SearchResults]) -> Evaluation:
    """Score one retrieval mode over every labelled query."""
    recalls: list[float] = []
    ranks: list[float] = []
    candidates: list[int] = []
    for labelled in QUERIES:
        outcome = search(labelled.query, 20)
        ranked = [hit.document_id for hit in outcome.results]
        recalls.append(recall_at_k(ranked, labelled.relevant, K))
        ranks.append(reciprocal_rank(ranked, labelled.relevant))
        candidates.append(outcome.total)
    return Evaluation(
        recall=sum(recalls) / len(recalls),
        mrr=sum(ranks) / len(ranks),
        ranks=ranks,
        candidates=candidates,
    )


def main() -> int:
    print("local development measurement on a small synthetic corpus — not a benchmark")
    print(f"{len(CORPUS)} documents, {len(QUERIES)} labelled queries, k={K}\n")

    embedder = Model2VecEmbedder.load()
    engine = SearchEngine(SqliteDocumentStore.open(IN_MEMORY), embedder=embedder)
    engine.initialize()
    for document_id, text in CORPUS.items():
        engine.index_document(Document(document_id=document_id, text=text))

    def lexical(query: str, limit: int) -> SearchResults:
        return engine.search(query, limit)

    def semantic(query: str, limit: int) -> SearchResults:
        return engine.semantic_search(engine.embed_query(query), limit)

    lexical_result = evaluate(lexical)
    semantic_result = evaluate(semantic)

    print(f"{'query':<46} {'BM25 RR':>8} {'sem RR':>7} {'BM25 hits':>10} {'sem hits':>9}")
    print("-" * 96)
    for position, labelled in enumerate(QUERIES):
        print(
            f"{labelled.query!r:<46} "
            f"{lexical_result.ranks[position]:>8.2f} "
            f"{semantic_result.ranks[position]:>7.2f} "
            f"{lexical_result.candidates[position]:>10} "
            f"{semantic_result.candidates[position]:>9}   ({labelled.note})"
        )
    print("-" * 96)
    print(f"{'Recall@' + str(K):<46} {lexical_result.recall:>8.2f} {semantic_result.recall:>7.2f}")
    print(f"{'MRR':<46} {lexical_result.mrr:>8.2f} {semantic_result.mrr:>7.2f}")

    print("\nRR is the reciprocal rank of the first relevant document: 1.00 means it came")
    print("first, 0.00 means it never appeared. 'hits' is how many documents each mode")
    print("considered a candidate at all — and that column is the real difference. BM25")
    print("has a notion of not matching, so it returns nothing for a paraphrase and")
    print("exactly one document for an identifier. A similarity is defined for every")
    print("document, so semantic search always ranks the whole corpus and never says")
    print("'no'. The two are reported separately and never combined: how to fuse them")
    print("is the next phase's question.")

    engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
