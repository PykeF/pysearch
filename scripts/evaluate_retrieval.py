"""Compare lexical, semantic and hybrid retrieval on the project's labelled set.

    uv run --extra semantic python scripts/evaluate_retrieval.py --develop
    uv run --extra semantic python scripts/evaluate_retrieval.py

``--develop`` runs the parameter experiment on the **development** queries: it
varies the RRF constant and the candidate depth and prints what each choice
scores. That is how the defaults in ``app/hybrid/fusion.py`` were picked.

The default run evaluates all three modes on the **held-out evaluation**
queries, with those parameters frozen. Tuning on the queries you then report is
how a measurement turns into an advertisement, so the two are kept apart.

This is a small synthetic set written for this project. It supports statements
about *these* queries and nothing wider.

Metrics
-------

Recall@k   fraction of the relevant documents that appear in the top k
MRR        mean reciprocal rank of the first relevant document
"""

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from app.hybrid.fusion import FusionConfig
from app.search.document import Document
from app.search.engine import SearchEngine
from app.semantic.embedder import Model2VecEmbedder
from app.storage.sqlite_store import IN_MEMORY, SqliteDocumentStore
from scripts.evaluation_data import (
    CATEGORIES,
    CORPUS,
    DEVELOPMENT_QUERIES,
    EVALUATION_QUERIES,
    LabelledQuery,
)

LIMIT = 10
RETRIEVE = 20

Searcher = Callable[[str, int], list[str]]


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
class Scores:
    """What one retrieval mode scored over a set of queries."""

    recall_at_5: float
    recall_at_10: float
    mrr: float
    per_query: dict[str, float]


def evaluate(search: Searcher, queries: Sequence[LabelledQuery]) -> Scores:
    """Score one retrieval mode over every query in a set."""
    recall5: list[float] = []
    recall10: list[float] = []
    ranks: list[float] = []
    per_query: dict[str, float] = {}

    for labelled in queries:
        ranked = search(labelled.query, RETRIEVE)
        recall5.append(recall_at_k(ranked, labelled.relevant, 5))
        recall10.append(recall_at_k(ranked, labelled.relevant, 10))
        rank = reciprocal_rank(ranked, labelled.relevant)
        ranks.append(rank)
        per_query[labelled.query] = rank

    count = len(queries)
    return Scores(
        recall_at_5=sum(recall5) / count,
        recall_at_10=sum(recall10) / count,
        mrr=sum(ranks) / count,
        per_query=per_query,
    )


def build_engine() -> SearchEngine:
    """One engine holding the whole corpus, with semantic retrieval enabled."""
    engine = SearchEngine(SqliteDocumentStore.open(IN_MEMORY), embedder=Model2VecEmbedder.load())
    engine.initialize()
    for document_id, text in CORPUS.items():
        engine.index_document(Document(document_id=document_id, text=text))
    return engine


def searchers(engine: SearchEngine, config: FusionConfig) -> dict[str, Searcher]:
    """The three retrieval modes, as interchangeable ranked-id functions."""
    return {
        "BM25": lambda query, limit: [
            hit.document_id for hit in engine.search(query, limit).results
        ],
        "semantic": lambda query, limit: [
            hit.document_id
            for hit in engine.semantic_search(engine.embed_query(query), limit).results
        ],
        "hybrid": lambda query, limit: [
            hit.document_id for hit in engine.hybrid_search(query, limit, config).results
        ],
    }


def develop(engine: SearchEngine) -> int:
    """Choose the RRF constant and candidate depth on the development queries."""
    print("PARAMETER DEVELOPMENT — development queries only")
    print(f"{len(CORPUS)} documents, {len(DEVELOPMENT_QUERIES)} development queries\n")

    baselines = searchers(engine, FusionConfig())
    for name in ("BM25", "semantic"):
        scores = evaluate(baselines[name], DEVELOPMENT_QUERIES)
        print(
            f"  {name:<9} Recall@5 {scores.recall_at_5:.3f}   "
            f"Recall@10 {scores.recall_at_10:.3f}   MRR {scores.mrr:.3f}"
        )

    print(f"\n{'rrf_k':>6} {'depth':>7} {'Recall@5':>9} {'Recall@10':>10} {'MRR':>7}")
    print("-" * 44)
    best: tuple[float, float, int, int] | None = None
    for multiplier in (1, 2, 5):
        for rrf_k in (10, 30, 60):
            config = FusionConfig(rrf_k=rrf_k, candidate_multiplier=multiplier)
            scores = evaluate(searchers(engine, config)["hybrid"], DEVELOPMENT_QUERIES)
            print(
                f"{rrf_k:>6} {config.candidate_depth(LIMIT):>7} "
                f"{scores.recall_at_5:>9.3f} {scores.recall_at_10:>10.3f} {scores.mrr:>7.3f}"
            )
            candidate = (scores.mrr, scores.recall_at_5, rrf_k, multiplier)
            if best is None or candidate[:2] > best[:2]:
                best = candidate

    if best is not None:
        chosen = FusionConfig(rrf_k=best[2], candidate_multiplier=best[3])
        print(
            f"\nbest on these queries: rrf_k={best[2]}, "
            f"candidate depth={chosen.candidate_depth(LIMIT)} "
            f"(MRR {best[0]:.3f}, Recall@5 {best[1]:.3f})"
        )
    print("\nThese numbers select the defaults. They are not an evaluation result:")
    print("they are measured on the queries that chose them, so they are optimistic.")
    return 0


def report(engine: SearchEngine) -> int:
    """Evaluate the three modes on the held-out queries with frozen parameters."""
    config = FusionConfig()
    modes = searchers(engine, config)
    scored = {name: evaluate(search, EVALUATION_QUERIES) for name, search in modes.items()}

    print("HELD-OUT EVALUATION — parameters frozen from the development queries")
    print(f"rrf_k={config.rrf_k}, candidate depth={config.candidate_depth(LIMIT)}")
    print(f"{len(CORPUS)} documents, {len(EVALUATION_QUERIES)} evaluation queries")
    print("small synthetic project-owned set; not a general retrieval benchmark\n")

    print(f"{'mode':<10} {'Recall@5':>9} {'Recall@10':>10} {'MRR':>7}")
    print("-" * 40)
    for name, scores in scored.items():
        print(
            f"{name:<10} {scores.recall_at_5:>9.3f} {scores.recall_at_10:>10.3f} {scores.mrr:>7.3f}"
        )

    print("\nBy category (MRR; sample sizes are small, so read these as direction only)")
    print(f"{'category':<12} {'n':>3} {'BM25':>7} {'semantic':>9} {'hybrid':>7}")
    print("-" * 42)
    for category in CATEGORIES:
        group = [query for query in EVALUATION_QUERIES if query.category == category]
        if not group:
            continue
        row = [
            sum(scored[name].per_query[query.query] for query in group) / len(group)
            for name in ("BM25", "semantic", "hybrid")
        ]
        print(f"{category:<12} {len(group):>3} {row[0]:>7.2f} {row[1]:>9.2f} {row[2]:>7.2f}")

    print("\nPer query (reciprocal rank of the first relevant document)")
    print(f"{'query':<44} {'cat':<11} {'BM25':>6} {'sem':>6} {'hyb':>6}")
    print("-" * 80)
    losses: list[str] = []
    for labelled in EVALUATION_QUERIES:
        values = [scored[name].per_query[labelled.query] for name in ("BM25", "semantic", "hybrid")]
        worse = values[2] < max(values[0], values[1]) - 1e-9
        if worse:
            losses.append(labelled.query)
        print(
            f"{labelled.query!r:<44} {labelled.category:<11} "
            f"{values[0]:>6.2f} {values[1]:>6.2f} {values[2]:>6.2f}"
            f"{'   <- hybrid below its best input' if worse else ''}"
        )

    print(
        f"\nhybrid ranked below its best input on {len(losses)} of "
        f"{len(EVALUATION_QUERIES)} queries"
    )
    for query in losses:
        print(f"  - {query!r}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare lexical, semantic and hybrid retrieval on the labelled set.",
        epilog=(
            "Without --develop, all three modes are evaluated on the held-out queries "
            "with the parameters frozen. See docs/evaluation.md for the results and "
            "their limitations."
        ),
    )
    parser.add_argument(
        "--develop",
        action="store_true",
        help="run the parameter experiment on the development queries",
    )
    arguments = parser.parse_args()

    engine = build_engine()
    try:
        return develop(engine) if arguments.develop else report(engine)
    finally:
        engine.close()


if __name__ == "__main__":
    raise SystemExit(main())
