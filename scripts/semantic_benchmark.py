"""Measure what semantic retrieval costs.

    uv run --extra semantic python scripts/semantic_benchmark.py

Every number here is a local development measurement on one machine with a
synthetic corpus. None of it is a production benchmark, and none of it should be
used to argue about how the system scales.

The reason it exists is that two Phase 5 decisions were made *pending
measurement* — rebuilding vectors at startup instead of persisting them, and
searching exhaustively instead of approximately. These are the numbers that
would eventually overturn either one.
"""

import tempfile
import time
from pathlib import Path

from app.search.document import Document
from app.search.engine import SearchEngine
from app.semantic.embedder import Model2VecEmbedder
from app.storage.sqlite_store import SqliteDocumentStore

CORPUS_SIZES = (100, 1_000, 5_000)

VOCABULARY = [
    "distributed",
    "search",
    "index",
    "ranking",
    "function",
    "document",
    "term",
    "posting",
    "list",
    "frequency",
    "shard",
    "replica",
    "node",
    "cluster",
    "query",
    "scoring",
    "relevance",
    "corpus",
    "token",
    "analysis",
    "storage",
    "vector",
    "embedding",
    "similarity",
    "semantic",
    "retrieval",
    "neighbour",
    "cosine",
    "dimension",
]

WORDS_PER_DOCUMENT = 40
QUERY = "distributed vector similarity over a sharded corpus"


def synthetic_text(seed: int) -> str:
    """A deterministic pseudo-document from the fixed vocabulary."""
    return " ".join(
        VOCABULARY[(seed * 7 + offset) % len(VOCABULARY)] for offset in range(WORDS_PER_DOCUMENT)
    )


def time_it(action, repeats: int) -> float:  # type: ignore[no-untyped-def]
    """Mean seconds per call over ``repeats`` calls."""
    started = time.perf_counter()
    for _ in range(repeats):
        action()
    return (time.perf_counter() - started) / repeats


def main() -> int:
    print("local development measurements — not a benchmark\n")

    started = time.perf_counter()
    embedder = Model2VecEmbedder.load()
    load_seconds = time.perf_counter() - started
    dimension = embedder.identity.dimension
    print(f"model            : {embedder.identity.fingerprint}")
    print(f"load time        : {load_seconds:.3f} s")

    batch = [synthetic_text(n) for n in range(256)]
    started = time.perf_counter()
    embedder.embed_documents(batch)
    batch_seconds = time.perf_counter() - started
    print(f"embedding        : {len(batch) / batch_seconds:,.0f} documents/s (batched)")

    query_seconds = time_it(lambda: embedder.embed_query(QUERY), repeats=200)
    print(f"query embedding  : {query_seconds * 1000:.3f} ms\n")

    print(
        f"{'documents':>10}  {'index (s)':>10}  {'rebuild: lexical':>17}  "
        f"{'semantic':>10}  {'search (ms)':>12}  {'vectors':>9}"
    )
    print("-" * 78)

    with tempfile.TemporaryDirectory() as directory:
        for size in CORPUS_SIZES:
            database = Path(directory) / f"bench-{size}.db"
            engine = SearchEngine(SqliteDocumentStore.open(database), embedder=embedder)
            engine.initialize()

            started = time.perf_counter()
            for n in range(size):
                engine.index_document(Document(document_id=f"doc-{n}", text=synthetic_text(n)))
            index_seconds = time.perf_counter() - started
            engine.close()

            restarted = SearchEngine(SqliteDocumentStore.open(database), embedder=embedder)
            report = restarted.initialize()

            vector = restarted.embed_query(QUERY)
            search_seconds = time_it(
                lambda engine=restarted, query=vector: engine.semantic_search(query, 10),
                repeats=100,
            )
            restarted.close()

            # float32, one row per document.
            megabytes = size * dimension * 4 / 1_000_000
            print(
                f"{size:>10}  {index_seconds:>10.3f}  {report.duration_seconds:>17.3f}  "
                f"{report.semantic_duration_seconds:>10.3f}  {search_seconds * 1000:>12.3f}  "
                f"{megabytes:>7.1f} MB"
            )

    print("\nvector memory is N x d x 4 bytes for float32, excluding Python overhead")
    print(f"({dimension} dimensions here). Search is one N x d matrix multiply, so its cost")
    print("grows linearly with the corpus — that column is what would eventually justify")
    print("an approximate index, and the rebuild column is what would justify persisting")
    print("vectors instead of re-embedding at startup.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
