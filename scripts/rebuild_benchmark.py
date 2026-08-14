"""Measure how startup recovery scales with corpus size.

Run it with::

    uv run python scripts/rebuild_benchmark.py

This exists to put a number on the one real cost of the rebuild-on-startup
design: the index is derived, so every start re-analyses the whole corpus.

It is a local development measurement on synthetic documents, not a benchmark.
It says nothing about production throughput, and the absolute numbers depend
entirely on the machine it runs on. Everything happens in a temporary
directory, so no artifacts are produced.
"""

import tempfile
import time
from pathlib import Path

from app.search.document import Document
from app.search.engine import SearchEngine
from app.storage.sqlite_store import SqliteDocumentStore

CORPUS_SIZES = (100, 1_000, 10_000)

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
]

WORDS_PER_DOCUMENT = 40


def synthetic_text(seed: int) -> str:
    """Build a deterministic pseudo-document from the fixed vocabulary."""
    words = [
        VOCABULARY[(seed * 7 + offset) % len(VOCABULARY)] for offset in range(WORDS_PER_DOCUMENT)
    ]
    return " ".join(words)


def measure(document_count: int, directory: Path) -> tuple[float, float]:
    """Return (seconds to index, seconds to rebuild) for a corpus of this size."""
    database = directory / f"bench-{document_count}.db"

    engine = SearchEngine(SqliteDocumentStore.open(database))
    engine.initialize()
    started = time.perf_counter()
    for n in range(document_count):
        engine.index_document(Document(document_id=f"doc-{n}", text=synthetic_text(n)))
    index_seconds = time.perf_counter() - started
    engine.close()

    restarted = SearchEngine(SqliteDocumentStore.open(database))
    report = restarted.initialize()
    restarted.close()

    return index_seconds, report.duration_seconds


def main() -> None:
    print("local development measurement — synthetic documents, not a benchmark")
    print(f"{WORDS_PER_DOCUMENT} words per document\n")
    print(f"{'documents':>10}  {'index (s)':>10}  {'rebuild (s)':>12}  {'docs/s rebuilt':>15}")
    print("-" * 54)

    with tempfile.TemporaryDirectory() as directory:
        for document_count in CORPUS_SIZES:
            index_seconds, rebuild_seconds = measure(document_count, Path(directory))
            rate = document_count / rebuild_seconds if rebuild_seconds else float("inf")
            print(
                f"{document_count:>10}  {index_seconds:>10.3f}  "
                f"{rebuild_seconds:>12.3f}  {rate:>15,.0f}"
            )


if __name__ == "__main__":
    main()
