"""A deterministic walkthrough of the search engine.

Run it with::

    uv run python scripts/demo.py

It uses the search core directly, without starting the web application, which
is the point: the retrieval code has no dependency on FastAPI.

The three sections work bottom-up — the inverted index on its own, then the
engine, then a restart proving the corpus is durable. A temporary directory is
used so running the demo leaves nothing behind.
"""

import tempfile
from pathlib import Path

from app.search.analysis import analyze
from app.search.document import Document
from app.search.engine import SearchEngine
from app.search.index import InvertedIndex
from app.storage.sqlite_store import SqliteDocumentStore

CORPUS = {
    "doc-1": "Distributed systems make search scalable across many machines.",
    "doc-2": "Search engines rank documents by relevance to a query.",
    "doc-3": "BM25 is the ranking function used by most lexical search engines.",
    "doc-4": "An inverted index maps each term to the documents containing it.",
    "doc-5": "Sharding splits an index so that each node holds part of the data.",
    "doc-6": "Cooking pasta well requires salted, vigorously boiling water.",
}

QUERIES = ("search", "search engines", "ranking function", "pasta", "quantum")


def show_index_internals() -> None:
    """Print the analysis output and a posting list for a couple of terms."""
    print("=" * 70)
    print("the index layer")
    print("=" * 70)

    sample = CORPUS["doc-1"]
    print(f"\ntext   : {sample}")
    print(f"tokens : {analyze(sample)}")

    index = InvertedIndex()
    for document_id, text in CORPUS.items():
        index.add_document(document_id, analyze(text))

    for term in ("search", "index"):
        print(f"\nposting list for {term!r}  (df={index.document_frequency(term)})")
        for posting in sorted(index.postings(term), key=lambda p: p.document_id):
            print(f"  {posting.document_id}: tf={posting.term_frequency}")


def print_results(engine: SearchEngine, query: str) -> None:
    """Run a query and print its ranked results."""
    outcome = engine.search(query, limit=3)
    print(f"\nquery {query!r}  (matched {outcome.total})")
    if not outcome.results:
        print("  no results")
        return
    for rank, hit in enumerate(outcome.results, start=1):
        print(f"  {rank}. {hit.document_id}  score={hit.score:.4f}  {hit.text}")


def start_engine(database: Path) -> SearchEngine:
    """Open storage and bring an engine up against it, exactly as startup does."""
    engine = SearchEngine(SqliteDocumentStore.open(database))
    report = engine.initialize()
    print(
        f"\nrecovered {report.document_count} documents from {database.name} "
        f"in {report.duration_seconds * 1000:.2f} ms"
    )
    return engine


def show_engine(database: Path) -> None:
    """Index the corpus, query it, then update and delete a document."""
    print("\n" + "=" * 70)
    print("the engine")
    print("=" * 70)

    engine = start_engine(database)
    try:
        for document_id, text in CORPUS.items():
            engine.index_document(Document(document_id=document_id, text=text))

        stats = engine.stats()
        print(f"\ndocuments {stats.document_count}", end="")
        print(f" | unique terms {stats.unique_term_count}", end="")
        print(f" | avg length {stats.average_document_length:.2f}")

        for query in QUERIES:
            print_results(engine, query)

        print("\n-- replacing doc-6 --")
        engine.index_document(
            Document(
                document_id="doc-6", text="Replication keeps copies of a shard on other nodes."
            )
        )
        print_results(engine, "pasta")
        print_results(engine, "replication")

        print("\n-- deleting doc-1 --")
        engine.delete_document("doc-1")
        print_results(engine, "search")

        engine.validate()
        print("\ninvariants hold")
    finally:
        engine.close()


def show_restart(database: Path) -> None:
    """Reopen the same database and prove the corpus and results survived."""
    print("\n" + "=" * 70)
    print("restart")
    print("=" * 70)

    engine = start_engine(database)
    try:
        stats = engine.stats()
        print(f"\ndocuments {stats.document_count}", end="")
        print(f" | unique terms {stats.unique_term_count}", end="")
        print(f" | avg length {stats.average_document_length:.2f}")

        # The update and the deletion from the previous section are still in
        # effect, because SQLite — not the in-memory index — is the corpus.
        for query in ("search", "replication", "pasta"):
            print_results(engine, query)

        engine.validate()
        print("\ninvariants hold after recovery")
    finally:
        engine.close()


def main() -> None:
    show_index_internals()
    with tempfile.TemporaryDirectory() as directory:
        database = Path(directory) / "demo.db"
        show_engine(database)
        show_restart(database)


if __name__ == "__main__":
    main()
