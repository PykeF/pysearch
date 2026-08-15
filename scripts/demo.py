"""A deterministic walkthrough of the search engine.

Run it with::

    uv run python scripts/demo.py                    # lexical only
    uv run --extra semantic python scripts/demo.py   # all three retrieval modes

It uses the search core directly, without starting the web application, which
is the point: the retrieval code has no dependency on FastAPI.

The sections work bottom-up — the inverted index on its own, then the engine,
then a restart proving the corpus is durable, then the semantic and hybrid modes
that build on the same documents. The last two need the optional ``semantic``
extra; without it they are skipped with a note rather than failing, so the demo
works on a plain ``uv sync``.

A temporary directory is used so running the demo leaves nothing behind.
"""

import tempfile
from pathlib import Path

from app.hybrid.fusion import FusionConfig
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


def load_embedder() -> tuple[object | None, str]:
    """Return a real embedder, or None and the reason it is unavailable.

    The demo must work on a plain ``uv sync``, so a missing model2vec is an
    expected outcome rather than an error.
    """
    try:
        from app.semantic.embedder import Model2VecEmbedder

        return Model2VecEmbedder.load(), ""
    except Exception as error:
        # Broad on purpose: a missing extra, a missing model and a failed
        # download are all just "no semantic section", and the reason is shown.
        return None, str(error)


def show_semantic_and_hybrid(database: Path) -> None:
    """Compare the three retrieval modes on one corpus.

    A separate database from the lexical sections, because those deliberately
    delete and replace documents and this section wants the corpus intact.
    """
    print("\n" + "=" * 70)
    print("semantic and hybrid retrieval")
    print("=" * 70)

    embedder, reason = load_embedder()
    if embedder is None:
        print(f"\nSkipped — {reason}")
        print("\nRun `uv sync --extra semantic` and try again to see:")
        print("  - semantic search matching a paraphrase with no shared words")
        print("  - Reciprocal Rank Fusion combining both rankings")
        return

    engine = SearchEngine(SqliteDocumentStore.open(database), embedder=embedder)  # type: ignore[arg-type]
    engine.initialize()
    try:
        for document_id, text in CORPUS.items():
            engine.index_document(Document(document_id=document_id, text=text))
        print(f"\nindexed {len(CORPUS)} documents with vectors")

        # The first query is a paraphrase whose only lexical match is the stop
        # word "a" — so BM25 confidently returns the wrong document, and fusion
        # inherits that mistake. That is the failure mode documented in
        # docs/evaluation.md, reproduced here in six documents. The second query
        # shares real vocabulary with two documents, so both signals agree and
        # fusion behaves as intended.
        for query in ("preparing a meal", "splitting data over many computers"):
            print(f"\n{'-' * 70}\nquery {query!r}")

            lexical = engine.search(query, limit=3)
            print(f"\n  BM25          (matched {lexical.total})")
            if not lexical.results:
                print("    no results — no document contains any of these terms")
            for rank, hit in enumerate(lexical.results, start=1):
                print(f"    {rank}. {hit.document_id}  score={hit.score:.4f}")

            semantic = engine.semantic_search(engine.embed_query(query), limit=3)
            print(f"\n  semantic      (searched {semantic.total})")
            for rank, hit in enumerate(semantic.results, start=1):
                print(f"    {rank}. {hit.document_id}  score={hit.score:.4f}")

            hybrid = engine.hybrid_search(query, limit=3, config=FusionConfig())
            print(f"\n  hybrid (RRF)  (fused {hybrid.total} candidates)")
            for rank, hit in enumerate(hybrid.results, start=1):
                origin = f"lex={hit.lexical_rank or '-'} sem={hit.semantic_rank or '-'}"
                print(f"    {rank}. {hit.document_id}  score={hit.score:.6f}  [{origin}]")

        print("\n" + "-" * 70)
        print("The fusion score is a sum of reciprocal ranks — not a BM25 score,")
        print("not a cosine similarity, and not a probability. It means something")
        print("only relative to the other results in the same response.")
        print()
        print("Note the first query: BM25's only match was the stop word 'a', and")
        print("fusion promoted that wrong document above the one semantic search")
        print("ranked first. There is no stop-word filtering, so a confidently")
        print("wrong lexical hit outranks a correct semantic one. This is the")
        print("measured failure mode in docs/evaluation.md, not a bug in fusion.")
    finally:
        engine.close()


def main() -> None:
    show_index_internals()
    with tempfile.TemporaryDirectory() as directory:
        database = Path(directory) / "demo.db"
        show_engine(database)
        show_restart(database)
        show_semantic_and_hybrid(Path(directory) / "demo-semantic.db")


if __name__ == "__main__":
    main()
