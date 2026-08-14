"""A SQLite-backed document store.

SQLite is used for what it is good at — atomic, durable, crash-recoverable
storage of the corpus — and for nothing else. Analysis, indexing and BM25
scoring remain PySearch's own work; there is no SQL involved in retrieval.

Durability
----------

The default rollback journal is kept rather than WAL, with
``synchronous=FULL``. WAL's advantage is letting readers run alongside a
writer, and the engine's global lock means SQLite never sees concurrent access
in the first place, so that advantage is unrealised here while the sidecar
``-wal``/``-shm`` files would complicate "the corpus is one file". WAL becomes
worth revisiting exactly when that global lock is relaxed to allow concurrent
reads.

``synchronous=FULL`` means a commit has reached the platform's durable storage
before it returns, which is what makes "the API reported success" mean "the
write survived".

Threading
---------

One connection is shared across FastAPI's thread pool with
``check_same_thread=False``. That check exists to catch *unsynchronised*
sharing of a connection; here every call is made under the engine's lock, so
the serialisation the check protects is provided explicitly instead. The two
decisions are a package: relaxing the engine lock would require revisiting the
connection strategy.

Portability
-----------

The schema deliberately avoids ``STRICT`` tables (SQLite 3.37+) and upsert
syntax (3.24+), so the project does not raise its minimum SQLite version for
conveniences it does not need. Document validity is the domain model's job, not
the database's.
"""

import sqlite3
from collections.abc import Iterator
from pathlib import Path

from app.search.document import Document
from app.storage.errors import StorageError, StorageInitializationError

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    text TEXT NOT NULL
)
"""

_SCHEMA_VERSION = 1

#: Passed to ``sqlite3.connect`` to request a private, non-persistent database.
IN_MEMORY = ":memory:"


class SqliteDocumentStore:
    """Stores documents in a SQLite database."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    @classmethod
    def open(cls, path: Path | str) -> "SqliteDocumentStore":
        """Open (creating if necessary) the database at ``path``.

        Construction is a factory rather than a constructor so that opening a
        file and running DDL is an explicit act rather than a side effect of
        building an object.

        Raises:
            StorageInitializationError: if the database cannot be opened or the
                schema cannot be applied.
        """
        location = str(path)
        try:
            if location != IN_MEMORY:
                Path(location).parent.mkdir(parents=True, exist_ok=True)

            connection = sqlite3.connect(location, check_same_thread=False)
            connection.execute("PRAGMA synchronous = FULL")
            connection.execute(_SCHEMA)
            connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
            connection.commit()
        except (sqlite3.Error, OSError) as error:
            raise StorageInitializationError(f"could not open storage at {location}") from error

        return cls(connection)

    def put(self, document: Document) -> bool:
        """Store a document, replacing any document with the same id."""
        try:
            with self._connection:
                existing = self._connection.execute(
                    "SELECT 1 FROM documents WHERE document_id = ?",
                    (document.document_id,),
                ).fetchone()
                self._connection.execute(
                    "INSERT OR REPLACE INTO documents (document_id, text) VALUES (?, ?)",
                    (document.document_id, document.text),
                )
        except sqlite3.Error as error:
            raise StorageError("failed to store document") from error

        return existing is None

    def get(self, document_id: str) -> Document | None:
        """Return a document, or ``None`` if it is not stored."""
        try:
            row = self._connection.execute(
                "SELECT document_id, text FROM documents WHERE document_id = ?",
                (document_id,),
            ).fetchone()
        except sqlite3.Error as error:
            raise StorageError("failed to read document") from error

        if row is None:
            return None
        return Document(document_id=row[0], text=row[1])

    def delete(self, document_id: str) -> bool:
        """Remove a document, reporting whether one existed."""
        try:
            with self._connection:
                cursor = self._connection.execute(
                    "DELETE FROM documents WHERE document_id = ?",
                    (document_id,),
                )
        except sqlite3.Error as error:
            raise StorageError("failed to delete document") from error

        return cursor.rowcount > 0

    def iter_documents(self) -> Iterator[Document]:
        """Yield every stored document, ordered by id so rebuilds are reproducible."""
        try:
            cursor = self._connection.execute(
                "SELECT document_id, text FROM documents ORDER BY document_id"
            )
            for document_id, text in cursor:
                yield Document(document_id=document_id, text=text)
        except sqlite3.Error as error:
            raise StorageError("failed to read documents") from error

    def count(self) -> int:
        """Return the number of stored documents."""
        try:
            row = self._connection.execute("SELECT COUNT(*) FROM documents").fetchone()
        except sqlite3.Error as error:
            raise StorageError("failed to count documents") from error

        count: int = row[0]
        return count

    def close(self) -> None:
        """Close the database connection."""
        try:
            self._connection.close()
        except sqlite3.Error as error:
            raise StorageError("failed to close storage") from error
