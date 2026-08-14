"""Errors raised by the storage layer.

Database-specific exceptions are wrapped at the storage boundary so that no
caller — and in particular no HTTP response — ever sees a raw driver error or a
fragment of SQL.
"""


class StorageError(Exception):
    """A storage operation failed."""


class StorageInitializationError(StorageError):
    """Durable storage could not be opened or prepared."""
