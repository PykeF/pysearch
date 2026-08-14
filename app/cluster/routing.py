"""Deterministic document-to-shard routing.

A document's shard is a pure function of its identifier and the shard count::

    shard_id = int(blake2b(document_id)) % shard_count

``hash()`` cannot be used for this. Python randomises string hashing per
process unless ``PYTHONHASHSEED`` is fixed, so the same document would route
differently after a restart, and differently again on another node — which
would silently scatter a document's history across shards.

BLAKE2b is used instead: it is in the standard library, it is fast, and its
output depends on nothing but the input bytes. The encoding (UTF-8), the digest
size (8 bytes) and the byte order (big-endian) are all fixed, because each of
them would change the result if left to a default.

Routing is plain modulo rather than consistent hashing. Modulo is easy to
reason about and easy to verify with pinned vectors; the price is that changing
``shard_count`` moves almost every document, so the shard count is fixed for
the lifetime of a cluster. That limitation is what motivates rebalancing work
later, and it is documented rather than pre-solved.
"""

import hashlib

_DIGEST_SIZE = 8


class ShardRouter:
    """Maps document identifiers onto a fixed number of shards."""

    def __init__(self, shard_count: int) -> None:
        if shard_count < 1:
            raise ValueError(f"shard_count must be at least 1, got {shard_count}")
        self._shard_count = shard_count

    @property
    def shard_count(self) -> int:
        """The number of shards this router distributes across."""
        return self._shard_count

    def shard_for(self, document_id: str) -> int:
        """Return the shard that owns ``document_id``.

        Stable across processes, restarts and machines for a fixed shard count.
        """
        digest = hashlib.blake2b(document_id.encode("utf-8"), digest_size=_DIGEST_SIZE).digest()
        return int.from_bytes(digest, "big") % self._shard_count
