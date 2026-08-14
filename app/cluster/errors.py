"""Errors raised by the distributed layer.

Transport exceptions are translated here so that no HTTP-client type escapes
into coordinator logic or into a public response.
"""


class ClusterError(Exception):
    """Base class for distributed failures."""


class ShardUnavailableError(ClusterError):
    """A shard could not be reached, or answered with a failure."""


class ShardTimeoutError(ShardUnavailableError):
    """A shard did not answer within the configured timeout.

    A kind of unavailability rather than a separate outcome: from the
    coordinator's position, a shard that has not answered in time and a shard
    that refused the connection are the same thing.
    """


class DistributedSearchError(ClusterError):
    """One or more shards failed to take part in a distributed operation.

    Carries the shard identifiers so the failure can be reported without
    guessing which node caused it.
    """

    def __init__(self, message: str, shard_ids: tuple[int, ...]) -> None:
        super().__init__(message)
        self.shard_ids = shard_ids


class ReplicationError(ClusterError):
    """A replica failed to apply a mutation the primary had already committed.

    The write is durable on the primary and is deliberately not rolled back —
    storage is authoritative and a compensating delete could fail too. It is
    simply not acknowledged, so the client is told the write failed even though
    it may be present on the primary.
    """


class ShardCopiesExhaustedError(ClusterError):
    """Every copy of a logical shard failed, so the corpus is incomplete."""

    def __init__(self, message: str, shard_id: int) -> None:
        super().__init__(message)
        self.shard_id = shard_id
