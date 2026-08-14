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
