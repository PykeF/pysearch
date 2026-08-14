"""The coordinator's view of where each logical shard lives.

A logical shard is a deterministic subset of documents. A physical node is one
process with one database. Phase 3 collapsed the two; keeping them apart is what
lets a shard survive losing a node.

Routing keys on the **logical** shard and never on a physical node, so adding or
removing replicas cannot move a single document.
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from app.cluster.client import ShardClient


@dataclass(frozen=True, slots=True)
class ShardCopies:
    """Every physical copy of one logical shard.

    ``serving_order`` puts the primary first, so a healthy cluster always reads
    from the most advanced copy — the primary is the only writer, so it can
    never be behind a replica.
    """

    shard_id: int
    primary: ShardClient
    replicas: tuple[ShardClient, ...]

    @property
    def serving_order(self) -> tuple[ShardClient, ...]:
        """Copies to try for reads, best first."""
        return (self.primary, *self.replicas)

    @property
    def replication_factor(self) -> int:
        """How many physical copies this logical shard has."""
        return 1 + len(self.replicas)


@dataclass(frozen=True, slots=True)
class ClusterTopology:
    """The logical shards of a cluster and the nodes holding them."""

    shards: tuple[ShardCopies, ...]

    def __post_init__(self) -> None:
        for position, shard in enumerate(self.shards):
            if shard.shard_id != position:
                raise ValueError(f"shard at position {position} reports id {shard.shard_id}")

    @property
    def shard_count(self) -> int:
        """The number of logical shards, which is what routing divides by."""
        return len(self.shards)

    @property
    def replication_factor(self) -> int:
        """Copies per logical shard, or 1 when no replicas are configured."""
        if not self.shards:
            return 1
        return self.shards[0].replication_factor

    def copies_for(self, shard_id: int) -> ShardCopies:
        """Return every copy of one logical shard."""
        return self.shards[shard_id]

    def primary_for(self, shard_id: int) -> ShardClient:
        """Return the only node permitted to accept writes for a logical shard."""
        return self.shards[shard_id].primary

    def __iter__(self) -> Iterator[ShardCopies]:
        return iter(self.shards)


def build_topology(
    primaries: Sequence[ShardClient], replicas: Sequence[Sequence[ShardClient]]
) -> ClusterTopology:
    """Assemble a topology from per-shard primaries and replica groups."""
    if replicas and len(replicas) != len(primaries):
        raise ValueError(
            f"{len(primaries)} primaries but {len(replicas)} replica groups were given"
        )

    return ClusterTopology(
        shards=tuple(
            ShardCopies(
                shard_id=shard_id,
                primary=primary,
                replicas=tuple(replicas[shard_id]) if replicas else (),
            )
            for shard_id, primary in enumerate(primaries)
        )
    )
