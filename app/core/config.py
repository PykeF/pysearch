"""Application configuration, resolved from the environment.

Configuration is deliberately kept free of application logic: this module knows
how to *read* settings, and nothing about what they are used for.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["local", "test", "production"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
NodeRole = Literal["single", "shard", "coordinator"]


class Settings(BaseSettings):
    """Runtime configuration for a PySearch node.

    Values are resolved in order of decreasing precedence: arguments passed to
    the constructor, ``PYSEARCH_``-prefixed environment variables, a local
    ``.env`` file, and finally the defaults declared below.

    The role decides what a process is. ``single`` is a complete standalone
    search engine and is the default, so nothing about the distributed
    configuration has to be understood to run one node.
    """

    model_config = SettingsConfigDict(
        env_prefix="PYSEARCH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    app_name: str = "pysearch"
    environment: Environment = "local"
    log_level: LogLevel = "INFO"

    #: SQLite database holding the authoritative document corpus. Parent
    #: directories are created on startup if they do not exist. Every shard
    #: needs its own; two processes must never share one.
    storage_path: Path = Path("pysearch.db")

    node_role: NodeRole = "single"

    #: This node's shard, required when the role is ``shard``.
    shard_id: int | None = None

    #: How many shards the cluster has. Fixed for the life of a cluster:
    #: routing is modulo this number, so changing it moves nearly every
    #: document.
    shard_count: int = Field(default=1, ge=1)

    #: Shard base URLs as a comma-separated list, positionally indexed by shard
    #: id, required when the role is ``coordinator``. Kept as a string because
    #: pydantic-settings decodes structured fields as JSON, which would make an
    #: ordinary comma-separated environment variable a parse error.
    shard_urls: str = ""

    connect_timeout: float = Field(default=1.0, gt=0.0)
    request_timeout: float = Field(default=2.0, gt=0.0)

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalise_log_level(cls, value: object) -> object:
        """Accept log levels in any case, since ``LOG_LEVEL=debug`` is common."""
        return value.upper() if isinstance(value, str) else value

    @property
    def shard_addresses(self) -> tuple[str, ...]:
        """The configured shard URLs, indexed by shard id."""
        return tuple(url.strip() for url in self.shard_urls.split(",") if url.strip())

    @model_validator(mode="after")
    def _check_topology(self) -> "Settings":
        """Reject topologies that cannot work, at startup rather than in flight."""
        if self.node_role == "shard":
            if self.shard_id is None:
                raise ValueError("shard_id is required when node_role is 'shard'")
            if not 0 <= self.shard_id < self.shard_count:
                raise ValueError(
                    f"shard_id {self.shard_id} is outside the range of {self.shard_count} shards"
                )

        if self.node_role == "coordinator":
            addresses = self.shard_addresses
            if not addresses:
                raise ValueError("shard_urls is required when node_role is 'coordinator'")
            if len(addresses) != self.shard_count:
                raise ValueError(
                    f"shard_count is {self.shard_count} but {len(addresses)} shard urls were given"
                )
            if len(set(addresses)) != len(addresses):
                raise ValueError("shard_urls contains duplicate addresses")

        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, constructed once and then cached.

    Caching keeps environment parsing off the request path and gives the whole
    process a single consistent view of its configuration.
    """
    return Settings()
