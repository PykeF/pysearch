"""Application configuration, resolved from the environment.

Configuration is deliberately kept free of application logic: this module knows
how to *read* settings, and nothing about what they are used for.
"""

from functools import lru_cache
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["local", "test", "production"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseSettings):
    """Runtime configuration for the PySearch service.

    Values are resolved in order of decreasing precedence: arguments passed to
    the constructor, ``PYSEARCH_``-prefixed environment variables, a local
    ``.env`` file, and finally the defaults declared below.
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

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalise_log_level(cls, value: object) -> object:
        """Accept log levels in any case, since ``LOG_LEVEL=debug`` is common."""
        return value.upper() if isinstance(value, str) else value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, constructed once and then cached.

    Caching keeps environment parsing off the request path and gives the whole
    process a single consistent view of its configuration.
    """
    return Settings()
