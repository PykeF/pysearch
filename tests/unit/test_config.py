"""Tests for environment-driven configuration."""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.config import Settings, get_settings


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Remove any ambient PYSEARCH_* variables so results are deterministic."""
    for key in list(os.environ):
        if key.startswith("PYSEARCH_"):
            monkeypatch.delenv(key)
    yield


def _build() -> Settings:
    """Build settings from the environment only, ignoring any local .env file."""
    return Settings(_env_file=None)


def test_defaults_apply_when_environment_is_empty() -> None:
    settings = _build()

    assert settings.app_name == "pysearch"
    assert settings.environment == "local"
    assert settings.log_level == "INFO"
    assert settings.storage_path == Path("pysearch.db")


def test_storage_path_is_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYSEARCH_STORAGE_PATH", "/var/lib/pysearch/corpus.db")

    assert _build().storage_path == Path("/var/lib/pysearch/corpus.db")


def test_environment_variables_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYSEARCH_APP_NAME", "pysearch-node-1")
    monkeypatch.setenv("PYSEARCH_ENVIRONMENT", "production")
    monkeypatch.setenv("PYSEARCH_LOG_LEVEL", "ERROR")

    settings = _build()

    assert settings.app_name == "pysearch-node-1"
    assert settings.environment == "production"
    assert settings.log_level == "ERROR"


def test_log_level_is_case_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYSEARCH_LOG_LEVEL", "debug")

    assert _build().log_level == "DEBUG"


def test_invalid_log_level_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYSEARCH_LOG_LEVEL", "LOUD")

    with pytest.raises(ValidationError):
        _build()


def test_invalid_environment_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYSEARCH_ENVIRONMENT", "staging")

    with pytest.raises(ValidationError):
        _build()


def test_settings_are_immutable() -> None:
    settings = _build()

    with pytest.raises(ValidationError):
        settings.log_level = "DEBUG"  # type: ignore[misc]


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()
    try:
        assert get_settings() is get_settings()
    finally:
        get_settings.cache_clear()


# ----------------------------------------------------------------------
# Cluster topology
# ----------------------------------------------------------------------


def test_the_default_role_is_a_standalone_node() -> None:
    settings = _build()

    assert settings.node_role == "single"
    assert settings.shard_count == 1
    assert settings.shard_addresses == ()


def test_shard_urls_are_parsed_from_a_comma_separated_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYSEARCH_NODE_ROLE", "coordinator")
    monkeypatch.setenv("PYSEARCH_SHARD_COUNT", "3")
    monkeypatch.setenv("PYSEARCH_SHARD_URLS", "http://a:8000, http://b:8000 ,http://c:8000")

    assert _build().shard_addresses == ("http://a:8000", "http://b:8000", "http://c:8000")


def test_a_shard_requires_an_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYSEARCH_NODE_ROLE", "shard")

    with pytest.raises(ValidationError, match="shard_id is required"):
        _build()


def test_a_shard_id_outside_the_shard_count_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYSEARCH_NODE_ROLE", "shard")
    monkeypatch.setenv("PYSEARCH_SHARD_COUNT", "3")
    monkeypatch.setenv("PYSEARCH_SHARD_ID", "3")

    with pytest.raises(ValidationError, match="outside the range"):
        _build()


def test_a_valid_shard_configuration_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYSEARCH_NODE_ROLE", "shard")
    monkeypatch.setenv("PYSEARCH_SHARD_COUNT", "3")
    monkeypatch.setenv("PYSEARCH_SHARD_ID", "2")

    settings = _build()

    assert settings.shard_id == 2


def test_a_coordinator_requires_shard_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYSEARCH_NODE_ROLE", "coordinator")

    with pytest.raises(ValidationError, match="shard_urls is required"):
        _build()


def test_the_url_count_must_match_the_shard_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYSEARCH_NODE_ROLE", "coordinator")
    monkeypatch.setenv("PYSEARCH_SHARD_COUNT", "3")
    monkeypatch.setenv("PYSEARCH_SHARD_URLS", "http://a:8000,http://b:8000")

    with pytest.raises(ValidationError, match="2 shard urls"):
        _build()


def test_duplicate_shard_urls_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    # Two entries pointing at one node would mean a shard silently owning two
    # slots of the keyspace while another owns none.
    monkeypatch.setenv("PYSEARCH_NODE_ROLE", "coordinator")
    monkeypatch.setenv("PYSEARCH_SHARD_COUNT", "2")
    monkeypatch.setenv("PYSEARCH_SHARD_URLS", "http://a:8000,http://a:8000")

    with pytest.raises(ValidationError, match="duplicate"):
        _build()


def test_a_zero_shard_count_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYSEARCH_SHARD_COUNT", "0")

    with pytest.raises(ValidationError):
        _build()


def test_timeouts_have_defaults_and_are_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _build().request_timeout == 2.0
    assert _build().connect_timeout == 1.0

    monkeypatch.setenv("PYSEARCH_REQUEST_TIMEOUT", "0.25")
    assert _build().request_timeout == 0.25


def test_a_non_positive_timeout_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYSEARCH_REQUEST_TIMEOUT", "0")

    with pytest.raises(ValidationError):
        _build()
