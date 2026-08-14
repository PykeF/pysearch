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
