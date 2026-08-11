"""Shared test fixtures."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


@pytest.fixture
def settings() -> Settings:
    """Fully explicit settings, so tests never depend on the ambient environment."""
    return Settings(app_name="pysearch-test", environment="test", log_level="WARNING")


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    """An HTTP client bound to an isolated application with its own empty index."""
    with TestClient(create_app(settings)) as test_client:
        yield test_client
