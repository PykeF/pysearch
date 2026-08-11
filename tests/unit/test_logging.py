"""Tests for structured logging."""

import json
import logging
from typing import Any

import pytest

from app.core.logging import JsonFormatter, configure_logging


def _format(record: logging.LogRecord) -> dict[str, Any]:
    """Format a record and parse it back, proving the output is valid JSON."""
    parsed: dict[str, Any] = json.loads(JsonFormatter().format(record))
    return parsed


def _record(**kwargs: Any) -> logging.LogRecord:
    defaults: dict[str, Any] = {
        "name": "pysearch.test",
        "level": logging.INFO,
        "pathname": __file__,
        "lineno": 1,
        "msg": "hello %s",
        "args": ("world",),
        "exc_info": None,
    }
    return logging.LogRecord(**{**defaults, **kwargs})


def test_record_is_rendered_as_a_single_json_line() -> None:
    output = JsonFormatter().format(_record())

    assert "\n" not in output
    payload = json.loads(output)
    assert payload["level"] == "INFO"
    assert payload["logger"] == "pysearch.test"
    assert payload["message"] == "hello world"
    assert "timestamp" in payload


def test_extra_fields_are_merged_into_the_payload() -> None:
    record = _record()
    record.node_id = "node-1"  # type: ignore[attr-defined]
    record.shard = 3  # type: ignore[attr-defined]

    payload = _format(record)

    assert payload["node_id"] == "node-1"
    assert payload["shard"] == 3


def test_uvicorn_colour_copies_are_suppressed() -> None:
    record = _record()
    record.color_message = "hello \x1b[36m%s\x1b[0m"  # type: ignore[attr-defined]

    payload = _format(record)

    assert "color_message" not in payload
    assert payload["message"] == "hello world"


def test_non_serialisable_extras_do_not_break_logging() -> None:
    record = _record()
    record.value = object()  # type: ignore[attr-defined]

    payload = _format(record)

    assert isinstance(payload["value"], str)


def test_exceptions_are_included() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = _record(level=logging.ERROR, exc_info=sys.exc_info())

    payload = _format(record)

    assert "ValueError: boom" in payload["exception"]


def test_configure_logging_installs_exactly_one_json_handler() -> None:
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    original_level = root.level
    try:
        configure_logging("DEBUG")
        configure_logging("WARNING")

        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0].formatter, JsonFormatter)
        assert root.level == logging.WARNING
    finally:
        root.handlers[:] = original_handlers
        root.setLevel(original_level)


def test_configure_logging_rejects_an_unknown_level() -> None:
    with pytest.raises(ValueError):
        configure_logging("LOUD")
