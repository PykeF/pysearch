"""Structured logging built on the standard library.

Log records are emitted as one JSON object per line, which is what log
collectors expect and what makes logs queryable. Anything passed through the
standard ``extra={...}`` mechanism is merged into the JSON payload, so later
phases can attach node identifiers or request identifiers without changing the
formatter.
"""

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

# Attributes that ``logging`` puts on every record. Everything else on a record
# was supplied by the caller via ``extra={...}`` and belongs in the payload.
_STANDARD_RECORD_ATTRS = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)

# Uvicorn attaches an ANSI-coloured copy of the message to its records. It is
# meaningless once the message is already in the payload, and the escape codes
# corrupt structured output.
_SUPPRESSED_RECORD_ATTRS = frozenset({"color_message"})

# Uvicorn installs its own handlers and formatters at startup. Clearing them and
# letting the records propagate to the root logger keeps server logs in the same
# structured format as application logs.
_UVICORN_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access")


class JsonFormatter(logging.Formatter):
    """Render log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(_extra_fields(record))

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        # ``default=str`` keeps a stray non-serialisable value in ``extra`` from
        # turning a log call into an application error.
        return json.dumps(payload, default=str)


def _extra_fields(record: logging.LogRecord) -> dict[str, Any]:
    """Return the caller-supplied ``extra`` fields attached to a record."""
    excluded = _STANDARD_RECORD_ATTRS | _SUPPRESSED_RECORD_ATTRS
    return {key: value for key, value in record.__dict__.items() if key not in excluded}


def configure_logging(level: str) -> None:
    """Route all logging through a single JSON handler on stdout.

    This is idempotent: existing handlers are replaced, so calling it more than
    once (for instance when tests build several application instances) will not
    duplicate log output.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    for name in _UVICORN_LOGGERS:
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True
