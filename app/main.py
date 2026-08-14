"""Application entry point and composition root."""

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.api.documents import router as documents_router
from app.api.health import router as health_router
from app.api.index import router as index_router
from app.api.search import router as search_router
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.search.engine import SearchEngine
from app.search.errors import DocumentNotFoundError, EngineNotReadyError, InvalidDocumentError
from app.storage.errors import StorageError
from app.storage.sqlite_store import SqliteDocumentStore

logger = logging.getLogger(__name__)


def _handle_document_not_found(request: Request, exc: Exception) -> JSONResponse:
    """Translate a missing document into 404."""
    return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"detail": str(exc)})


def _handle_invalid_document(request: Request, exc: Exception) -> JSONResponse:
    """Translate a malformed document into 400."""
    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"detail": str(exc)})


def _handle_engine_not_ready(request: Request, exc: Exception) -> JSONResponse:
    """Translate an uninitialized or degraded engine into 503.

    Logged with the traceback because when this carries a cause, that cause is
    the underlying failure that degraded the engine.
    """
    logger.error("refusing request: engine not ready", exc_info=exc)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": str(exc)},
    )


def _handle_storage_error(request: Request, exc: Exception) -> JSONResponse:
    """Translate a storage failure into 503, without leaking database details."""
    # The cause may carry SQL or file paths, so it is logged rather than returned.
    logger.error("storage operation failed", exc_info=exc)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": "storage unavailable"},
    )


def _build_lifespan(
    settings: Settings,
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    """Build the startup/shutdown hook that owns storage and recovery.

    Recovery happens here rather than lazily on first use, so the application
    cannot serve a request against a half-built index. A failure to open storage
    or rebuild aborts startup instead of serving a broken service.

    The engine's lifecycle methods are synchronous and know nothing about
    FastAPI; this hook only orchestrates them. They block the event loop while
    they run, which is what we want during startup — there is nothing else to
    serve yet.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        store = SqliteDocumentStore.open(settings.storage_path)
        engine = SearchEngine(store)
        report = engine.initialize()
        app.state.engine = engine

        logger.info(
            "index rebuilt from storage",
            extra={
                "storage_path": str(settings.storage_path),
                "documents": report.document_count,
                "duration_ms": round(report.duration_seconds * 1000, 3),
            },
        )
        try:
            yield
        finally:
            engine.close()
            logger.info("storage closed")

    return lifespan


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build a configured FastAPI application.

    Settings are passed in rather than imported at module scope so that tests
    can build an isolated application without touching process-wide state. When
    omitted, the cached process settings are used.

    The search engine is created during startup, not here, because it owns a
    database connection and rebuilding its state is real work with a real
    failure mode.
    """
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(title=settings.app_name, lifespan=_build_lifespan(settings))

    app.include_router(health_router)
    app.include_router(documents_router)
    app.include_router(search_router)
    app.include_router(index_router)

    # The core raises transport-agnostic errors; mapping them to status codes
    # happens here, so no HTTP concern leaks into retrieval or storage code.
    app.add_exception_handler(DocumentNotFoundError, _handle_document_not_found)
    app.add_exception_handler(InvalidDocumentError, _handle_invalid_document)
    app.add_exception_handler(EngineNotReadyError, _handle_engine_not_ready)
    app.add_exception_handler(StorageError, _handle_storage_error)

    logger.info(
        "application configured",
        extra={"environment": settings.environment, "log_level": settings.log_level},
    )
    return app


app = create_app()
