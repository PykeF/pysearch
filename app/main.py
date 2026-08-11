"""Application entry point and composition root."""

import logging

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.api.documents import router as documents_router
from app.api.health import router as health_router
from app.api.index import router as index_router
from app.api.search import router as search_router
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.search.engine import SearchEngine
from app.search.errors import DocumentNotFoundError, InvalidDocumentError

logger = logging.getLogger(__name__)


def _handle_document_not_found(request: Request, exc: Exception) -> JSONResponse:
    """Translate a missing document into 404."""
    return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"detail": str(exc)})


def _handle_invalid_document(request: Request, exc: Exception) -> JSONResponse:
    """Translate a malformed document into 400."""
    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"detail": str(exc)})


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build a configured FastAPI application.

    Settings are passed in rather than imported at module scope so that tests
    can build an isolated application without touching process-wide state. When
    omitted, the cached process settings are used.

    Each application owns one search engine, held on the application state. The
    engine is in-memory, so its contents live and die with the process.
    """
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(title=settings.app_name)
    app.state.engine = SearchEngine()

    app.include_router(health_router)
    app.include_router(documents_router)
    app.include_router(search_router)
    app.include_router(index_router)

    # The search core raises transport-agnostic errors; mapping them to status
    # codes happens here, so no HTTP concern leaks into the retrieval code.
    app.add_exception_handler(DocumentNotFoundError, _handle_document_not_found)
    app.add_exception_handler(InvalidDocumentError, _handle_invalid_document)

    logger.info(
        "application configured",
        extra={"environment": settings.environment, "log_level": settings.log_level},
    )
    return app


app = create_app()
