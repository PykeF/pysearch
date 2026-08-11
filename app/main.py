"""Application entry point and composition root."""

import logging

from fastapi import FastAPI

from app.api.health import router as health_router
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build a configured FastAPI application.

    Settings are passed in rather than imported at module scope so that tests
    can build an isolated application without touching process-wide state. When
    omitted, the cached process settings are used.
    """
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(title=settings.app_name)
    app.include_router(health_router)

    logger.info(
        "application configured",
        extra={"environment": settings.environment, "log_level": settings.log_level},
    )
    return app


app = create_app()
