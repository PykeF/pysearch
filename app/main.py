"""Application entry point and composition root.

One application, three roles. ``single`` is a complete standalone search engine;
``shard`` owns a slice of the corpus and answers only internal calls; and
``coordinator`` owns no documents and routes, fans out and merges.

Roles decide which routers a process exposes, and a shard deliberately does not
expose a public ``/search``. Querying one shard directly would return silently
partial results scored against only that shard's statistics, and the cheapest
way to prevent that mistake is not to offer the path.
"""

import logging
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager

import httpx2
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.api.cluster import router as cluster_router
from app.api.documents import router as documents_router
from app.api.health import readiness_router
from app.api.health import router as health_router
from app.api.index import router as index_router
from app.api.internal import router as internal_router
from app.api.search import router as search_router
from app.cluster.client import HttpShardClient, ShardClient
from app.cluster.coordinator import Coordinator
from app.cluster.errors import ClusterError
from app.cluster.routing import ShardRouter
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


def _handle_cluster_error(request: Request, exc: Exception) -> JSONResponse:
    """Translate a shard failure into 503.

    Never a partial success: if a shard could not take part, the results would
    be both incomplete and — because scoring depends on cluster-wide statistics
    — computed from the wrong corpus. Saying so is the honest answer.
    """
    logger.error("distributed operation failed", exc_info=exc)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": str(exc)},
    )


def _register_error_handlers(app: FastAPI) -> None:
    """Map transport-agnostic errors onto status codes in one place."""
    app.add_exception_handler(DocumentNotFoundError, _handle_document_not_found)
    app.add_exception_handler(InvalidDocumentError, _handle_invalid_document)
    app.add_exception_handler(EngineNotReadyError, _handle_engine_not_ready)
    app.add_exception_handler(StorageError, _handle_storage_error)
    app.add_exception_handler(ClusterError, _handle_cluster_error)


def _node_lifespan(
    settings: Settings,
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    """Build the startup/shutdown hook for a node that owns documents.

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


def _coordinator_lifespan(
    settings: Settings, shard_clients: Sequence[ShardClient] | None
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    """Build the startup/shutdown hook for a coordinator.

    A coordinator opens no database. It holds one pooled HTTP client for the
    whole cluster and the topology, which is why restarting it loses nothing and
    routing resolves identically afterwards.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        http: httpx2.AsyncClient | None = None
        clients = shard_clients

        if clients is None:
            http = httpx2.AsyncClient(
                timeout=httpx2.Timeout(settings.request_timeout, connect=settings.connect_timeout)
            )
            clients = [HttpShardClient(url, http) for url in settings.shard_addresses]

        app.state.coordinator = Coordinator(ShardRouter(settings.shard_count), clients)
        logger.info(
            "coordinator ready",
            extra={"shard_count": settings.shard_count, "shards": list(settings.shard_addresses)},
        )
        try:
            yield
        finally:
            if http is not None:
                await http.aclose()
            logger.info("coordinator stopped")

    return lifespan


def create_app(
    settings: Settings | None = None,
    shard_clients: Sequence[ShardClient] | None = None,
) -> FastAPI:
    """Build a configured FastAPI application for this node's role.

    Settings are passed in rather than imported at module scope so that tests
    can build an isolated application without touching process-wide state. When
    omitted, the cached process settings are used.

    ``shard_clients`` overrides the coordinator's transport, which is what lets
    distributed tests wire a coordinator to in-process shard applications
    instead of sockets. It is the same kind of injection as ``settings``, and it
    is ignored by every role but ``coordinator``.
    """
    settings = settings or get_settings()

    context: dict[str, object] = {"node_role": settings.node_role}
    if settings.shard_id is not None:
        context["shard_id"] = settings.shard_id
    configure_logging(settings.log_level, context)

    if settings.node_role == "coordinator":
        app = FastAPI(
            title=settings.app_name,
            lifespan=_coordinator_lifespan(settings, shard_clients),
        )
        app.include_router(health_router)
        app.include_router(cluster_router)
    else:
        app = FastAPI(title=settings.app_name, lifespan=_node_lifespan(settings))
        app.include_router(health_router)
        app.include_router(readiness_router)
        if settings.node_role == "shard":
            app.include_router(internal_router)
        else:
            app.include_router(documents_router)
            app.include_router(search_router)
            app.include_router(index_router)

    _register_error_handlers(app)

    logger.info(
        "application configured",
        extra={"environment": settings.environment, "log_level": settings.log_level},
    )
    return app


app = create_app()
