"""Liveness and readiness endpoints."""

from typing import Literal

from fastapi import APIRouter, Response, status
from pydantic import BaseModel, Field

from app.api.dependencies import EngineDep

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Body returned by the health endpoint."""

    status: Literal["ok"] = "ok"


class ReadinessResponse(BaseModel):
    """Body returned by the readiness endpoint."""

    status: Literal["ready", "not_ready"]
    detail: str = Field(description="Why the service is not ready, when it is not.")


@router.get("/health", summary="Report process liveness")
def health() -> HealthResponse:
    """Report that the process is running.

    Deliberately checks nothing else: a liveness probe answers "should this
    process be restarted", and reporting failure here because storage is
    momentarily unavailable would restart a perfectly healthy process. Whether
    the service can actually serve is what ``/ready`` is for.
    """
    return HealthResponse()


@router.get(
    "/ready",
    summary="Report whether the service can serve requests",
    responses={
        status.HTTP_200_OK: {"description": "Storage is open and derived state is trusted."},
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "The engine is uninitialized or degraded."
        },
    },
)
def ready(engine: EngineDep, response: Response) -> ReadinessResponse:
    """Report readiness: storage open, index rebuilt, derived state trusted.

    Returns 503 while the engine is degraded, which is the state it enters if a
    durable write commits but the derived in-memory structures fail to follow.
    """
    engine_status = engine.status()
    if not engine_status.ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadinessResponse(status="not_ready", detail=engine_status.detail)
    return ReadinessResponse(status="ready", detail=engine_status.detail)
