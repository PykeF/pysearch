"""Health endpoint."""

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Body returned by the health endpoint."""

    status: Literal["ok"] = "ok"


@router.get("/health", summary="Report service health")
def health() -> HealthResponse:
    """Report that the service is running and able to serve requests.

    Phase 0 has no storage, indexes or peer nodes, so this is deliberately a
    pure liveness check rather than a readiness check over dependencies that do
    not exist yet.
    """
    return HealthResponse()
