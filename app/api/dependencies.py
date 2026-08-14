"""Shared API dependencies."""

from typing import Annotated

from fastapi import Depends, HTTPException, Request, status

from app.cluster.replication import ReplicaSynchronizer, Replicator
from app.core.config import Settings
from app.search.engine import SearchEngine


def get_engine(request: Request) -> SearchEngine:
    """Return the search engine attached to the running application.

    The engine is created once per application in ``create_app`` and stored on
    the application state, so a test that builds its own application gets its
    own empty engine.
    """
    engine: SearchEngine = request.app.state.engine
    return engine


EngineDep = Annotated[SearchEngine, Depends(get_engine)]


def get_settings_for(request: Request) -> Settings:
    """Return the settings this application was built with."""
    settings: Settings = request.app.state.settings
    return settings


SettingsDep = Annotated[Settings, Depends(get_settings_for)]


def require_primary(request: Request) -> Replicator:
    """Return the write path, refusing on a node that is not the primary.

    This is where split-brain is prevented structurally rather than by
    convention: a replica has no writer at all, so it cannot accept a
    coordinator write even if one were misrouted to it.
    """
    replicator: Replicator | None = getattr(request.app.state, "replicator", None)
    if replicator is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="this node is not the primary for its logical shard",
        )
    return replicator


PrimaryDep = Annotated[Replicator, Depends(require_primary)]


def require_replica(request: Request) -> ReplicaSynchronizer:
    """Return the synchronizer, refusing on a node that is not a replica.

    The mirror of :func:`require_primary`: a primary must never apply a
    mutation replicated from somewhere else, because it is the only node
    allowed to decide what its logical shard contains.
    """
    synchronizer: ReplicaSynchronizer | None = getattr(request.app.state, "synchronizer", None)
    if synchronizer is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="this node is not a replica",
        )
    return synchronizer


ReplicaDep = Annotated[ReplicaSynchronizer, Depends(require_replica)]
