"""Shared API dependencies."""

from typing import Annotated

from fastapi import Depends, Request

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
