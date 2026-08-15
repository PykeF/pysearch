"""The shape of a ranked result.

These live apart from the engine because more than one thing produces and
consumes them: the engine ranks, the shard client deserialises, the coordinator
merges, and rank fusion combines. Keeping them here lets fusion depend on the
result shape without depending on the engine that happens to produce it, which
is the right direction — the engine composes fusion, not the other way round.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SearchResult:
    """One ranked hit."""

    document_id: str
    score: float
    text: str


@dataclass(frozen=True, slots=True)
class SearchResults:
    """A page of ranked hits, plus how many documents matched in total."""

    total: int
    results: tuple[SearchResult, ...]
