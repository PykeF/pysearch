"""Reciprocal Rank Fusion.

Why not add the scores
----------------------

The obvious thing is wrong::

    hybrid = 0.5 * bm25 + 0.5 * cosine

BM25 is unbounded, depends on corpus statistics, and moves with the query — in
this project's own measurements the same document scored 0.3696 and then 0.4627
for the same query after one deletion changed ``df``. Cosine over unit vectors
lives in ``[-1, 1]`` and in practice occupies a much narrower band. Adding them
means the weight is not a weight: the lexical term dominates when the query has
rare terms and vanishes when it does not. And BM25 legitimately returns *no*
candidates for a paraphrase, so on exactly the queries semantic search exists to
answer, half the weighted sum would be contributing nothing at all.

Normalizing first does not rescue it. Min-max over the returned candidates is
candidate-set dependent: the best candidate maps to 1.0 whether it is an
excellent match or the least bad of a poor set, and an empty or single-element
list has no range to normalize over.

What RRF does instead
---------------------

Rank, not score::

    RRF(d) = sum over the lists containing d of  1 / (k + rank(d))

Ranks are 1-based. A document present in only one list contributes only that
term — there is deliberately **no penalty rank** for the list it is missing
from, because "BM25 found nothing" is a normal outcome here rather than evidence
against the document.

The honest cost is that magnitude is discarded. A document that beats the field
by a mile at rank 1 contributes exactly what a marginal rank 1 contributes. That
is the price of not having to reason about two incompatible scales, and it is
the reason a fused ranking can occasionally be worse than either input.

The parameter k
---------------

``k`` sets how quickly the contribution decays with rank. Small k makes the top
few positions dominate; large k flattens the list, so *appearing in both* comes
to matter more than *ranking highly in one*. With a candidate depth of 50, k=60
spans only a factor of 1.8 from first to last, while k=10 spans 5.4.

The default below was chosen by measuring on a fixed set of development queries
before the final evaluation was looked at; see scripts/evaluate_retrieval.py.
"""

from dataclasses import dataclass

from app.search.results import SearchResults

#: Chosen from the development-query sensitivity experiment, not from the
#: held-out evaluation set and not from convention.
DEFAULT_RRF_K = 60

#: Each retrieval path is asked for this multiple of the requested limit, so
#: fusion has candidates to work with beyond the ones that would be returned.
DEFAULT_CANDIDATE_MULTIPLIER = 5

#: Ceiling on candidate depth, matching the limit the internal APIs accept.
MAX_CANDIDATE_DEPTH = 100


@dataclass(frozen=True, slots=True)
class FusionConfig:
    """How the two rankings are combined.

    Deliberately a small immutable object rather than environment variables:
    there is no demonstrated need to retune these at runtime, and the evaluation
    script varies them directly.
    """

    rrf_k: int = DEFAULT_RRF_K
    candidate_multiplier: int = DEFAULT_CANDIDATE_MULTIPLIER
    max_candidate_depth: int = MAX_CANDIDATE_DEPTH

    def __post_init__(self) -> None:
        if self.rrf_k < 1:
            raise ValueError(f"rrf_k must be at least 1, got {self.rrf_k}")
        if self.candidate_multiplier < 1:
            raise ValueError(
                f"candidate_multiplier must be at least 1, got {self.candidate_multiplier}"
            )

    def candidate_depth(self, limit: int) -> int:
        """How many candidates to ask each retrieval path for."""
        return min(limit * self.candidate_multiplier, self.max_candidate_depth)


@dataclass(frozen=True, slots=True)
class HybridHit:
    """One fused result, with enough provenance to explain its position.

    ``score`` is a **fusion score**: a sum of reciprocal ranks. It is not a BM25
    relevance score, not a cosine similarity, not a probability and not a
    confidence. Values are small and only meaningful relative to each other
    within one response.

    The two ranks explain the score completely — a document at lexical rank 2
    and semantic rank 1 scores ``1/(k+2) + 1/(k+1)`` — which is why they are
    always present rather than hidden behind a debug flag. ``None`` means the
    document did not appear in that list at all.
    """

    document_id: str
    score: float
    text: str
    lexical_rank: int | None
    semantic_rank: int | None
    lexical_score: float | None
    semantic_score: float | None


@dataclass(frozen=True, slots=True)
class HybridResults:
    """A page of fused results.

    ``total`` is the size of the **candidate union** — how many distinct
    documents entered fusion, bounded by twice the candidate depth. It is not
    the corpus size, and not a count of relevant documents. The lexical and
    semantic endpoints each use ``total`` for something different again, which
    is why this one says so explicitly.
    """

    total: int
    results: tuple[HybridHit, ...]


@dataclass(slots=True)
class _Candidate:
    """Accumulates one document's contributions while the lists are walked."""

    document_id: str
    text: str
    lexical_rank: int | None = None
    semantic_rank: int | None = None
    lexical_score: float | None = None
    semantic_score: float | None = None
    score: float = 0.0


def reciprocal_rank_fusion(
    lexical: SearchResults,
    semantic: SearchResults,
    limit: int,
    config: FusionConfig | None = None,
) -> HybridResults:
    """Fuse a lexical and a semantic ranking into one.

    A document appearing in both lists appears **once** in the output, with its
    two contributions summed.

    Ordering is fusion score descending, then document id ascending — the same
    tie-break the lexical and semantic paths already use, so nothing depends on
    dictionary iteration order and no underlying score acts as a hidden
    tie-break.

    The result is the exact RRF ranking **of the two candidate lists given**. It
    is not necessarily the RRF ranking of the whole corpus: a document that fell
    outside both retrieval paths' candidate depth is absent here even though its
    fused score might have placed it. That is a real limitation of candidate
    truncation, and unlike the distributed top-k arguments elsewhere in this
    project it cannot be argued away, because fusion consumes ranks rather than
    comparable scores.

    Cost is O(L + S) to accumulate and O(U log U) to order, for list sizes L and
    S and a union of U <= L + S documents.
    """
    if limit < 1:
        raise ValueError(f"limit must be at least 1, got {limit}")
    settings = config if config is not None else FusionConfig()

    candidates: dict[str, _Candidate] = {}

    for rank, hit in enumerate(lexical.results, start=1):
        candidate = candidates.setdefault(
            hit.document_id, _Candidate(document_id=hit.document_id, text=hit.text)
        )
        candidate.lexical_rank = rank
        candidate.lexical_score = hit.score
        candidate.score += 1.0 / (settings.rrf_k + rank)

    for rank, hit in enumerate(semantic.results, start=1):
        candidate = candidates.setdefault(
            hit.document_id, _Candidate(document_id=hit.document_id, text=hit.text)
        )
        candidate.semantic_rank = rank
        candidate.semantic_score = hit.score
        candidate.score += 1.0 / (settings.rrf_k + rank)

    ordered = sorted(
        candidates.values(), key=lambda candidate: (-candidate.score, candidate.document_id)
    )

    return HybridResults(
        total=len(candidates),
        results=tuple(
            HybridHit(
                document_id=candidate.document_id,
                score=candidate.score,
                text=candidate.text,
                lexical_rank=candidate.lexical_rank,
                semantic_rank=candidate.semantic_rank,
                lexical_score=candidate.lexical_score,
                semantic_score=candidate.semantic_score,
            )
            for candidate in ordered[:limit]
        ),
    )
