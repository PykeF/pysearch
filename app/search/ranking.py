"""BM25 ranking.

The score of a document ``D`` for a query ``Q`` is a sum over the query terms::

                              tf(q,D) * (k1 + 1)
    score(D,Q) = sum  idf(q) * ---------------------------------------
                 q in Q        tf(q,D) + k1 * (1 - b + b * dl(D)/avgdl)

with

    idf(q) = ln(1 + (N - df(q) + 0.5) / (df(q) + 0.5))

Two parts do the work. The ``tf`` fraction saturates: the tenth occurrence of a
term adds far less than the second, and ``k1`` controls how quickly that
happens. The ``dl/avgdl`` factor penalises long documents, on the reasoning
that a term appearing once in a 20-word document is stronger evidence than the
same term appearing once in a 2000-word one; ``b`` controls how much that
matters, from ``b=0`` (ignore length) to ``b=1`` (fully normalise).

The IDF above is the variant used by Lucene rather than the classic
``ln((N - df + 0.5) / (df + 0.5))``. The classic form goes negative once a term
appears in more than half the corpus, which means a common term can *reduce* a
document's score below a document that lacks it entirely. Adding one inside the
logarithm keeps the result positive for every ``df``, so scores stay monotone in
the evidence, and it removes the division-by-zero and domain errors that the
``+0.5`` smoothing exists to avoid.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

from app.search.index import InvertedIndex


@dataclass(frozen=True, slots=True)
class BM25Params:
    """The free parameters of BM25.

    The defaults are the standard starting point from the literature. They are
    named and passed around rather than written into the scoring expression so
    that tuning them later is a change in one place.
    """

    k1: float = 1.2
    b: float = 0.75


DEFAULT_BM25_PARAMS = BM25Params()


class BM25Scorer:
    """Scores documents against a query using BM25."""

    def __init__(self, params: BM25Params = DEFAULT_BM25_PARAMS) -> None:
        self._params = params

    @property
    def params(self) -> BM25Params:
        """The parameters this scorer was built with."""
        return self._params

    def inverse_document_frequency(self, document_frequency: int, document_count: int) -> float:
        """Return ``idf`` for a term appearing in ``document_frequency`` documents."""
        numerator = document_count - document_frequency + 0.5
        denominator = document_frequency + 0.5
        return math.log(1.0 + numerator / denominator)

    def score_query(self, index: InvertedIndex, terms: Sequence[str]) -> dict[str, float]:
        """Score every document containing at least one query term.

        Only the posting lists of the query terms are visited, so the cost is
        proportional to how many documents contain those terms, not to the size
        of the corpus.

        A term repeated in the query is scored once per occurrence, so repeating
        it emphasises it. Unknown terms have no posting list and contribute
        nothing.

        Iteration follows the given term order, so each document's score is
        accumulated in a fixed sequence and floating-point addition stays
        reproducible run to run.
        """
        scores: dict[str, float] = {}

        document_count = index.document_count
        average_length = index.average_document_length
        # An empty corpus, or one holding only empty documents, has nothing to
        # score — and guarding here keeps avgdl out of the denominator below.
        if document_count == 0 or average_length == 0.0:
            return scores

        k1 = self._params.k1
        b = self._params.b

        for term in terms:
            postings = index.posting_map(term)
            if not postings:
                continue

            idf = self.inverse_document_frequency(len(postings), document_count)

            for document_id, term_frequency in postings.items():
                length_ratio = index.document_length(document_id) / average_length
                saturation = term_frequency + k1 * (1.0 - b + b * length_ratio)
                contribution = idf * (term_frequency * (k1 + 1.0)) / saturation
                scores[document_id] = scores.get(document_id, 0.0) + contribution

        return scores
