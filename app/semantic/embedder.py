"""Turning text into vectors.

The model is behind a protocol for one concrete reason: almost every test in the
suite needs the semantic path without loading a model or touching the network,
and the production implementation needs to be swappable if measurement says a
different model is worth its weight.

Normalization policy
--------------------

Every vector this module produces is L2-normalized. That is a deliberate choice
made in exactly one place, because it collapses the similarity metric: for unit
vectors, cosine similarity *is* the dot product, so search is one matrix
multiply and the index has a single invariant to check. Mixing normalized and
unnormalized vectors would silently corrupt ranking, so normalization happens
here rather than being left to callers.

Semantic identity
-----------------

Two copies of a logical shard can only serve interchangeable semantic results if
they embed with the same thing. "The same thing" is more than a model name: a
repository can move while keeping its name, so the identity pins a revision, and
it also carries the dimension, the normalization policy and which implementation
produced the vectors. Those five fields must match exactly.

What is deliberately *not* claimed is bit-for-bit identity of the vectors
themselves. Library versions, BLAS implementations and floating-point reduction
order can all move the last bits. The guarantee is that copies sharing an
identity produce numerically equivalent embeddings, and therefore identical
ordering and identical tie-breaking, with scores equal to within a strict
tolerance.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

from app.semantic.errors import EmbeddingError

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    from numpy.typing import NDArray

#: The model this project embeds with, pinned to an exact commit. Loading a
#: repository's default branch would mean the corpus could be re-embedded into a
#: different space after an upstream push, with nothing in the system noticing.
DEFAULT_MODEL_ID = "minishlab/potion-base-8M"
DEFAULT_MODEL_REVISION = "bf8b056651a2c21b8d2565580b8569da283cab23"

#: The only normalization policy in use; part of the semantic identity so that a
#: future change to it cannot be mistaken for compatible state.
L2_NORMALIZATION = "l2"


@dataclass(frozen=True, slots=True)
class SemanticIdentity:
    """Everything that must match for two copies' vectors to be comparable."""

    implementation: str
    model_id: str
    model_revision: str
    dimension: int
    normalization: str

    @property
    def fingerprint(self) -> str:
        """A stable string form, for configuration checks and status reporting."""
        return (
            f"{self.implementation}:{self.model_id}@{self.model_revision}"
            f"/d{self.dimension}/{self.normalization}"
        )

    def is_compatible_with(self, other: "SemanticIdentity") -> bool:
        """Whether vectors from the two identities may be compared at all."""
        return self == other


class Embedder(Protocol):
    """Turns text into unit-length vectors."""

    @property
    def identity(self) -> SemanticIdentity:
        """What produced these vectors, and in what space."""
        ...

    def embed_documents(self, texts: Sequence[str]) -> "NDArray[np.float32]":
        """Embed a batch of documents, returning one unit row per text."""
        ...

    def embed_query(self, text: str) -> "NDArray[np.float32]":
        """Embed a single query into one unit vector."""
        ...


def normalize_rows(vectors: "NDArray[np.float32]") -> "NDArray[np.float32]":
    """Scale each row to unit length.

    A zero row cannot be normalized — it has no direction — so it is left at
    zero rather than divided by zero. That happens for text that embeds to
    nothing, and such a document simply never resembles any query, which is the
    honest outcome.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe = np.where(norms == 0.0, 1.0, norms)
    return np.asarray(vectors / safe, dtype=np.float32)


def validate_vectors(vectors: "NDArray[np.float32]", dimension: int) -> None:
    """Reject anything the vector index must never be asked to hold.

    Raises:
        EmbeddingError: on the wrong shape, wrong width, or non-finite values.
    """
    if vectors.ndim != 2:
        raise EmbeddingError(f"expected a 2-D array of vectors, got {vectors.ndim} dimensions")
    if vectors.shape[1] != dimension:
        raise EmbeddingError(f"expected vectors of dimension {dimension}, got {vectors.shape[1]}")
    if not bool(np.isfinite(vectors).all()):
        raise EmbeddingError("embedding produced non-finite values")


class Model2VecEmbedder:
    """Embeds with a model2vec static model.

    Static means every token has a fixed vector and a document embedding is the
    mean of its token vectors — there is no neural network at inference time.
    That is why it runs in milliseconds on a CPU, needs no torch, and behaves
    deterministically, all of which matter more here than the last few points of
    benchmark quality. The protocol above is what keeps that a revisable
    decision.
    """

    IMPLEMENTATION = "model2vec"

    def __init__(self, model: object, identity: SemanticIdentity) -> None:
        self._model = model
        self._identity = identity

    @classmethod
    def load(
        cls,
        model_id: str = DEFAULT_MODEL_ID,
        revision: str = DEFAULT_MODEL_REVISION,
    ) -> "Model2VecEmbedder":
        """Load the pinned revision, downloading it once if it is not cached.

        The revision is fetched explicitly and the model is then loaded from that
        exact snapshot, because the loader itself has no revision parameter and
        would otherwise take whatever the default branch currently holds.

        Raises:
            EmbeddingError: if the model cannot be fetched or loaded.
        """
        try:
            from huggingface_hub import snapshot_download
            from model2vec import StaticModel
        except ImportError as error:  # pragma: no cover - depends on the install
            raise EmbeddingError(
                "semantic search is enabled but its dependencies are not installed; "
                "install the 'semantic' extra"
            ) from error

        try:
            snapshot = snapshot_download(model_id, revision=revision)
            model = StaticModel.from_pretrained(snapshot)
        except Exception as error:
            raise EmbeddingError(f"could not load embedding model {model_id}@{revision}") from error

        dimension = int(model.dim)
        return cls(
            model,
            SemanticIdentity(
                implementation=cls.IMPLEMENTATION,
                model_id=model_id,
                model_revision=revision,
                dimension=dimension,
                normalization=L2_NORMALIZATION,
            ),
        )

    @property
    def identity(self) -> SemanticIdentity:
        return self._identity

    def embed_documents(self, texts: Sequence[str]) -> "NDArray[np.float32]":
        """Embed a batch of documents, returning one unit row per text."""
        if not texts:
            return np.zeros((0, self._identity.dimension), dtype=np.float32)
        return self._encode(list(texts))

    def embed_query(self, text: str) -> "NDArray[np.float32]":
        """Embed a single query into one unit vector."""
        return np.asarray(self._encode([text])[0], dtype=np.float32)

    def _encode(self, texts: list[str]) -> "NDArray[np.float32]":
        """Run the model and enforce this module's normalization policy."""
        try:
            raw = np.asarray(self._model.encode(texts), dtype=np.float32)  # type: ignore[attr-defined]
        except Exception as error:
            raise EmbeddingError("embedding failed") from error

        validate_vectors(raw, self._identity.dimension)
        return normalize_rows(raw)
