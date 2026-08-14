"""Errors raised by the semantic path.

Library exceptions — from the model, the hub or NumPy — are wrapped here, so no
transformer or array internals reach a client or the coordinator.
"""


class SemanticError(Exception):
    """Base class for semantic-retrieval failures."""


class SemanticDisabledError(SemanticError):
    """Semantic search was requested on a node that does not have it enabled."""


class EmbeddingError(SemanticError):
    """Text could not be embedded, or the resulting vector was unusable.

    Raised before any durable mutation, so a failure here leaves the corpus and
    every derived structure untouched.
    """


class SemanticIdentityMismatchError(SemanticError):
    """Two nodes disagree about which embedding model they are using.

    Vectors from different models measure different spaces, so comparing them
    produces numbers that look like similarities and mean nothing.
    """
