"""Errors raised by the search core.

These are transport-agnostic: the core signals what went wrong, and the HTTP
layer decides which status code that corresponds to.
"""


class SearchError(Exception):
    """Base class for every error raised by the search core."""


class InvalidDocumentError(SearchError):
    """A document was rejected because it is not well formed."""


class DocumentNotFoundError(SearchError):
    """An operation referenced a document that is not indexed."""


class IndexInvariantError(SearchError):
    """The index's internal structures disagree with one another.

    Raised only by the explicit invariant checks, which are a testing and
    debugging aid rather than part of the request path.
    """
