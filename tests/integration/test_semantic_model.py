"""Tests that load the real embedding model.

Excluded from the default run, because the rest of the suite must never
download a model or reach the network. Run them with::

    uv run --extra semantic pytest -m semantic_model

Everything here is deliberately small: the model's quality is measured by
scripts/evaluate_retrieval.py, and these only check that the thing we ship
loads, behaves, and produces the vectors the rest of the system assumes.
"""

import numpy as np
import pytest

from app.search.document import Document
from app.search.engine import SearchEngine
from app.semantic.embedder import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    Model2VecEmbedder,
)
from tests.conftest import InMemoryDocumentStore

pytestmark = pytest.mark.semantic_model


@pytest.fixture(scope="module")
def real_embedder() -> Model2VecEmbedder:
    """Loaded once for the module: the model is the expensive part."""
    return Model2VecEmbedder.load()


def test_the_pinned_model_loads(real_embedder: Model2VecEmbedder) -> None:
    identity = real_embedder.identity

    assert identity.model_id == DEFAULT_MODEL_ID
    assert identity.model_revision == DEFAULT_MODEL_REVISION
    assert identity.implementation == "model2vec"
    assert identity.normalization == "l2"
    assert identity.dimension == 256


def test_embeddings_have_the_expected_shape_and_are_finite(
    real_embedder: Model2VecEmbedder,
) -> None:
    vectors = real_embedder.embed_documents(["car maintenance", "search ranking"])

    assert vectors.shape == (2, real_embedder.identity.dimension)
    assert vectors.dtype == np.float32
    assert bool(np.isfinite(vectors).all())


def test_embeddings_are_unit_length(real_embedder: Model2VecEmbedder) -> None:
    """The index relies on this: unit vectors make the dot product a cosine."""
    vectors = real_embedder.embed_documents(["car maintenance", "a much longer sentence here"])

    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-5, atol=1e-5)


def test_the_same_text_embeds_equivalently_every_time(
    real_embedder: Model2VecEmbedder,
) -> None:
    """Equivalence within tolerance, which is what replicas actually rely on."""
    first = real_embedder.embed_query("distributed search engine")
    second = real_embedder.embed_query("distributed search engine")

    np.testing.assert_allclose(first, second, rtol=1e-6, atol=1e-6)


def test_a_query_and_a_document_embedding_agree(real_embedder: Model2VecEmbedder) -> None:
    text = "cooking pasta"

    np.testing.assert_allclose(
        real_embedder.embed_query(text),
        real_embedder.embed_documents([text])[0],
        rtol=1e-6,
        atol=1e-6,
    )


def test_related_text_scores_higher_than_unrelated(real_embedder: Model2VecEmbedder) -> None:
    anchor = real_embedder.embed_query("car maintenance and engine service")
    related = real_embedder.embed_query("automobile repair")
    unrelated = real_embedder.embed_query("quantum chromodynamics")

    assert float(anchor @ related) > float(anchor @ unrelated)


def test_semantic_search_finds_a_paraphrase_that_shares_no_words(
    real_embedder: Model2VecEmbedder,
) -> None:
    """The property that justifies the whole phase."""
    engine = SearchEngine(InMemoryDocumentStore(), embedder=real_embedder)
    engine.initialize()
    for document_id, text in {
        "doc-car": "automobile repair",
        "doc-search": "ranking documents by relevance",
        "doc-cook": "boiling water for pasta",
    }.items():
        engine.index_document(Document(document_id=document_id, text=text))

    # No word in the query appears in the document.
    top = engine.semantic_search(engine.embed_query("car maintenance"), limit=1).results[0]

    assert top.document_id == "doc-car"
    # And BM25 finds nothing at all for the same query.
    assert engine.search("car maintenance", limit=10).total == 0
