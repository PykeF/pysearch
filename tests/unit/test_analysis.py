"""Tests for normalization and tokenization."""

from app.search.analysis import analyze, normalize, tokenize


def test_normalization_folds_case() -> None:
    assert normalize("Distributed SEARCH") == "distributed search"


def test_normalization_applies_nfkc() -> None:
    # Full-width characters are compatibility variants of their ASCII forms.
    # The lookalike characters are the point of the test, hence the suppression.
    assert normalize("ＢＭ２５") == "bm25"  # noqa: RUF001


def test_casefold_handles_non_ascii_case_pairs() -> None:
    # str.lower would leave this unchanged; casefold is the Unicode-aware form.
    assert normalize("STRASSE") == normalize("Straße")


def test_tokenizer_splits_on_punctuation_and_whitespace() -> None:
    assert tokenize("distributed, search; engines!") == ["distributed", "search", "engines"]


def test_tokenizer_preserves_repeated_terms() -> None:
    # Term frequency is derived from this sequence, so duplicates must survive.
    assert tokenize("search search index") == ["search", "search", "index"]


def test_tokenizer_keeps_alphanumeric_runs_together() -> None:
    assert tokenize("bm25 utf8") == ["bm25", "utf8"]


def test_tokenizer_returns_nothing_for_empty_input() -> None:
    assert tokenize("") == []


def test_tokenizer_returns_nothing_for_whitespace_or_punctuation() -> None:
    assert tokenize("   ") == []
    assert tokenize("!!! ... ???") == []


def test_apostrophes_separate_tokens() -> None:
    # A documented limitation rather than a desirable outcome.
    assert analyze("don't") == ["don", "t"]


def test_contiguous_cjk_is_a_single_token() -> None:
    # No word segmentation is attempted; also a documented limitation.
    assert analyze("分布式搜索") == ["分布式搜索"]


def test_analyze_applies_normalization_before_tokenization() -> None:
    assert analyze("Distributed, SEARCH!") == ["distributed", "search"]


def test_documents_and_queries_share_one_pipeline() -> None:
    # The property the whole design depends on: a query term and the document
    # term it should match analyse identically.
    assert analyze("Scalable Search!") == analyze("scalable, search")
