"""Text analysis: the single normalization and tokenization pipeline.

The same pipeline runs over documents at index time and over queries at search
time. Sharing one entry point is what guarantees the two cannot drift apart —
a query term that was normalised differently from the document term it should
match would simply never match it.

The rules are deliberately simple and language-agnostic:

1. Unicode NFKC normalization, so that compatibility variants of the same
   character (full-width forms, ligatures) compare equal.
2. Case folding via ``str.casefold``, which is the Unicode-aware operation
   ``str.lower`` is not: it maps ``ß`` to ``ss``.
3. Tokens are maximal runs of alphanumeric characters. Everything else —
   punctuation, symbols, whitespace — separates tokens and is discarded.

Two consequences are worth stating plainly, because they are limitations
rather than decisions:

* An apostrophe separates tokens, so ``don't`` analyses to ``don`` and ``t``.
* There is no word segmentation for scripts that do not delimit words with
  spaces, so a run of CJK characters becomes a single token.

Stemming, lemmatization, stop-word removal and language detection are all out
of scope; they change what "the same word" means and deserve to be introduced
deliberately, with evaluation.
"""

import unicodedata


def normalize(text: str) -> str:
    """Apply Unicode normalization and case folding."""
    return unicodedata.normalize("NFKC", text).casefold()


def tokenize(text: str) -> list[str]:
    """Split text into maximal runs of alphanumeric characters.

    Repeated terms are preserved rather than deduplicated, because term
    frequency is computed from this sequence.
    """
    tokens: list[str] = []
    current: list[str] = []

    for character in text:
        if character.isalnum():
            current.append(character)
        elif current:
            tokens.append("".join(current))
            current.clear()

    if current:
        tokens.append("".join(current))

    return tokens


def analyze(text: str) -> list[str]:
    """Normalize and tokenize text into the terms used by the index."""
    return tokenize(normalize(text))
