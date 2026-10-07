"""Classical transcription metrics.

Defined once here and imported everywhere. A second implementation in a script or a
notebook is how two "CER" numbers end up in one paper meaning different things, so
``tests/architecture/test_metrics_centralized.py`` fails the build on one.

Aggregation is **corpus-level by default**: total edits over total reference length, not
the mean of per-document rates. Averaging rates weights a five-word caption the same as a
full page, which systematically flatters methods on short documents. The per-document
variant is available and named, because it is the right unit for a document-level
bootstrap.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rapidfuzz.distance import Levenshtein

__all__ = [
    "ErrorRate",
    "cer",
    "character_error_counts",
    "corpus_cer",
    "corpus_wer",
    "exact_match_rate",
    "levenshtein",
    "per_document_cer",
    "wer",
    "word_error_counts",
]


@dataclass(frozen=True, slots=True)
class ErrorRate:
    """An error rate with the counts it came from.

    Keeping the numerator and denominator is what makes corpus-level aggregation possible
    downstream, and what lets a reader see that a 100% rate came from one short line.
    """

    errors: int
    reference_length: int

    @property
    def rate(self) -> float:
        """Errors per reference unit. Zero-length references yield 0.0 when the
        hypothesis is also empty, and 1.0 otherwise, rather than dividing by zero."""
        if self.reference_length == 0:
            return 0.0 if self.errors == 0 else 1.0
        return self.errors / self.reference_length

    def __add__(self, other: ErrorRate) -> ErrorRate:
        return ErrorRate(
            errors=self.errors + other.errors,
            reference_length=self.reference_length + other.reference_length,
        )


def levenshtein(a: str, b: str) -> int:
    """Raw character edit distance."""
    return int(Levenshtein.distance(a, b))


def character_error_counts(reference: str, hypothesis: str) -> ErrorRate:
    """Character edits and reference length for one pair."""
    return ErrorRate(errors=levenshtein(reference, hypothesis), reference_length=len(reference))


def word_error_counts(reference: str, hypothesis: str) -> ErrorRate:
    """Word edits and reference word count, on whitespace tokenization.

    Whitespace tokenization is deliberate and recorded: any smarter tokenizer would make
    WER depend on a linguistic model, and two corpora tokenized differently would produce
    incomparable numbers.
    """
    reference_words = reference.split()
    hypothesis_words = hypothesis.split()
    return ErrorRate(
        errors=int(Levenshtein.distance(reference_words, hypothesis_words)),
        reference_length=len(reference_words),
    )


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate for one pair."""
    return character_error_counts(reference, hypothesis).rate


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate for one pair."""
    return word_error_counts(reference, hypothesis).rate


def corpus_cer(pairs: Sequence[tuple[str, str]]) -> ErrorRate:
    """Corpus-level CER: total character edits over total reference characters."""
    total = ErrorRate(0, 0)
    for reference, hypothesis in pairs:
        total = total + character_error_counts(reference, hypothesis)
    return total


def corpus_wer(pairs: Sequence[tuple[str, str]]) -> ErrorRate:
    """Corpus-level WER: total word edits over total reference words."""
    total = ErrorRate(0, 0)
    for reference, hypothesis in pairs:
        total = total + word_error_counts(reference, hypothesis)
    return total


def per_document_cer(pairs: Sequence[tuple[str, str]]) -> list[float]:
    """CER per document, for document-level resampling and stratification."""
    return [cer(reference, hypothesis) for reference, hypothesis in pairs]


def exact_match_rate(pairs: Sequence[tuple[str, str]]) -> float:
    """Fraction of pairs that match exactly."""
    if not pairs:
        return 0.0
    return sum(1 for reference, hypothesis in pairs if reference == hypothesis) / len(pairs)
