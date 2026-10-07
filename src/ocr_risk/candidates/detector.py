"""A ground-truth-blind error detector, and the gate that puts one in front of a corrector.

Detection before correction is **established prior art**, not a contribution here. It is in
the ladder because the H1 pilot measured what happens without it: 36.3% of the natural
candidate pool sat on sites where the OCR was already correct, so more than a third of
every proposed edit was overcorrection before a verifier saw it. A generator that proposes
everywhere makes the verifier's job the detector's job as well.

The detector sees OCR text, its neighbours, and the engine's own confidence. It never sees
ground truth — enforced by the :class:`~ocr_risk.candidates.base.GenerationContext` type,
which has no field for it, and red-teamed in ``tests/leakage/test_generator_is_gt_blind.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ocr_risk.candidates.base import BaseGenerator, CandidateProposal, GenerationContext
from ocr_risk.candidates.registry import build_generator, register_generator
from ocr_risk.canonical.confidence import normalize_confidence

__all__ = ["ErrorGatedGenerator", "ErrorSignals", "error_signals"]


@dataclass(frozen=True, slots=True)
class ErrorSignals:
    """What the detector reads. Every field is derivable without ground truth."""

    normalized_confidence: float | None
    in_vocabulary: bool
    length: int
    has_digit: bool
    has_alpha: bool
    mixed_alnum: bool
    """A token mixing letters and digits, e.g. ``l0022`` — a classic recognition slip and
    also a perfectly ordinary form field, so it is a signal and not a verdict."""
    non_alnum_fraction: float
    repeated_character_run: bool


def error_signals(
    context: GenerationContext, vocabulary: frozenset[str] | None = None
) -> ErrorSignals:
    """Derive the detector's features for one site."""
    text = context.original_ocr
    values = [
        v
        for v in (normalize_confidence(c, context.conf_scale) for c in context.native_confidences)
        if v is not None
    ]
    letters = sum(c.isalpha() for c in text)
    digits = sum(c.isdigit() for c in text)
    runs = any(text[i] == text[i + 1] == text[i + 2] for i in range(max(0, len(text) - 2)))
    return ErrorSignals(
        # The weakest span drives the risk at a multi-span site, matching how
        # ``conf_normalized`` is summarized in the evidence bundle.
        normalized_confidence=min(values) if values else None,
        in_vocabulary=bool(vocabulary and text in vocabulary),
        length=len(text),
        has_digit=digits > 0,
        has_alpha=letters > 0,
        mixed_alnum=letters > 0 and digits > 0,
        non_alnum_fraction=(sum(not c.isalnum() for c in text) / len(text) if text else 0.0),
        repeated_character_run=runs,
    )


@register_generator("error_gated")
class ErrorGatedGenerator(BaseGenerator):
    """Run an inner generator only where a ground-truth-blind detector suspects an error.

    The gate is a small scored rule rather than a learned model on purpose: a learned
    detector would need its own fit split, its own leakage story, and its own selection
    protocol, and this is a *baseline* whose job is to establish whether gating helps at
    all. If it does, replacing the rule with a learned detector is a later, separate
    question.

    A site passes when its evidence for being wrong reaches ``threshold``. Confidence
    carries the most weight because it is the one signal the engine itself provides and the
    one a deployed system always has; vocabulary membership is the strongest single
    predictor available without it, and is the gate the plain lexical generator already
    applies on its own.
    """

    generator_id = "error_gated"
    version = "1"

    def __init__(
        self,
        generator_id: str = "error_gated",
        inner_kind: str = "lexical",
        inner_params: dict[str, object] | None = None,
        threshold: float = 0.5,
        low_confidence: float = 0.80,
        very_low_confidence: float = 0.55,
    ) -> None:
        self.generator_id = generator_id
        self.threshold = threshold
        self.low_confidence = low_confidence
        self.very_low_confidence = very_low_confidence
        params = dict(inner_params or {})
        # The inner corrector must not apply its own vocabulary gate as well: that would
        # make the ablation "gate" versus "gate twice" instead of "gate" versus "no gate".
        params.setdefault("skip_in_vocabulary", False)
        self._inner = build_generator(
            inner_kind, generator_id=f"{generator_id}:{inner_kind}", **params
        )
        self._vocabulary: frozenset[str] = frozenset()

    def available(self) -> bool:
        return self._inner.available()

    def fit(self, corpus: Sequence[str]) -> None:
        self._inner.fit(corpus)
        # The corrector's FITTED LEXICON, not the raw corpus tokens. Splitting the corpus
        # again produced a different object: the lexicon keeps tokens of length >= 3 seen
        # at least twice, while a raw split keeps every hapax and every one-character
        # fragment. On the pilot corpus that made the detector's vocabulary 3-4x larger,
        # and 10-45% of the extra entries were real OCR errors another engine happened to
        # emit once -- so the gate treated them as known words and declined to look at
        # 280 genuine Tesseract errors. "Known word" now means the same thing to the
        # detector and to the corrector it gates.
        lexicon = getattr(self._inner, "lexicon", None)
        self._vocabulary = (
            frozenset(lexicon)
            if lexicon is not None
            else frozenset(token for text in corpus for token in text.split())
        )

    @property
    def vocabulary_size(self) -> int:
        return len(self._vocabulary)

    @property
    def lexicon_size(self) -> int:
        """Delegated, so a study record does not fall back to counting corpus strings."""
        size = getattr(self._inner, "lexicon_size", None)
        return int(size) if size is not None else len(self._vocabulary)

    def suspicion(self, signals: ErrorSignals) -> float:
        """Evidence that this span is misrecognized, on an arbitrary 0-1-ish scale."""
        score = 0.0
        confidence = signals.normalized_confidence
        if confidence is not None:
            if confidence < self.very_low_confidence:
                score += 0.6
            elif confidence < self.low_confidence:
                score += 0.3
        else:
            # No confidence reported is not evidence of correctness. Treat it as neutral
            # rather than as a pass, or engines that report nothing become ungated.
            score += 0.2
        if not signals.in_vocabulary:
            score += 0.35
        if signals.mixed_alnum:
            score += 0.2
        if signals.repeated_character_run:
            score += 0.1
        if signals.non_alnum_fraction > 0.4:
            score += 0.1
        if signals.length < 3:
            # Very short spans are both the least informative and the most dangerous to
            # rewrite, since a one-character edit is a large relative change.
            score -= 0.3
        return score

    def propose(self, context: GenerationContext, max_candidates: int) -> list[CandidateProposal]:
        signals = error_signals(context, self._vocabulary)
        suspicion = self.suspicion(signals)
        if suspicion < self.threshold:
            return []
        proposals = self._inner.propose(context, max_candidates)
        return [
            CandidateProposal(
                text=p.text,
                score=p.score,
                is_synthetic_hard_negative=p.is_synthetic_hard_negative,
                hard_negative_family=p.hard_negative_family,
                metadata={**p.metadata, "gate_suspicion": f"{suspicion:.3f}"},
            )
            for p in proposals
        ]
