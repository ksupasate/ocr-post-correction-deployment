"""A generator whose edit vocabulary is not limited to substitution.

The pilot's failure analysis found a structural ceiling that no amount of lexical accuracy
could lift: 13 511 of 59 211 natural candidates (22.8%) sat on **insertion** sites, where
the OCR invented text that has no ground-truth counterpart and the only correct edit is a
deletion. A generator that can only replace one token with another token cannot express
that edit, so on those sites its best possible proposal is still wrong — 8 691 of them came
back ``lateral_change`` and 901 ``miscorrection``.

This generator adds the two edit shapes the alignment layer already recognizes as distinct
site kinds and the corrector could not produce:

- **deletion** — propose the empty string, i.e. drop the span;
- **split** — propose reinserting a space where a merge is likely.

Merge is deliberately absent. Merging two OCR tokens is an edit spanning two sites, and a
generator that sees one site plus its context strings cannot apply it without the
neighbouring site also being edited. That is a limitation of the site granularity, not an
oversight, and it is reported rather than approximated.

Ground-truth-blind like every generator: the cues are confidence, vocabulary membership,
length, and character class.
"""

from __future__ import annotations

from collections.abc import Sequence

from ocr_risk.candidates.base import BaseGenerator, CandidateProposal, GenerationContext
from ocr_risk.candidates.detector import error_signals
from ocr_risk.candidates.registry import build_generator, register_generator

__all__ = ["EditAwareGenerator"]


@register_generator("edit_aware")
class EditAwareGenerator(BaseGenerator):
    """Substitution from an inner corrector, plus deletion and split proposals."""

    generator_id = "edit_aware"
    version = "1"

    def __init__(
        self,
        generator_id: str = "edit_aware",
        inner_kind: str = "error_gated",
        inner_params: dict[str, object] | None = None,
        delete_below_confidence: float = 0.45,
        delete_max_length: int = 8,
        min_split_part: int = 3,
        max_splits: int = 1,
    ) -> None:
        self.generator_id = generator_id
        self.delete_below_confidence = delete_below_confidence
        self.delete_max_length = delete_max_length
        self.min_split_part = min_split_part
        self.max_splits = max_splits
        self._inner = build_generator(
            inner_kind, generator_id=f"{generator_id}:{inner_kind}", **dict(inner_params or {})
        )
        self._vocabulary: frozenset[str] = frozenset()

    def available(self) -> bool:
        return self._inner.available()

    def fit(self, corpus: Sequence[str]) -> None:
        self._inner.fit(corpus)
        # The inner gate's vocabulary, which is the corrector's fitted lexicon -- so the
        # deletion and split gates use the same notion of "known word" as the corrector.
        inner = getattr(self._inner, "_vocabulary", None)
        self._vocabulary = (
            frozenset(inner)
            if inner
            else frozenset(token for text in corpus for token in text.split())
        )

    @property
    def lexicon_size(self) -> int:
        size = getattr(self._inner, "lexicon_size", None)
        return int(size) if size is not None else len(self._vocabulary)

    def _deletion(self, context: GenerationContext) -> CandidateProposal | None:
        """Propose dropping the span, where the evidence says it should not be there.

        Confidence does most of the work. A hallucinated span is short, unrecognized, and
        the engine usually knows: the deletion gate fires only well below the gate that
        merely suspects an error, because deleting real text is the most damaging edit
        available and there is no partial credit for it.
        """
        text = context.original_ocr
        if not text or len(text) > self.delete_max_length:
            return None
        signals = error_signals(context, self._vocabulary)
        if signals.in_vocabulary:
            return None
        confidence = signals.normalized_confidence
        if confidence is None or confidence >= self.delete_below_confidence:
            return None
        return CandidateProposal(
            text="",
            score=1.0 - confidence,
            metadata=self._meta(edit_shape="deletion", confidence=round(confidence, 4)),
        )

    def _splits(self, context: GenerationContext, limit: int) -> list[CandidateProposal]:
        """Propose reinserting a space where both halves are known words.

        Requiring *both* halves to be in the lexicon is what keeps this from firing on
        every long token: an unrecognized run of characters is not evidence of a merge
        unless the pieces are themselves words the corpus contains.
        """
        text = context.original_ocr
        if len(text) < 2 * self.min_split_part or " " in text:
            return []
        found: list[CandidateProposal] = []
        for cut in range(self.min_split_part, len(text) - self.min_split_part + 1):
            left, right = text[:cut], text[cut:]
            if left in self._vocabulary and right in self._vocabulary:
                found.append(
                    CandidateProposal(
                        text=f"{left} {right}",
                        score=float(min(len(left), len(right))),
                        metadata=self._meta(edit_shape="split", cut=cut),
                    )
                )
            if len(found) >= limit:
                break
        return found

    def propose(self, context: GenerationContext, max_candidates: int) -> list[CandidateProposal]:
        proposals: list[CandidateProposal] = []
        deletion = self._deletion(context)
        if deletion is not None:
            proposals.append(deletion)
        proposals.extend(self._splits(context, self.max_splits))
        remaining = max_candidates - len(proposals)
        if remaining > 0:
            proposals.extend(self._inner.propose(context, remaining))
        return proposals[:max_candidates]
