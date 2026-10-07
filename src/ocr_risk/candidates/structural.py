"""A generator for the edit shapes no site-level rung can produce.

The recovery phase measured the structural ceiling of the frozen ladder: at sites whose
ground truth has a token the OCR never emitted (site kind ``deletion``, 5 115 evaluable
sites on the pilot corpus) every rung proposes nothing — ``LexicalGenerator.propose``
returns ``[]`` on empty text and no other rung targets them — and an edit that rewrites
two adjacent sites jointly is inexpressible at site granularity altogether
(``edit_aware`` deliberately reports rather than approximates it).

This rung adds exactly those shapes:

- **insertion** — at an empty-OCR site, propose the token the fold corpus most strongly
  attests between the surrounding tokens;
- **merge** — at a multi-span site, propose the separator-free (or hyphen) join when it
  is a known word;
- **region_join / region_substitution** — for two adjacent sites, propose a lexicon word
  for the pair jointly (see ``propose_region``).

Ground-truth-blind like every generator: the cues are vocabulary membership and bigram
counts from the fold's allowed corpus, and nothing else.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from itertools import pairwise

from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from ocr_risk.candidates.base import BaseGenerator, CandidateProposal, GenerationContext
from ocr_risk.candidates.lexical import LexicalGenerator
from ocr_risk.candidates.registry import register_generator

__all__ = ["StructuralGenerator"]


@register_generator("structural")
class StructuralGenerator(BaseGenerator):
    """Insertion, merge-join, and region proposals; no substitution competence."""

    generator_id = "structural"
    version = "1"

    def __init__(
        self,
        generator_id: str = "structural",
        min_bigram: int = 2,
        max_insertion: int = 1,
        max_region_candidates: int = 2,
        max_edit_distance: int = 2,
        min_lexicon_frequency: int = 2,
    ) -> None:
        self.generator_id = generator_id
        self.min_bigram = min_bigram
        self.max_insertion = max_insertion
        self.max_region_candidates = max_region_candidates
        # The same construction as the corrector's lexicon, so "known word" means the
        # same thing here as it does for every other rung.
        self._lexical = LexicalGenerator(
            max_edit_distance=max_edit_distance, min_lexicon_frequency=min_lexicon_frequency
        )
        self._vocabulary: frozenset[str] = frozenset()
        # The insertion vocabulary is the full fold token distribution, punctuation and
        # single digits included: the calibration census of empty-OCR sites showed their
        # ground truth is 3 characters at the median -- '-', '1', 'of', '(' -- none of
        # which a min-length-3 lexicon can ever propose (amendment A2).
        self._insertion_vocab: frozenset[str] = frozenset()
        self._bigrams: Counter[tuple[str, str]] = Counter()

    def available(self) -> bool:
        return True

    @property
    def lexicon_size(self) -> int:
        return len(self._vocabulary)

    def fit(self, corpus: Sequence[str]) -> None:
        self._lexical.fit(corpus)
        self._vocabulary = frozenset(self._lexical.lexicon)
        tokens: Counter[str] = Counter()
        counts: Counter[tuple[str, str]] = Counter()
        for text in corpus:
            parts = text.split()
            tokens.update(parts)
            counts.update(pairwise(parts))
        self._insertion_vocab = frozenset(t for t, n in tokens.items() if n >= self.min_bigram)
        self._bigrams = counts

    def _insertion(self, context: GenerationContext) -> list[CandidateProposal]:
        """The missing token, where either neighbour attests one.

        Attestation may be one-sided: a missing form value follows its field label, and
        whatever comes after it is the next field's first token, which no bigram of the
        fold corpus constrains. The candidate still has to be a token the fold corpus
        actually contains, so an unattested site proposes nothing rather than guessing.
        """
        if context.original_ocr.strip() or self.max_insertion <= 0:
            return []
        before = context.context_before.split()
        after = context.context_after.split()
        if not before or not after:
            return []
        prev_token, next_token = before[-1], after[0]
        # (total, forward, backward, token) -- counts ride along so the metadata reports
        # the winner's evidence, not the last loop iteration's.
        best: tuple[int, int, int, str] | None = None
        for (left, right), count in self._bigrams.items():
            if left != prev_token or right == next_token or count < self.min_bigram:
                continue
            if right not in self._insertion_vocab:
                continue
            backward = self._bigrams.get((right, next_token), 0)
            key = (count + backward, count, backward, right)
            if best is None or key > best:
                best = key
        for (left, right), count in self._bigrams.items():
            # The one-sided backward pass: the token is attested before the next token
            # even when nothing attests it after the previous one.
            if right != next_token or left == prev_token or count < self.min_bigram:
                continue
            if left not in self._insertion_vocab:
                continue
            forward = self._bigrams.get((prev_token, left), 0)
            key = (count + forward, forward, count, left)
            if best is None or key > best:
                best = key
        if best is None:
            return []
        total, forward, backward, word = best
        return [
            CandidateProposal(
                text=word,
                score=float(total),
                metadata=self._meta(
                    edit_shape="insertion", bigram_forward=forward, bigram_backward=backward
                ),
            )
        ]

    def _merges(self, context: GenerationContext) -> list[CandidateProposal]:
        """The merge repair at a site the OCR split: join the parts into one known word."""
        text = context.original_ocr
        if context.n_spans <= 1 or not text.strip():
            return []
        found: list[CandidateProposal] = []
        for variant in (text.replace(" ", ""), text.replace(" ", "-")):
            if variant in self._vocabulary:
                found.append(
                    CandidateProposal(
                        text=variant,
                        score=1.0,
                        metadata=self._meta(edit_shape="merge", variant=variant),
                    )
                )
            if len(found) >= 2:
                break
        return found

    def propose(self, context: GenerationContext, max_candidates: int) -> list[CandidateProposal]:
        proposals = self._insertion(context)
        proposals.extend(self._merges(context))
        return proposals[:max_candidates]

    def propose_region(
        self, left_text: str, right_text: str, max_candidates: int | None = None
    ) -> list[CandidateProposal]:
        """Propose a joint replacement for two adjacent sites' OCR text.

        Pure function of the two strings and the fitted lexicon — a region candidate
        needs no more of the page than its members, and giving it more would blur the
        boundary the region is defined by.
        """
        limit = self.max_region_candidates if max_candidates is None else max_candidates
        if not left_text.strip() or not right_text.strip() or limit <= 0:
            return []
        joined = f"{left_text}{right_text}"
        proposals: list[CandidateProposal] = []
        if joined in self._vocabulary:
            proposals.append(
                CandidateProposal(
                    text=joined,
                    score=1.0,
                    metadata=self._meta(edit_shape="region_join"),
                )
            )
        remaining = limit - len(proposals)
        if remaining > 0:
            for word, cost, _index in process.extract(
                joined,
                self._lexical.lexicon,
                scorer=Levenshtein.distance,
                limit=remaining + 1,
                score_cutoff=self._lexical.max_edit_distance,
            ):
                if word == joined:
                    continue
                proposals.append(
                    CandidateProposal(
                        text=word,
                        score=-float(cost),
                        metadata=self._meta(edit_shape="region_substitution", edit_distance=cost),
                    )
                )
                if len(proposals) >= limit:
                    break
        return proposals[:limit]
