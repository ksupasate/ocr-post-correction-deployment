"""Lexical candidate generator: nearest in-vocabulary forms.

A baseline, and a deliberately *plausible-but-uninformed* one. It proposes the word the
text most looks like, with no access to the pixels — which makes it exactly the generator
whose mistakes a source-grounded verifier ought to catch and a text-only verifier ought to
wave through.

Its lexicon is fitted, not global. Building it over every document would leak
test-document vocabulary into the generator (leakage vector L5), so ``fit`` takes an
explicit corpus and the caller is responsible for passing only fit-split text.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from ocr_risk.candidates.base import BaseGenerator, CandidateProposal, GenerationContext
from ocr_risk.candidates.registry import register_generator

__all__ = ["LexicalGenerator"]


@register_generator("lexical")
class LexicalGenerator(BaseGenerator):
    """Proposes the closest lexicon entries within an edit-distance budget."""

    generator_id = "lexical"
    version = "1"

    def __init__(
        self,
        generator_id: str = "lexical",
        max_edit_distance: int = 2,
        min_lexicon_frequency: int = 2,
        use_confusable_costs: bool = True,
        min_token_length: int = 3,
        skip_in_vocabulary: bool = True,
    ) -> None:
        self.generator_id = generator_id
        self.max_edit_distance = max_edit_distance
        self.min_lexicon_frequency = min_lexicon_frequency
        self.use_confusable_costs = use_confusable_costs
        self.min_token_length = min_token_length
        # Propose only for out-of-vocabulary spans. A corrector that rewrites tokens it
        # already recognizes is proposing an edit at every site, and since most sites are
        # already correct, ~96% of the resulting pool is overcorrection by construction.
        # Gating on vocabulary membership is standard detector-then-correct practice, uses
        # no ground truth, and is what a deployed lexical corrector would do.
        self.skip_in_vocabulary = skip_in_vocabulary
        self._lexicon: tuple[str, ...] = ()
        self._frequency: Counter[str] = Counter()
        self._fitted = False

    def available(self) -> bool:
        return True

    @property
    def fitted(self) -> bool:
        return self._fitted

    @property
    def lexicon_size(self) -> int:
        return len(self._lexicon)

    @property
    def lexicon(self) -> tuple[str, ...]:
        """The fitted vocabulary, so a gate in front of this corrector can share it.

        Exposed because a detector that rebuilds "known words" from the same corpus does
        not get the same set: this one applies a minimum length and a minimum frequency,
        and a raw re-split keeps every hapax.
        """
        return self._lexicon

    def fit(self, corpus: Sequence[str]) -> None:
        """Build the lexicon from in-scope text only.

        Rare forms are dropped: a lexicon that contains every OCR error it ever saw would
        happily "correct" a word to another engine's mistake.
        """
        counts: Counter[str] = Counter()
        for text in corpus:
            counts.update(token for token in text.split() if len(token) >= self.min_token_length)
        self._frequency = Counter(
            {word: n for word, n in counts.items() if n >= self.min_lexicon_frequency}
        )
        self._lexicon = tuple(sorted(self._frequency))
        self._fitted = True

    def propose(self, context: GenerationContext, max_candidates: int) -> list[CandidateProposal]:
        original = context.original_ocr
        if not self._fitted or not self._lexicon or not original.strip():
            return []
        if len(original) < self.min_token_length:
            return []
        if self.skip_in_vocabulary and original in self._frequency:
            return []

        matches = process.extract(
            original,
            self._lexicon,
            scorer=Levenshtein.distance,
            limit=max_candidates + 1,
            score_cutoff=self.max_edit_distance,
        )

        proposals: list[CandidateProposal] = []
        for word, cost, _index in matches:
            if word == original:
                continue  # an identity proposal is not an edit
            proposals.append(
                CandidateProposal(
                    text=word,
                    # Frequency breaks ties toward the commoner form, which is what makes
                    # this baseline linguistically plausible and pixel-blind.
                    score=-float(cost) + 1e-6 * self._frequency[word],
                    metadata=self._meta(
                        edit_distance=cost, lexicon_frequency=self._frequency[word]
                    ),
                )
            )
            if len(proposals) >= max_candidates:
                break

        proposals.sort(key=lambda p: -(p.score or 0.0))
        return proposals
