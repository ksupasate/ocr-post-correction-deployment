"""The CGV3 structural generator: proposals at OCR-only discovered anchors.

``g5_structural`` (CGV2) proposed at GT-anchored sites; ``g7_structural_v2`` proposes
at :class:`~ocr_risk.discovery.enumerator.DiscoveredSite` anchors, so every proposal
is keyed to a location a deployed system could have found on its own. The competence
set is unchanged in spirit -- the shapes the substitution rungs cannot express:

- **insertion** at ``GAP`` anchors: the fold-bigram-attested token between the flanks
  (one-sided attestation, the amendment-A2 lesson, reused verbatim);
- **split** at ``TOKEN`` anchors: the two-word decomposition when both halves are
  fold-lexicon entries;
- **merge** and **pair substitution** at ``TOKEN_PAIR`` anchors: the separator-free
  join when it is a lexicon entry, and lexicon entries within edit distance of it.

Ground-truth-blind like every generator: the inputs are the site's anchor, the page's
OCR tokens, and fold-scoped lexical statistics.
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
from ocr_risk.discovery.enumerator import DiscoveredSite
from ocr_risk.discovery.views import OcrPageView
from ocr_risk.schemas.enums import AnchorKind

__all__ = ["StructuralV2Generator"]


@register_generator("structural_v2")
class StructuralV2Generator(BaseGenerator):
    """Insertion, split, merge, and pair-substitution proposals at discovered anchors."""

    generator_id = "g7_structural_v2"
    version = "1"

    def __init__(
        self,
        generator_id: str = "g7_structural_v2",
        max_edit_distance: int = 2,
        min_lexicon_frequency: int = 2,
        min_bigram: int = 2,
        max_insertions: int = 1,
        max_pair_substitutions: int = 2,
        max_per_site: int = 2,
    ) -> None:
        self.generator_id = generator_id
        self.max_edit_distance = max_edit_distance
        self.min_bigram = min_bigram
        self.max_insertions = max_insertions
        self.max_pair_substitutions = max_pair_substitutions
        self.max_per_site = max_per_site
        # The same construction as every other rung, so "known word" means one thing.
        self._lexical = LexicalGenerator(
            max_edit_distance=max_edit_distance, min_lexicon_frequency=min_lexicon_frequency
        )
        self._vocabulary: frozenset[str] = frozenset()
        self._bigrams: Counter[tuple[str, str]] = Counter()

    @property
    def lexicon(self) -> tuple[str, ...]:
        return self._lexical.lexicon

    @property
    def lexicon_size(self) -> int:
        return len(self._lexical.lexicon)

    def available(self) -> bool:
        return True

    def fit(self, corpus: Sequence[str]) -> None:
        self._lexical.fit(corpus)
        self._vocabulary = frozenset(self._lexical.lexicon)
        counts: Counter[tuple[str, str]] = Counter()
        for text in corpus:
            counts.update(pairwise(text.split()))
        self._bigrams = counts

    def _attested_insertions(self, left: str, right: str) -> list[tuple[str, int]]:
        """Fold-attested tokens for the gap, strongest first, one-sided allowed."""
        scored: dict[str, int] = {}
        for (first, second), count in self._bigrams.items():
            if count < self.min_bigram:
                continue
            if first == left:
                token = second
            elif second == right:
                token = first
            else:
                continue
            evidence = (
                count + self._bigrams.get((token, right), 0) + self._bigrams.get((left, token), 0)
            )
            scored[token] = max(scored.get(token, 0), evidence)
        return sorted(scored.items(), key=lambda item: (-item[1], item[0]))

    def _split_candidate(self, token: str) -> str | None:
        """The two-word decomposition, preferring the most balanced cut."""
        if token in self._vocabulary or len(token) < 4:
            return None
        best: tuple[int, str] | None = None
        for cut in range(2, len(token) - 1):
            left, right = token[:cut], token[cut:]
            if left in self._vocabulary and right in self._vocabulary:
                balance = abs(len(left) - len(right))
                key = (balance, left)
                if best is None or key < best:
                    best = (balance, f"{left} {right}")
        return best[1] if best is not None else None

    def propose_at_site(self, site: DiscoveredSite, page: OcrPageView) -> list[CandidateProposal]:
        """Structural proposals for one discovered anchor, best first.

        The incumbent substitution rungs run at the same anchors separately; this
        method contributes only the shapes they cannot express.
        """
        proposals: list[CandidateProposal] = []
        by_span = {token.span_id: token for token in page.tokens}
        if site.anchor_kind is AnchorKind.GAP:
            parts = site.anchor_ref.split("\0")
            left = by_span[parts[1]].text.strip() if len(parts) > 1 and parts[1] in by_span else ""
            right = by_span[parts[2]].text.strip() if len(parts) > 2 and parts[2] in by_span else ""
            for token, evidence in self._attested_insertions(left, right)[: self.max_insertions]:
                proposals.append(
                    CandidateProposal(
                        text=token,
                        score=float(evidence),
                        metadata=self._meta(edit_shape="insertion", bigram_evidence=evidence),
                    )
                )
        elif site.anchor_kind is AnchorKind.TOKEN:
            span_id = site.anchor_ref.split("\0")[1]
            anchor_token = by_span.get(span_id)
            if anchor_token is not None:
                split = self._split_candidate(anchor_token.text.strip())
                if split is not None:
                    proposals.append(
                        CandidateProposal(
                            text=split,
                            score=1.0,
                            metadata=self._meta(edit_shape="split"),
                        )
                    )
        elif site.anchor_kind is AnchorKind.TOKEN_PAIR:
            parts = site.anchor_ref.split("\0")
            left = by_span[parts[1]].text.strip() if len(parts) > 1 and parts[1] in by_span else ""
            right = by_span[parts[2]].text.strip() if len(parts) > 2 and parts[2] in by_span else ""
            joined = f"{left}{right}"
            if joined in self._vocabulary:
                proposals.append(
                    CandidateProposal(
                        text=joined,
                        score=1.0,
                        metadata=self._meta(edit_shape="merge"),
                    )
                )
            remaining = self.max_per_site - len(proposals)
            if remaining > 0 and joined:
                for word, cost, _index in process.extract(
                    joined,
                    self._lexical.lexicon,
                    scorer=Levenshtein.distance,
                    limit=remaining + 1,
                    score_cutoff=self.max_edit_distance,
                ):
                    if word == joined:
                        continue
                    proposals.append(
                        CandidateProposal(
                            text=word,
                            score=-float(cost),
                            metadata=self._meta(edit_shape="pair_substitution", edit_distance=cost),
                        )
                    )
                    if len(proposals) >= self.max_per_site:
                        break
        return proposals[: self.max_per_site]

    def propose(self, context: GenerationContext, max_candidates: int) -> list[CandidateProposal]:
        """Anchored by contract: g7 has no site-less fallback, and pretending to have
        one would blur the discovery/generation boundary the protocol separates."""
        return []
