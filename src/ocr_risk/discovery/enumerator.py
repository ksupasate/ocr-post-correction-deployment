"""The CGV3 OCR-only site enumerator (protocol section 5).

Discovery runs on :class:`~ocr_risk.discovery.views.OcrPageView` alone: OCR tokens,
native confidence, boxes, line ids, and fold-scoped lexical resources passed in by the
caller. Ground truth, alignments, and the frozen GT-informed site tables are absent by
type and by layer -- ``discovery`` cannot import the layers that carry them.

Every rule is deterministic, every threshold is a named field of
:class:`DiscoveryRules`, and every emitted site records the signals that fired so the
section-11 audit can reconstruct its creation from OCR artifacts alone.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from ocr_risk.discovery.views import OcrPageView
from ocr_risk.schemas.enums import AnchorKind, DiscoveryProvenance

__all__ = ["DiscoveredSite", "DiscoveryResources", "DiscoveryRules", "enumerate_sites"]

ENUMERATOR_VERSION = "cgv3-enumerator-v1"

_SPLIT_MIN_PART = 2
"""Shortest token half accepted by the split rule; single characters match far too much."""

_REASON_RANK: dict[DiscoveryProvenance, int] = {
    # Specific structural evidence outranks generic suspicion: a token that is both
    # low-confidence and a lexicon-word concatenation is recorded as a split find,
    # with every weaker signal retained beside it.
    DiscoveryProvenance.LOW_CONFIDENCE_TOKEN: 1,
    DiscoveryProvenance.BOUNDARY_ANOMALY: 1,
    DiscoveryProvenance.LEXICAL_ANOMALY: 2,
    DiscoveryProvenance.GAP_ANOMALY: 2,
    DiscoveryProvenance.POSSIBLE_SPLIT: 3,
    DiscoveryProvenance.POSSIBLE_MERGE: 3,
    DiscoveryProvenance.SEQUENCE_ANOMALY: 3,
}


@dataclass(frozen=True, slots=True)
class DiscoveryResources:
    """Fold-scoped fitted material: the only statistics discovery may consult."""

    lexicon: frozenset[str] = frozenset()
    """Fold lexicon from ``(E \\ {held_out}) x D_fit`` -- known words, nothing else."""
    bigrams: Counter[tuple[str, str]] = field(default_factory=Counter)
    """Token bigram counts over the same fold corpus."""
    min_bigram_count: int = 2

    def attested_between(self, left: str, right: str) -> str | None:
        """The fold-attested token most strongly suggested between two neighbours.

        Attestation is one-sided, the amendment-A2 lesson: a missing form value follows
        its field label while the next token belongs to the next field, so requiring
        both bigrams fires almost never. Either ``(left, w)`` or ``(w, right)`` at or
        above the count floor qualifies; the winner is scored by total evidence.
        """
        best: tuple[int, str] | None = None
        for (first, second), count in self.bigrams.items():
            if count < self.min_bigram_count:
                continue
            if first == left:
                token = second
            elif second == right:
                token = first
            else:
                continue
            evidence = (
                count + self.bigrams.get((token, right), 0) + self.bigrams.get((left, token), 0)
            )
            key = (evidence, token)
            if best is None or key > best:
                best = key
        return best[1] if best is not None else None


@dataclass(frozen=True, slots=True)
class DiscoveryRules:
    """Every threshold discovery uses; calibrated on Track-A fit documents only."""

    conf_floor: float = 0.55
    """Normalized confidence below which a token becomes a site (per-engine overrides
    via ``conf_floor_by_engine`` take precedence)."""
    conf_floor_by_engine: dict[str, float] = field(default_factory=dict)
    lexical_min_length: int = 3
    require_lexical_neighbour: bool = True
    """A lone OOV token in a field of OOV tokens is noise, not a discovery."""
    gap_mad_k: float = 6.0
    """A gap site fires when the inter-token box gap exceeds median + k x MAD of the
    page's own within-line gaps -- a page-local statistic, computable at deployment."""
    gap_min_pixels: float = 1.0
    enable_boundary_rule: bool = False
    """Line-boundary pairs are a diagnostic population, off in the primary pool."""


@dataclass(frozen=True, slots=True)
class DiscoveredSite:
    """One discovered repair location, anchored in OCR artifacts alone."""

    site_id: str
    document_id: str
    dataset_id: str
    engine_id: str
    anchor_kind: AnchorKind
    anchor_ref: str
    char_start: int
    char_end: int
    site_type: str
    """Best-guess structural class (substitution/insertion/deletion/split/merge/
    boundary); the anchor kind, not this label, is authoritative for coverage."""
    suspicion_score: float
    provenance_reason: DiscoveryProvenance
    signals: dict[str, str]
    """Every signal that fired at this anchor, with its value."""
    enumerator_version: str = ENUMERATOR_VERSION

    def as_dict(self) -> dict[str, object]:
        return {
            "site_id": self.site_id,
            "document_id": self.document_id,
            "dataset_id": self.dataset_id,
            "engine_id": self.engine_id,
            "anchor_kind": self.anchor_kind.value,
            "anchor_ref": self.anchor_ref,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "site_type": self.site_type,
            "suspicion_score": self.suspicion_score,
            "provenance_reason": self.provenance_reason.value,
            "enumerator_version": self.enumerator_version,
            **{f"signal_{key}": value for key, value in self.signals.items()},
        }


def _within_line_gap_stats(
    page: OcrPageView,
) -> tuple[float | None, float | None]:
    """Median and MAD of within-line adjacent box gaps on this page.

    Page-local by design: the deployment system can measure exactly this from the
    page it is processing, so no corpus statistic leaks into the threshold.
    """
    gaps: list[float] = []
    for left, right in page.adjacent_pairs():
        if left.line_id is None or left.line_id != right.line_id:
            continue
        if left.x1 is None or right.x0 is None:
            continue
        gaps.append(max(right.x0 - left.x1, 0.0))
    if len(gaps) < 3:
        return None, None
    ordered = sorted(gaps)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    deviations = sorted(abs(gap - median) for gap in gaps)
    mad = deviations[mid] if len(deviations) % 2 else (deviations[mid - 1] + deviations[mid]) / 2
    return median, mad


def _splits_into_lexicon_words(token: str, lexicon: frozenset[str]) -> bool:
    """Whether the token concatenates exactly two lexicon entries (the split shape).

    A token that is itself a lexicon entry never fires: a known whole word is not a
    discovery, however it might also decompose (``database`` = ``data`` + ``base``).
    """
    if token in lexicon or len(token) < 2 * _SPLIT_MIN_PART:
        return False
    for cut in range(_SPLIT_MIN_PART, len(token) - _SPLIT_MIN_PART + 1):
        left, right = token[:cut], token[cut:]
        if left in lexicon and right in lexicon:
            return True
    return False


@dataclass(slots=True)
class _PendingSite:
    """Mutable accumulator for one anchor while the rules run."""

    anchor_kind: AnchorKind
    anchor_ref: str
    char_start: int
    char_end: int
    site_type: str
    reason: DiscoveryProvenance
    suspicion: float = 0.0
    signals: dict[str, str] = field(default_factory=dict)


def enumerate_sites(
    page: OcrPageView,
    resources: DiscoveryResources | None = None,
    rules: DiscoveryRules | None = None,
) -> list[DiscoveredSite]:
    """Discover candidate repair locations on one page, deterministically.

    Sites are ordered by stream position and receive ids
    ``{document}:{engine}:dsite:{index:05d}``. Multiple rules firing at one anchor
    collapse into a single site whose ``signals`` records all of them; the
    ``provenance_reason`` is the first rule in precedence order.
    """
    resources = resources or DiscoveryResources()
    rules = rules or DiscoveryRules()
    conf_floor = rules.conf_floor_by_engine.get(page.engine_id, rules.conf_floor)
    lexicon = resources.lexicon
    pending: dict[str, _PendingSite] = {}

    def _record(
        anchor_key: str,
        *,
        anchor_kind: AnchorKind,
        anchor_ref: str,
        char_start: int,
        char_end: int,
        site_type: str,
        reason: DiscoveryProvenance,
        suspicion: float,
        signal: str,
        value: str,
    ) -> None:
        entry = pending.setdefault(
            anchor_key,
            _PendingSite(
                anchor_kind=anchor_kind,
                anchor_ref=anchor_ref,
                char_start=char_start,
                char_end=char_end,
                site_type=site_type,
                reason=reason,
            ),
        )
        entry.suspicion = max(entry.suspicion, suspicion)
        entry.signals[signal] = value
        if _REASON_RANK[reason] > _REASON_RANK[entry.reason]:
            entry.reason = reason
        # The most specific typing wins: a token anchor that splits into lexicon words
        # is a split find regardless of which rule fired first.
        if site_type == "split":
            entry.site_type = "split"

    for position, token in enumerate(page.tokens):
        text = token.text.strip()
        anchor_key = page.anchor_ref(AnchorKind.TOKEN, token)
        if token.normalized_confidence is not None and token.normalized_confidence < conf_floor:
            _record(
                anchor_key,
                anchor_kind=AnchorKind.TOKEN,
                anchor_ref=anchor_key,
                char_start=token.char_start,
                char_end=token.char_end,
                site_type="substitution",
                reason=DiscoveryProvenance.LOW_CONFIDENCE_TOKEN,
                suspicion=1.0 - token.normalized_confidence,
                signal="normalized_confidence",
                value=f"{token.normalized_confidence:.4f}",
            )
        neighbours = [
            page.tokens[position - 1].text.strip() if position else "",
            page.tokens[position + 1].text.strip() if position + 1 < len(page.tokens) else "",
        ]
        lexically_anomalous = (
            len(text) >= rules.lexical_min_length
            and bool(text)
            and text not in lexicon
            and (
                not rules.require_lexical_neighbour
                or any(neighbour and neighbour in lexicon for neighbour in neighbours)
            )
        )
        if lexically_anomalous:
            _record(
                anchor_key,
                anchor_kind=AnchorKind.TOKEN,
                anchor_ref=anchor_key,
                char_start=token.char_start,
                char_end=token.char_end,
                site_type="substitution",
                reason=DiscoveryProvenance.LEXICAL_ANOMALY,
                suspicion=0.6,
                signal="oov_token",
                value=text,
            )
        if _splits_into_lexicon_words(text, lexicon):
            _record(
                anchor_key,
                anchor_kind=AnchorKind.TOKEN,
                anchor_ref=anchor_key,
                char_start=token.char_start,
                char_end=token.char_end,
                site_type="split",
                reason=DiscoveryProvenance.POSSIBLE_SPLIT,
                suspicion=0.7,
                signal="concatenated_words",
                value=text,
            )

    gap_median, gap_mad = _within_line_gap_stats(page)
    gap_threshold = (
        gap_median + rules.gap_mad_k * gap_mad
        if gap_median is not None and gap_mad is not None
        else None
    )
    for left, right in page.adjacent_pairs():
        pair_key = page.anchor_ref(AnchorKind.TOKEN_PAIR, left, right)
        same_line = left.line_id is not None and left.line_id == right.line_id
        join = f"{left.text.strip()}{right.text.strip()}"
        if same_line and join and join in lexicon:
            _record(
                pair_key,
                anchor_kind=AnchorKind.TOKEN_PAIR,
                anchor_ref=pair_key,
                char_start=left.char_start,
                char_end=right.char_end,
                site_type="merge",
                reason=DiscoveryProvenance.POSSIBLE_MERGE,
                suspicion=0.7,
                signal="joined_lexicon_word",
                value=join,
            )
        gap_key = page.anchor_ref(AnchorKind.GAP, left, right)
        if same_line:
            if (
                gap_threshold is not None
                and left.x1 is not None
                and right.x0 is not None
                and right.x0 - left.x1 >= max(rules.gap_min_pixels, gap_threshold)
            ):
                _record(
                    gap_key,
                    anchor_kind=AnchorKind.GAP,
                    anchor_ref=gap_key,
                    char_start=left.char_end,
                    char_end=right.char_start,
                    site_type="insertion",
                    reason=DiscoveryProvenance.GAP_ANOMALY,
                    suspicion=0.5,
                    signal="box_gap_pixels",
                    value=f"{right.x0 - left.x1:.1f}",
                )
            attested = resources.attested_between(left.text.strip(), right.text.strip())
            if attested is not None:
                _record(
                    gap_key,
                    anchor_kind=AnchorKind.GAP,
                    anchor_ref=gap_key,
                    char_start=left.char_end,
                    char_end=right.char_start,
                    site_type="insertion",
                    reason=DiscoveryProvenance.SEQUENCE_ANOMALY,
                    suspicion=0.6,
                    signal="fold_attested_between",
                    value=attested,
                )
        elif rules.enable_boundary_rule:
            _record(
                pair_key,
                anchor_kind=AnchorKind.TOKEN_PAIR,
                anchor_ref=pair_key,
                char_start=left.char_start,
                char_end=right.char_end,
                site_type="boundary",
                reason=DiscoveryProvenance.BOUNDARY_ANOMALY,
                suspicion=0.4,
                signal="line_boundary_pair",
                value=f"{left.text.strip()}|{right.text.strip()}",
            )

    ordered = sorted(
        pending.values(),
        key=lambda entry: (entry.char_start, entry.anchor_kind.value, entry.anchor_ref),
    )
    return [
        DiscoveredSite(
            site_id=f"{page.document_id}:{page.engine_id}:dsite:{index:05d}",
            document_id=page.document_id,
            dataset_id=page.dataset_id,
            engine_id=page.engine_id,
            anchor_kind=entry.anchor_kind,
            anchor_ref=entry.anchor_ref,
            char_start=entry.char_start,
            char_end=entry.char_end,
            site_type=entry.site_type,
            suspicion_score=entry.suspicion,
            provenance_reason=entry.reason,
            signals=dict(entry.signals),
        )
        for index, entry in enumerate(ordered)
    ]
