"""CGV3 Track-A study: discovery, offline coverage, generation, failure decomposition.

Layer roles are strict. Discovery runs on :class:`OcrPageView` alone (ground-truth
blind, enforced by layering). Everything in this module that touches GT -- true-region
construction, coverage matching, candidate labelling, failure classification -- runs
*after* enumeration, on saved frames, exactly as the protocol's offline-evaluation
stage demands.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from ocr_risk.candidates.base import CandidateProposal
from ocr_risk.discovery.enumerator import DiscoveredSite
from ocr_risk.discovery.views import OcrPageView
from ocr_risk.edits.outcome import classify_accepted, distance
from ocr_risk.schemas.enums import AnchorKind, OutcomeIfAccepted

__all__ = [
    "AlignmentIndex",
    "CandidateRow",
    "anchor_stream_text",
    "coverage_table",
    "true_region_rows",
]


# --------------------------------------------------------------------------- GT index


@dataclass(frozen=True, slots=True)
class _Component:
    alignment_id: str
    ocr_span_ids: tuple[str, ...]
    gt_token_ids: tuple[str, ...]
    ocr_text: str
    gt_text: str


class AlignmentIndex:
    """Offline GT coupling for one (document, engine): components ordered by GT token.

    Built from the frozen alignment table and the manifest's GT token order. Used only
    downstream of enumeration -- to give true regions their flanking OCR spans and
    discovered anchors their region ground truth.
    """

    def __init__(
        self,
        components: Sequence[_Component],
        span_order: dict[str, int],
        token_order: dict[str, int],
    ) -> None:
        def _key(component: _Component) -> tuple[int, int, str]:
            token_keys = [token_order[t] for t in component.gt_token_ids if t in token_order]
            span_keys = [span_order[s] for s in component.ocr_span_ids if s in span_order]
            return (
                min(token_keys, default=len(token_order)),
                min(span_keys, default=-1),
                component.alignment_id,
            )

        self.components = sorted(components, key=_key)
        self._by_span: dict[str, _Component] = {}
        for component in self.components:
            for span_id in component.ocr_span_ids:
                self._by_span[span_id] = component

    @classmethod
    def from_frames(
        cls, alignments: pd.DataFrame, gt_tokens: pd.DataFrame
    ) -> dict[tuple[str, str], AlignmentIndex]:
        """One index per (document, engine) pair present in the alignment table."""
        token_order = {
            str(row.gt_token_id): int(getattr(row, "index", position))
            for position, row in enumerate(gt_tokens.itertuples(index=False))
        }
        grouped: dict[tuple[str, str], list[_Component]] = {}
        span_order: dict[str, int] = {}
        for row in alignments.itertuples(index=False):
            span_ids = _id_list(row.ocr_span_ids)
            token_ids = _id_list(row.gt_token_ids)
            for span_id in span_ids:
                span_order.setdefault(span_id, len(span_order))
            grouped.setdefault((str(row.document_id), str(row.engine_id)), []).append(
                _Component(
                    alignment_id=str(row.alignment_id),
                    ocr_span_ids=span_ids,
                    gt_token_ids=token_ids,
                    ocr_text=str(row.ocr_text or ""),
                    gt_text=str(row.gt_text or ""),
                )
            )
        return {
            pair: cls(components, span_order, token_order) for pair, components in grouped.items()
        }

    def component_for_span(self, span_id: str) -> _Component | None:
        return self._by_span.get(span_id)

    def flanks_for_empty_component(self, alignment_id: str) -> tuple[str | None, str | None]:
        """Last span before and first span after an OCR-empty component, in GT order."""
        position = next(
            (i for i, c in enumerate(self.components) if c.alignment_id == alignment_id), None
        )
        if position is None:
            return None, None
        before = next(
            (c.ocr_span_ids[-1] for c in reversed(self.components[:position]) if c.ocr_span_ids),
            None,
        )
        after = next(
            (c.ocr_span_ids[0] for c in self.components[position + 1 :] if c.ocr_span_ids),
            None,
        )
        return before, after

    def region_truth(self, span_ids: Sequence[str]) -> tuple[str, str]:
        """(ocr_text, gt_text) of the region covering exactly these spans."""
        texts: list[str] = []
        gt_texts: list[str] = []
        seen: set[str] = set()
        for span_id in span_ids:
            component = self._by_span.get(span_id)
            if component is None or component.alignment_id in seen:
                continue
            seen.add(component.alignment_id)
            if component.ocr_text.strip():
                texts.append(component.ocr_text)
            if component.gt_text.strip():
                gt_texts.append(component.gt_text)
        return " ".join(texts), " ".join(gt_texts)

    def gap_truth(self, left_span: str, right_span: str) -> str:
        """GT text of the OCR-empty components between two flanking spans."""
        position_left = self._span_position(left_span)
        position_right = self._span_position(right_span)
        if position_left is None or position_right is None or position_left >= position_right:
            return ""
        gt_texts = [
            c.gt_text
            for c in self.components[position_left + 1 : position_right]
            if not c.ocr_span_ids and c.gt_text.strip()
        ]
        return " ".join(gt_texts)

    def _span_position(self, span_id: str) -> int | None:
        for position, component in enumerate(self.components):
            if span_id in component.ocr_span_ids:
                return position
        return None


def _id_list(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    if hasattr(value, "tolist"):  # numpy arrays arrive from parquet-backed frames
        converted = value.tolist()
        return tuple(str(item) for item in converted) if isinstance(converted, list) else ()
    text = str(value)
    if not text.strip():
        return ()
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        return tuple(part for part in text.split() if part)
    if isinstance(parsed, list):
        return tuple(str(item) for item in parsed)
    return ()


# ---------------------------------------------------------------------- true regions


def true_region_rows(
    sites: pd.DataFrame, indexes: dict[tuple[str, str], AlignmentIndex]
) -> pd.DataFrame:
    """Eligible GT-informed error regions with the anchors a discovery could match.

    Deletion-kind sites (a GT token the OCR never emitted) carry no spans, so their
    matchable evidence is the pair of flanking OCR spans, derived here from the frozen
    alignment -- offline only; the enumerator never sees it.
    """
    rows: list[dict[str, object]] = []
    for site in sites.itertuples(index=False):
        if not bool(site.evaluable) or int(str(site.d_before)) <= 0:
            continue
        document_id, engine_id = str(site.document_id), str(site.engine_id)
        span_ids = _id_list(site.ocr_span_ids)
        flank_left = flank_right = None
        if not span_ids:
            index = indexes.get((document_id, engine_id))
            if index is not None:
                for alignment_id in _id_list(site.alignment_ids):
                    flank_left, flank_right = index.flanks_for_empty_component(alignment_id)
                    if flank_left is not None or flank_right is not None:
                        break
        rows.append(
            {
                "true_site_id": str(site.site_id),
                "document_id": document_id,
                "dataset_id": str(site.dataset_id),
                "engine_id": engine_id,
                "site_kind": str(site.site_kind),
                "d_before": int(str(site.d_before)),
                "span_ids": json.dumps(list(span_ids)),
                "flank_left_span": flank_left,
                "flank_right_span": flank_right,
                "char_start": int(str(site.char_start)),
                "char_end": int(str(site.char_end)),
                "gt_text": str(site.gt_text),
                "ocr_text": str(site.ocr_text),
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------ coverage


def _anchor_span_ids(site: DiscoveredSite) -> tuple[str, ...]:
    return tuple(part for part in site.anchor_ref.split("\0")[1:])


def coverage_table(
    discovered: Sequence[DiscoveredSite],
    true_regions: pd.DataFrame,
    site_states: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match discovered anchors against true regions, both directions.

    Returns (per-true-region dispositions, per-discovered-site outcomes). A true region
    is ``discovered`` only on an exact anchor match **of the structurally matching
    kind**: a TOKEN anchor on the same single span, a TOKEN_PAIR anchor on the same
    adjacent pair, or -- for span-less deletion-kind regions -- a GAP anchor on exactly
    the flanking spans. A GAP anchor never covers a two-span region: an insertion
    hypothesis cannot host a merge/pair-substitution repair, so crediting it would
    count regions the pipeline cannot repair (amendment A1; on Track-A, 126 such
    matches held zero beneficial candidates). Softer overlaps are ``boundary_mismatch``,
    counted, never promoted.

    The inherited audit state excludes a region only where the *pairing* is unreliable:
    ``ambiguous`` alignment, or ``unresolved`` for reasons other than the empty
    insertion anchor. Protocol section 8 defines gap-anchor coverage precisely for
    insertion-kind regions -- the CGV2 overlay's ``empty_insertion_anchor`` exclusion
    marked the OCR-side anchor missing, which is the condition CGV3's gap anchor
    exists to satisfy, not a defect in the region itself. Those regions stay eligible.
    """
    state_by_site: dict[str, str] = {}
    if site_states is not None and not site_states.empty:
        has_reason = "reason" in site_states.columns
        for row in site_states.itertuples(index=False):
            state = str(row.state)
            if state == "unresolved" and has_reason and str(row.reason) == "empty_insertion_anchor":
                state = "eligible"
            state_by_site[str(row.site_id)] = state
    discovered_by_span: dict[tuple[str, str], DiscoveredSite] = {}
    discovered_by_token_pair: dict[tuple[str, str, str], DiscoveredSite] = {}
    discovered_by_gap: dict[tuple[str, str, str], DiscoveredSite] = {}
    for site in discovered:
        spans = _anchor_span_ids(site)
        if site.anchor_kind is AnchorKind.TOKEN and spans:
            discovered_by_span.setdefault((site.document_id, spans[0]), site)
        elif site.anchor_kind is AnchorKind.TOKEN_PAIR and len(spans) == 2:
            discovered_by_token_pair.setdefault((site.document_id, *spans), site)
        elif site.anchor_kind is AnchorKind.GAP and len(spans) == 2:
            discovered_by_gap.setdefault((site.document_id, *spans), site)

    true_rows: list[dict[str, object]] = []
    matched_sites: set[str] = set()
    for region in true_regions.itertuples(index=False):
        document_id = str(region.document_id)
        span_ids = _id_list(region.span_ids)
        engine_id = str(region.engine_id)
        state = state_by_site.get(str(region.true_site_id), "eligible")
        match: DiscoveredSite | None = None
        region_start = int(str(region.char_start))
        region_end = int(str(region.char_end))
        if len(span_ids) == 1:
            match = discovered_by_span.get((document_id, span_ids[0]))
        elif len(span_ids) == 2:
            match = discovered_by_token_pair.get((document_id, *span_ids))
        if match is None and not span_ids and region.flank_left_span and region.flank_right_span:
            match = discovered_by_gap.get(
                (document_id, str(region.flank_left_span), str(region.flank_right_span))
            )
        # Any same-page anchor whose stream range touches the region is an imperfect
        # discovery attempt: counted as a boundary mismatch, never promoted to a match.
        # Span-less regions carry the CGV2 [0, 0] char range, where a stream-range test
        # degenerates to "any site at offset 0" (amendment A5); their near-miss test is
        # instead a GAP anchor sharing exactly one flank -- the enumerator pointed at an
        # adjacent boundary, not at the missing region.
        if span_ids:
            overlap = any(
                d.char_start <= region_end and d.char_end >= region_start
                for d in discovered
                if d.document_id == document_id and d.engine_id == engine_id
            )
        else:
            flanks = {
                span
                for span in (region.flank_left_span, region.flank_right_span)
                if span not in (None, "")
            }
            overlap = any(
                len(set(_anchor_span_ids(d)) & flanks) == 1
                for d in discovered
                if d.document_id == document_id
                and d.engine_id == engine_id
                and d.anchor_kind is AnchorKind.GAP
            )

        if state != "eligible":
            disposition = "excluded_" + state
        elif match is not None:
            disposition = "discovered"
            matched_sites.add(match.site_id)
        elif overlap:
            disposition = "boundary_mismatch"
        else:
            disposition = "not_discovered"
        true_rows.append(
            {
                "true_site_id": str(region.true_site_id),
                "document_id": document_id,
                "dataset_id": str(region.dataset_id),
                "engine_id": engine_id,
                "site_kind": str(region.site_kind),
                "d_before": int(str(region.d_before)),
                "audit_state": state,
                "disposition": disposition,
                "matched_site_id": match.site_id if match is not None else "",
            }
        )

    discovered_rows = [
        {
            "site_id": site.site_id,
            "document_id": site.document_id,
            "dataset_id": site.dataset_id,
            "engine_id": site.engine_id,
            "anchor_kind": site.anchor_kind.value,
            "site_type": site.site_type,
            "provenance_reason": site.provenance_reason.value,
            "suspicion_score": site.suspicion_score,
            "char_start": site.char_start,
            "char_end": site.char_end,
            "covers_true_region": site.site_id in matched_sites,
        }
        for site in discovered
    ]
    return pd.DataFrame(true_rows), pd.DataFrame(discovered_rows)


# ---------------------------------------------------------------------- generation


@dataclass(frozen=True, slots=True)
class CandidateRow:
    """One generated candidate at a discovered site, with its offline label."""

    site_id: str
    document_id: str
    dataset_id: str
    engine_id: str
    generator_id: str
    candidate_text: str
    operation: str
    generator_rank: int
    generator_score: float | None
    region_ocr: str
    region_gt: str
    d_before: int
    d_after: int
    outcome: OutcomeIfAccepted

    def as_dict(self) -> dict[str, object]:
        return {
            "site_id": self.site_id,
            "document_id": self.document_id,
            "dataset_id": self.dataset_id,
            "engine_id": self.engine_id,
            "generator_id": self.generator_id,
            "candidate_text": self.candidate_text,
            "operation": self.operation,
            "generator_rank": self.generator_rank,
            "generator_score": self.generator_score,
            "region_ocr": self.region_ocr,
            "region_gt": self.region_gt,
            "d_before": self.d_before,
            "d_after": self.d_after,
            "outcome": self.outcome.value,
        }


def anchor_stream_text(stream: str, site: DiscoveredSite) -> str:
    """The exact stream slice of an anchor -- the deployed view of "what is here".

    ``stream`` is the canonicalizer's linearized text for the page; token offsets
    index into it, so an anchor's slice is well defined for every anchor kind.
    """
    if not 0 <= site.char_start <= site.char_end <= len(stream):
        return ""
    return stream[site.char_start : site.char_end]


def label_candidate(
    region_ocr: str, candidate: str, region_gt: str
) -> tuple[int, int, OutcomeIfAccepted]:
    d_before = distance(region_ocr, region_gt)
    d_after = distance(candidate, region_gt)
    return d_before, d_after, classify_accepted(region_ocr, candidate, region_gt)


def candidates_from_proposals(
    site: DiscoveredSite,
    page: OcrPageView,
    region_ocr: str,
    region_gt: str,
    proposals: Sequence[tuple[str, CandidateProposal, str]],
) -> list[CandidateRow]:
    """Label one site's proposals; identity candidates are dropped at creation."""
    rows: list[CandidateRow] = []
    seen: set[str] = set()
    for generator_id, proposal, operation in proposals:
        if proposal.text == region_ocr or proposal.text in seen:
            continue
        seen.add(proposal.text)
        d_before, d_after, outcome = label_candidate(region_ocr, proposal.text, region_gt)
        rows.append(
            CandidateRow(
                site_id=site.site_id,
                document_id=site.document_id,
                dataset_id=site.dataset_id,
                engine_id=site.engine_id,
                generator_id=generator_id,
                candidate_text=proposal.text,
                operation=operation,
                generator_rank=len(rows),
                generator_score=proposal.score,
                region_ocr=region_ocr,
                region_gt=region_gt,
                d_before=d_before,
                d_after=d_after,
                outcome=outcome,
            )
        )
    return rows
