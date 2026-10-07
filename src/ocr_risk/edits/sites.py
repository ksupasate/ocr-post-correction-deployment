"""Correction sites: the decision points where PRESERVE or CORRECT is chosen.

Built from alignment components. Two design decisions matter scientifically:

**Clean sites are enumerated.** A site where the OCR is already correct
(``d_before == 0``) is a real decision point, and without those the benchmark cannot
observe overcorrection at all — which would make a reckless method look perfect.

**Adjacent components sharing a ground-truth token are merged into one site.** When an
engine splits "Smith" into "Sm" and "ith", those are not two independent decisions; the
edit that repairs them is a single replacement. Emitting two overlapping sites would
double-count both the opportunity and the harm.
"""

from __future__ import annotations

from collections.abc import Sequence

from ocr_risk.config.models import SiteConfig
from ocr_risk.edits.outcome import distance
from ocr_risk.schemas.alignment import AlignmentRecord
from ocr_risk.schemas.base import BBox
from ocr_risk.schemas.enums import AlignmentRelation, AlignmentStatus, SiteKind
from ocr_risk.schemas.sites import CorrectionSite
from ocr_risk.schemas.spans import CanonicalSpan

__all__ = ["build_sites", "site_kind_for"]


def site_kind_for(relation: AlignmentRelation, d_before: int, evaluable: bool) -> SiteKind:
    """Classify what kind of discrepancy a site represents."""
    if not evaluable:
        return SiteKind.EXCLUDED
    if d_before == 0:
        return SiteKind.CLEAN
    if relation is AlignmentRelation.OCR_INSERTION:
        return SiteKind.INSERTION
    if relation is AlignmentRelation.OCR_DELETION:
        return SiteKind.DELETION
    if relation in (
        AlignmentRelation.SPLIT,
        AlignmentRelation.MERGE,
        AlignmentRelation.MANY_TO_MANY,
    ):
        return SiteKind.SEGMENTATION
    return SiteKind.SUBSTITUTION


def build_sites(
    alignments: Sequence[AlignmentRecord],
    spans: Sequence[CanonicalSpan],
    config: SiteConfig,
) -> list[CorrectionSite]:
    """Turn one (document, engine) pair's alignment components into correction sites."""
    if not alignments:
        return []

    span_by_id = {span.span_id: span for span in spans}
    groups = _merge_components(alignments, config)

    sites: list[CorrectionSite] = []
    for index, group in enumerate(groups):
        first = group[0]
        ocr_text = " ".join(r.ocr_text for r in group if r.ocr_text)
        gt_text = " ".join(r.gt_text for r in group if r.gt_text)
        span_ids = tuple(sid for r in group for sid in r.ocr_span_ids)
        token_ids = tuple(tid for r in group for tid in r.gt_token_ids)

        evaluable = all(r.status is AlignmentStatus.RESOLVED for r in group)
        if config.max_site_chars and max(len(ocr_text), len(gt_text)) > config.max_site_chars:
            # An over-long site is usually an alignment failure wearing a site's clothes.
            # Carry it so the volume is reportable, but do not evaluate it.
            evaluable = False

        d_before = distance(ocr_text, gt_text)
        kind = site_kind_for(first.relation, d_before, evaluable)
        if kind is SiteKind.CLEAN and not config.include_clean_sites:
            continue

        group_spans = [span_by_id[sid] for sid in span_ids if sid in span_by_id]
        boxes = [s.bbox for s in group_spans if s.bbox is not None]
        char_start = min((s.char_start for s in group_spans), default=0)
        char_end = max((s.char_end for s in group_spans), default=0)

        sites.append(
            CorrectionSite(
                site_id=f"{first.document_id}:{first.engine_id}:site:{index:05d}",
                document_id=first.document_id,
                dataset_id=first.dataset_id,
                engine_id=first.engine_id,
                alignment_ids=tuple(r.alignment_id for r in group),
                ocr_span_ids=span_ids,
                gt_token_ids=token_ids,
                ocr_text=ocr_text,
                gt_text=gt_text,
                d_before=d_before,
                site_kind=kind,
                evaluable=evaluable,
                char_start=char_start,
                char_end=char_end,
                reading_order_start=min((s.reading_order for s in group_spans), default=0),
                bbox=_union(boxes),
                min_align_confidence=min(r.align_confidence for r in group),
            )
        )
    return sites


def _merge_components(
    alignments: Sequence[AlignmentRecord], config: SiteConfig
) -> list[list[AlignmentRecord]]:
    """Group components that describe one editable region.

    Components are merged when they are adjacent in reading order *and* one of them is a
    segmentation failure, because that is the case where a single replacement repairs
    both. Unrelated neighbours stay separate: merging them would fabricate a larger edit
    than any generator would propose.
    """
    ordered = sorted(
        alignments,
        key=lambda r: (
            r.gt_token_ids[0] if r.gt_token_ids else "~",
            r.ocr_span_ids[0] if r.ocr_span_ids else "~",
        ),
    )
    if not config.merge_adjacent_components:
        return [[record] for record in ordered]

    groups: list[list[AlignmentRecord]] = []
    for record in ordered:
        if groups and _should_merge(groups[-1][-1], record):
            groups[-1].append(record)
        else:
            groups.append([record])
    return groups


_SEGMENTATION = (
    AlignmentRelation.SPLIT,
    AlignmentRelation.MERGE,
    AlignmentRelation.MANY_TO_MANY,
)


def _should_merge(previous: AlignmentRecord, current: AlignmentRecord) -> bool:
    """Whether two adjacent components belong to the same editable region."""
    if previous.relation not in _SEGMENTATION and current.relation not in _SEGMENTATION:
        return False
    # An insertion or deletion sitting between segmentation components is part of the same
    # damaged region; a clean one-to-one neighbour is not.
    shared_tokens = set(previous.gt_token_ids) & set(current.gt_token_ids)
    return bool(shared_tokens)


def _union(boxes: Sequence[BBox]) -> BBox | None:
    if not boxes:
        return None
    return BBox(
        x0=min(b.x0 for b in boxes),
        y0=min(b.y0 for b in boxes),
        x1=max(b.x1 for b in boxes),
        y1=max(b.y1 for b in boxes),
    )
