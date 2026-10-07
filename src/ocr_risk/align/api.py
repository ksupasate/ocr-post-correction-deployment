"""Public alignment entry point: canonical OCR spans to ground-truth tokens.

Composes the three stages — region anchoring, character alignment, component induction —
and attaches the confidence and status that decide whether a component may be evaluated.

The design commitment throughout: **never force a match to raise coverage.** A component
the aligner cannot resolve is written out as ``AMBIGUOUS`` or ``UNRESOLVED`` with a reason
code and counted in the statistics, because per-engine ambiguity rate is itself a confound
for the cross-engine comparison and has to be visible in the results table.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rapidfuzz.distance import Levenshtein

from ocr_risk.align.anchors import build_regions
from ocr_risk.align.blocks import decompose
from ocr_risk.align.char_dp import CharAlignment, EditCosts, Op, align_chars, parse_confusable_pairs
from ocr_risk.align.components import Component, induce_components, orphan_components
from ocr_risk.align.confidence import (
    ConfidenceWeights,
    component_confidence,
    uniqueness_margin,
)
from ocr_risk.align.geometry import iou, union
from ocr_risk.align.streams import Stream, gt_stream, ocr_stream
from ocr_risk.config.models import AlignmentConfig
from ocr_risk.schemas.alignment import AlignmentRecord
from ocr_risk.schemas.documents import GtToken, SourceDocument
from ocr_risk.schemas.enums import AlignmentRelation, AlignmentStatus
from ocr_risk.schemas.spans import CanonicalSpan

__all__ = ["AlignmentOutcome", "align_document", "costs_from_config"]


@dataclass(slots=True)
class AlignmentOutcome:
    """Alignment records for one (document, engine) pair, plus why anything failed."""

    records: list[AlignmentRecord]
    used_geometry: bool
    n_unresolved_blocks: int = 0


def costs_from_config(config: AlignmentConfig) -> EditCosts:
    return EditCosts(
        substitute=config.substitution_cost,
        insert=config.insertion_cost,
        delete=config.deletion_cost,
        confusable=config.confusable_substitution_cost,
        blocks=parse_confusable_pairs(config.confusable_pairs),
        max_cells=config.max_block_chars * config.max_block_chars,
    )


def align_document(
    document: SourceDocument,
    spans: Sequence[CanonicalSpan],
    tokens: Sequence[GtToken],
    config: AlignmentConfig,
) -> AlignmentOutcome:
    """Align one engine's reading of one page against that page's ground truth."""
    ordered_spans = sorted(spans, key=lambda s: s.reading_order)
    ordered_tokens = sorted(tokens, key=lambda t: t.index)
    costs = costs_from_config(config)
    weights = ConfidenceWeights(
        char_agreement=config.weight_char_agreement,
        geometry=config.weight_geometry,
        uniqueness=config.weight_uniqueness,
    )

    plan = build_regions(
        ordered_spans,
        ordered_tokens,
        use_geometry=config.use_geometry,
        max_anchor_cost=config.max_anchor_cost,
        page_width=float(document.width),
        page_height=float(document.height),
    )

    components: list[Component] = []
    region_of: dict[int, str] = {}
    unresolved_blocks = 0

    for region in plan.regions:
        region_spans = [ordered_spans[i] for i in region.span_indices]
        region_tokens = [ordered_tokens[j] for j in region.token_indices]
        ocr = ocr_stream(region_spans)
        gt = gt_stream(region_tokens)

        alignment, failed, refused_ocr, refused_gt = _align_region(ocr, gt, costs, config)
        unresolved_blocks += failed

        region_components = induce_components(alignment, ocr, gt)
        for component in region_components:
            translated = Component(
                span_indices=tuple(region.span_indices[i] for i in component.span_indices),
                token_indices=tuple(region.token_indices[j] for j in component.token_indices),
                relation=component.relation,
                matched_chars=component.matched_chars,
                ocr_chars=component.ocr_chars,
                gt_chars=component.gt_chars,
                origin=(
                    "refused_block"
                    if _from_refused_block(component, ocr, gt, refused_ocr, refused_gt)
                    else "in_region"
                ),
            )
            region_of[len(components)] = region.region_id
            components.append(translated)

    components.extend(
        orphan_components(
            plan.unmatched_span_indices,
            plan.unmatched_token_indices,
            [len(ordered_spans[i].text) for i in plan.unmatched_span_indices],
        )
    )

    records = _to_records(
        document=document,
        spans=ordered_spans,
        tokens=ordered_tokens,
        components=components,
        region_of=region_of,
        config=config,
        weights=weights,
        used_geometry=plan.used_geometry,
        unresolved_blocks=unresolved_blocks,
    )
    return AlignmentOutcome(
        records=records, used_geometry=plan.used_geometry, n_unresolved_blocks=unresolved_blocks
    )


def _align_region(
    ocr: Stream, gt: Stream, costs: EditCosts, config: AlignmentConfig
) -> tuple[CharAlignment, int, list[tuple[int, int]], list[tuple[int, int]]]:
    """Align one region, decomposing it first when it is too large for an exact DP.

    Returns the alignment, the number of refused blocks, and the OCR and ground-truth
    character ranges those blocks covered, so their components can be marked UNRESOLVED
    rather than surfacing as confident one-sided findings.
    """
    if len(ocr.text) * len(gt.text) <= costs.max_cells:
        return align_chars(ocr.text, gt.text, costs), 0, [], []

    blocks = decompose(
        ocr.text,
        gt.text,
        min_anchor_length=config.anchor_min_length,
        max_block_cells=costs.max_cells,
    )
    combined = CharAlignment(ocr_length=len(ocr.text), gt_length=len(gt.text))
    failed = 0
    refused_ocr: list[tuple[int, int]] = []
    refused_gt: list[tuple[int, int]] = []
    for block in blocks:
        piece_ocr = ocr.text[block.ocr_start : block.ocr_end]
        piece_gt = gt.text[block.gt_start : block.gt_end]
        if block.anchored:
            combined.ops.append(
                (Op.MATCH, block.ocr_start, block.ocr_end, block.gt_start, block.gt_end)
            )
            continue
        if block.cells > costs.max_cells:
            # Refuse rather than align badly. The character ranges are reported back so
            # the components they produce can be tagged and marked UNRESOLVED: without
            # that they are pure one-sided components, indistinguishable from a genuine
            # hallucination or omission, and were being reported as resolved findings.
            failed += 1
            refused_ocr.append((block.ocr_start, block.ocr_end))
            refused_gt.append((block.gt_start, block.gt_end))
            if piece_ocr:
                combined.ops.append(
                    (Op.INSERT, block.ocr_start, block.ocr_end, block.gt_start, block.gt_start)
                )
            if piece_gt:
                combined.ops.append(
                    (Op.DELETE, block.ocr_end, block.ocr_end, block.gt_start, block.gt_end)
                )
            continue
        piece = align_chars(piece_ocr, piece_gt, costs)
        for op, a0, a1, b0, b1 in piece.ops:
            combined.ops.append(
                (
                    op,
                    a0 + block.ocr_start,
                    a1 + block.ocr_start,
                    b0 + block.gt_start,
                    b1 + block.gt_start,
                )
            )
        combined.cost += piece.cost
    return combined, failed, refused_ocr, refused_gt


def _from_refused_block(
    component: Component,
    ocr: Stream,
    gt: Stream,
    refused_ocr: Sequence[tuple[int, int]],
    refused_gt: Sequence[tuple[int, int]],
) -> bool:
    """Whether every character of this component came from a block the DP refused."""
    if not refused_ocr and not refused_gt:
        return False

    def _covered(positions: list[int], ranges: Sequence[tuple[int, int]]) -> bool:
        return bool(positions) and all(
            any(start <= p < end for start, end in ranges) for p in positions
        )

    ocr_positions = [i for i, owner in enumerate(ocr.owner) if owner in component.span_indices]
    gt_positions = [j for j, owner in enumerate(gt.owner) if owner in component.token_indices]
    if ocr_positions and not _covered(ocr_positions, refused_ocr):
        return False
    if gt_positions and not _covered(gt_positions, refused_gt):
        return False
    return bool(ocr_positions or gt_positions)


def _to_records(
    *,
    document: SourceDocument,
    spans: Sequence[CanonicalSpan],
    tokens: Sequence[GtToken],
    components: Sequence[Component],
    region_of: dict[int, str],
    config: AlignmentConfig,
    weights: ConfidenceWeights,
    used_geometry: bool,
    unresolved_blocks: int,
) -> list[AlignmentRecord]:
    # Alternatives for the uniqueness margin: every ground-truth token in the page that
    # this component did not claim. A near-tie against any of them means the aligner had
    # a real choice to make.
    all_gt_texts = [t.text for t in tokens]

    records: list[AlignmentRecord] = []
    for index, component in enumerate(components):
        component_spans = [spans[i] for i in component.span_indices]
        component_tokens = [tokens[j] for j in component.token_indices]
        ocr_text = " ".join(s.text for s in component_spans)
        gt_text = " ".join(t.text for t in component_tokens)

        boxes = [s.bbox for s in component_spans if s.bbox is not None]
        gt_boxes = [t.bbox for t in component_tokens if t.bbox is not None]
        ocr_union = union(boxes)
        gt_union = union(gt_boxes)
        geom_score = iou(ocr_union, gt_union) if used_geometry and ocr_union and gt_union else None

        claimed = set(component.token_indices)
        alternatives = [text for j, text in enumerate(all_gt_texts) if j not in claimed]
        margin = uniqueness_margin(ocr_text, gt_text, alternatives) if ocr_text and gt_text else 1.0

        confidence, diagnostics = component_confidence(
            component.char_agreement, geom_score, margin, weights
        )
        status, reason = _status_for(component, confidence, config, unresolved_blocks)
        diagnostics["relation_shape"] = (
            f"{len(component.span_indices)}:{len(component.token_indices)}"
        )
        if not used_geometry:
            diagnostics["anchoring"] = "sequential_no_gt_geometry"

        engine_id = (
            component_spans[0].engine_id
            if component_spans
            else spans[0].engine_id
            if spans
            else "unknown"
        )
        records.append(
            AlignmentRecord(
                alignment_id=f"{document.document_id}:{engine_id}:al:{index:05d}",
                document_id=document.document_id,
                dataset_id=document.dataset_id,
                engine_id=engine_id,
                ocr_span_ids=tuple(s.span_id for s in component_spans),
                gt_token_ids=tuple(t.gt_token_id for t in component_tokens),
                relation=component.relation,
                status=status,
                ocr_text=ocr_text,
                gt_text=gt_text,
                align_confidence=confidence,
                char_agreement=component.char_agreement,
                geom_score=geom_score,
                uniqueness_margin=margin,
                # Raw character distance. Recovering it by multiplying the normalized
                # value back out would round-trip through a division for no reason.
                edit_distance=Levenshtein.distance(ocr_text, gt_text),
                region_id=region_of.get(index),
                bbox=ocr_union,
                reason_code=reason,
                diagnostics=diagnostics,
            )
        )
    return records


def _status_for(
    component: Component,
    confidence: float,
    config: AlignmentConfig,
    unresolved_blocks: int,
) -> tuple[AlignmentStatus, str | None]:
    """Decide whether a component may be evaluated, and say why if not."""
    # A refused block aligned nothing, so nothing it produced is evidence. Checked first,
    # and keyed on the component's own origin rather than on a document-wide counter: the
    # counter fired on innocent components elsewhere on the page and never on the guilty
    # ones, because those exited early as "resolved" insertions.
    if component.origin == "refused_block":
        return AlignmentStatus.UNRESOLVED, "block_exceeded_dp_limit"

    one_sided = (
        component.relation is AlignmentRelation.OCR_INSERTION and not component.token_indices
    ) or (component.relation is AlignmentRelation.OCR_DELETION and not component.span_indices)

    if one_sided and component.origin == "orphan":
        # Never anchored, so this says nothing about the engine — only that anchoring
        # failed to place the item. Excluded from evaluation and counted, rather than
        # reported as a hallucination or an omission the engine did not commit.
        return AlignmentStatus.OUT_OF_REGION, "never_anchored"

    if one_sided:
        # Found by the DP inside an anchored region: positive evidence, and a resolved
        # finding rather than an alignment failure.
        relation_name = (
            "ocr_insertion"
            if component.relation is AlignmentRelation.OCR_INSERTION
            else "ocr_deletion"
        )
        return AlignmentStatus.RESOLVED, relation_name

    if unresolved_blocks and component.char_agreement == 0.0:
        return AlignmentStatus.UNRESOLVED, "block_exceeded_dp_limit"
    if confidence < config.min_align_confidence:
        return AlignmentStatus.AMBIGUOUS, "below_confidence_floor"
    return AlignmentStatus.RESOLVED, None
