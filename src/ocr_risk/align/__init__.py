"""Canonical OCR spans to ground truth: anchoring, character DP, relation induction.

Layer 4. Operates on canonical spans only, never on engine adapters, so no engine can
special-case how it is aligned.
"""

from __future__ import annotations

from ocr_risk.align.anchors import Region, RegionPlan, build_regions
from ocr_risk.align.api import AlignmentOutcome, align_document, costs_from_config
from ocr_risk.align.blocks import Block, decompose
from ocr_risk.align.char_dp import CharAlignment, EditCosts, Op, align_chars, parse_confusable_pairs
from ocr_risk.align.components import Component, classify_relation, induce_components
from ocr_risk.align.confidence import (
    ConfidenceWeights,
    component_confidence,
    normalized_distance,
    uniqueness_margin,
)
from ocr_risk.align.diagnostics import AlignmentStats, summarize
from ocr_risk.align.geometry import center_distance, iou, union, vertical_overlap
from ocr_risk.align.streams import Stream, gt_stream, ocr_stream

__all__ = [
    "AlignmentOutcome",
    "AlignmentStats",
    "Block",
    "CharAlignment",
    "Component",
    "ConfidenceWeights",
    "EditCosts",
    "Op",
    "Region",
    "RegionPlan",
    "Stream",
    "align_chars",
    "align_document",
    "build_regions",
    "center_distance",
    "classify_relation",
    "component_confidence",
    "costs_from_config",
    "decompose",
    "gt_stream",
    "induce_components",
    "iou",
    "normalized_distance",
    "ocr_stream",
    "parse_confusable_pairs",
    "summarize",
    "union",
    "uniqueness_margin",
    "vertical_overlap",
]
