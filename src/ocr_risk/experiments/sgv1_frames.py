"""SGV1 dual-frame construction, with the freeze order enforced in code.

The dual-frame design (docs/sgv1/protocol.md) has one non-negotiable ordering:

    OCR -> sites -> candidates -> FREEZE -> labels -> frames

Candidates are generated GT-blind and frozen (hash-bound) before any ground truth is
attached; only then may labels drive stratified Frame A sampling. Every function here
that consumes labels re-verifies the freeze binding first, so "sampling before freeze"
and "labels drifting onto an unfrozen table" are exceptions, not discipline failures.

Frame B is the natural stream: every frozen candidate, no resampling, no class
balancing. Frame A is the stratified verification benchmark: class composition is a
design parameter for observability and is **never** a prevalence statement -- every row
carries ``sampling_weight`` so population-weighted secondary analyses stay possible.

Ground truth reaches this module only through the labels table (candidate_id, outcome,
is_harmful, ...). The candidates table itself is checked against GT-looking columns and
rejected if any are present.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ocr_risk.experiments.sgv1_design import (
    FRAME_NATURAL,
    VALID_FRAMES,
)
from ocr_risk.schemas.evidence import field_names_leaking_ground_truth

__all__ = [
    "CANDIDATE_KEY_COLUMNS",
    "FRAME_A_CLASS_COLUMNS",
    "FRAME_CLASS_BENEFICIAL",
    "FRAME_CLASS_HARMFUL",
    "FRAME_CLASS_NEUTRAL",
    "FRAME_CLASS_UNRESOLVED",
    "OUTCOME_UNRESOLVED",
    "CandidateFreeze",
    "FrameError",
    "attach_labels",
    "build_frame_a",
    "build_frame_b",
    "build_matched_pairs",
    "evaluation_class",
    "freeze_candidates",
    "label_columns_present",
    "verify_frozen",
]

CANDIDATE_KEY_COLUMNS: tuple[str, ...] = (
    "candidate_id",
    "site_id",
    "document_id",
    "engine_id",
    "generator_id",
)
LABEL_COLUMNS: tuple[str, ...] = ("candidate_id", "outcome", "is_harmful")
_ALLOWED_LABEL_COLUMNS: frozenset[str] = frozenset(
    {
        "candidate_id",
        "outcome",
        "is_harmful",
        "d_before",
        "d_after",
        "labelable",
        "region_gt_status",
        "region_is_whitespace_only",
    }
)
FRAME_A_CLASS_COLUMNS: tuple[str, ...] = (
    *CANDIDATE_KEY_COLUMNS,
    "outcome",
    "is_harmful",
    "evaluation_class",
    "source_population_stratum",
    "evaluation_stratum",
    "sampling_weight",
)

FRAME_CLASS_BENEFICIAL = "beneficial"
FRAME_CLASS_HARMFUL = "harmful"
FRAME_CLASS_NEUTRAL = "neutral"
FRAME_CLASS_UNRESOLVED = "unresolved"
"""No ground truth was established for the region, so no outcome exists.

Distinct from ``neutral`` on purpose. Neutral means the edit was measured and found not
to help; unresolved means it was never measurable. Folding the second into the first
would silently convert absent evidence into evidence of no effect."""

OUTCOME_UNRESOLVED = "unresolved"
"""The ``outcome`` value a candidate carries when its region ground truth is not
established. It is not a member of the outcome taxonomy in ``edits/outcome.py``: it
records the absence of a label rather than a kind of label."""

_EVALUATION_CLASSES = frozenset(
    {
        FRAME_CLASS_BENEFICIAL,
        FRAME_CLASS_HARMFUL,
        FRAME_CLASS_NEUTRAL,
        FRAME_CLASS_UNRESOLVED,
    }
)
_SAMPLED_CLASSES: tuple[str, ...] = (FRAME_CLASS_BENEFICIAL, FRAME_CLASS_HARMFUL)

_BENEFICIAL_OUTCOMES = frozenset({"true_correction", "partial_improvement"})
_HARMFUL_OUTCOMES = frozenset({"miscorrection", "overcorrection"})


class FrameError(RuntimeError):
    """Raised when frame construction would violate the freeze order or GT blindness."""


@dataclass(frozen=True, slots=True)
class CandidateFreeze:
    """The hash binding between a frozen candidate table and everything downstream."""

    candidates_sha256: str
    n_rows: int
    n_documents: int
    generator_ids: tuple[str, ...]
    frame: str


def label_columns_present(columns: Iterable[str]) -> list[str]:
    """Columns that carry ground truth or labels; must never appear on candidates."""
    offenders = [
        c
        for c in columns
        if c
        in {
            "outcome",
            "is_harmful",
            "d_before",
            "d_after",
            "labelable",
            "region_gt_status",
            "region_is_whitespace_only",
        }
    ]
    return list(dict.fromkeys([*offenders, *field_names_leaking_ground_truth(columns)]))


def _canonical_payload(frame_table: pd.DataFrame) -> str:
    record = {
        "columns": list(frame_table.columns),
        "rows": frame_table.to_dict(orient="records"),
    }
    return json.dumps(record, sort_keys=True, default=str)


def freeze_candidates(candidates: pd.DataFrame, *, frame: str) -> CandidateFreeze:
    """Freeze a GT-blind candidate table: validate, hash, and record.

    The hash covers columns and rows exactly as given, so the freeze binds this table
    and this construction, byte-for-byte semantics rather than set-semantics: a
    different row order is a different frozen table, never silently the same one.
    """
    if frame not in VALID_FRAMES:
        msg = f"unknown frame {frame!r}; known: {sorted(VALID_FRAMES)}"
        raise FrameError(msg)
    missing = [c for c in CANDIDATE_KEY_COLUMNS if c not in candidates.columns]
    if missing:
        msg = f"candidate table missing key columns {missing}"
        raise FrameError(msg)
    leaking = label_columns_present(candidates.columns)
    if leaking:
        msg = (
            f"candidate table carries ground-truth/label columns {leaking}; candidates "
            "must be GT-blind before the freeze (docs/sgv1/protocol.md section 4)"
        )
        raise FrameError(msg)
    digest = hashlib.sha256(_canonical_payload(candidates).encode("utf-8")).hexdigest()
    return CandidateFreeze(
        candidates_sha256=digest,
        n_rows=len(candidates),
        n_documents=int(candidates["document_id"].nunique()),
        generator_ids=tuple(sorted(candidates["generator_id"].astype(str).unique())),
        frame=frame,
    )


def verify_frozen(candidates: pd.DataFrame, freeze: CandidateFreeze) -> None:
    """Re-derive the freeze hash and refuse to proceed on a different table."""
    digest = hashlib.sha256(_canonical_payload(candidates).encode("utf-8")).hexdigest()
    if digest != freeze.candidates_sha256:
        msg = (
            "candidate table does not match its freeze record "
            f"({digest[:12]}... != {freeze.candidates_sha256[:12]}...); labels, frames, "
            "and verifiers may only consume the exact frozen table"
        )
        raise FrameError(msg)


def evaluation_class(outcome: str) -> str:
    """Map an accepted-outcome label to the Frame A evaluation class.

    Mirrors the outcome taxonomy in ``edits/outcome.py`` by name; the classes are the
    protocol's (section 5), not a new taxonomy: beneficial = true correction or partial
    improvement, harmful = miscorrection or overcorrection, everything else neutral.
    """
    if outcome in _BENEFICIAL_OUTCOMES:
        return FRAME_CLASS_BENEFICIAL
    if outcome in _HARMFUL_OUTCOMES:
        return FRAME_CLASS_HARMFUL
    if outcome == OUTCOME_UNRESOLVED:
        return FRAME_CLASS_UNRESOLVED
    return FRAME_CLASS_NEUTRAL


def attach_labels(
    candidates: pd.DataFrame, freeze: CandidateFreeze, labels: pd.DataFrame
) -> pd.DataFrame:
    """Join labels onto the frozen candidate table. Freeze order is enforced here.

    This is the single point where ground truth enters the SGV1 pipeline, and it is
    only reachable on a table whose hash matches the freeze record.
    """
    verify_frozen(candidates, freeze)
    missing = [c for c in LABEL_COLUMNS if c not in labels.columns]
    if missing:
        msg = f"labels table missing columns {missing}"
        raise FrameError(msg)
    leaked = label_columns_present(candidates.columns)
    if leaked:
        msg = f"refusing to attach labels onto a table that already has {leaked}"
        raise FrameError(msg)
    if labels["candidate_id"].duplicated().any():
        msg = "labels table has duplicate candidate_id rows"
        raise FrameError(msg)
    merged = candidates.merge(
        labels[[c for c in labels.columns if c in _ALLOWED_LABEL_COLUMNS]],
        on="candidate_id",
        how="left",
        validate="one_to_one",
    )
    unlabeled = merged["outcome"].isna().sum()
    if unlabeled:
        msg = f"{unlabeled} frozen candidates have no label; never silently drop them"
        raise FrameError(msg)
    return merged


def build_frame_a(
    labeled: pd.DataFrame,
    freeze: CandidateFreeze,
    *,
    per_document_cap: int,
    min_per_class: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Stratified verification benchmark (Frame A) from the labeled frozen pool.

    Document-stratified: within each evaluation class, at most ``per_document_cap``
    candidates per document, so no document can dominate a class. Inclusion
    probabilities are recorded as ``sampling_weight`` (stratum population / sampled),
    which is what makes a population-weighted secondary analysis possible and an
    unweighted one explicitly a discrimination question.

    Returns the benchmark table and the per-class population counts of the source
    pool -- including neutral and unresolved, which Frame A excludes from the risk set
    but whose sizes the record must state. An unresolved candidate has no ground truth
    for its region, so it is not a negative example and not a neutral one: it is not an
    example at all, and benchmarking a verifier against it would be scoring a guess
    against a label that was never established. The benchmark's composition is a design
    parameter and must never be readable as prevalence.

    ``min_per_class`` is the design floor passed by the caller -- the protocol's class
    observability requirement, never a default hidden here.
    """
    verify_frozen(labeled.drop(columns=_label_only_columns(labeled)), freeze)
    if per_document_cap < 1:
        msg = "per_document_cap must be >= 1"
        raise FrameError(msg)
    merged = labeled.assign(evaluation_class=labeled["outcome"].map(evaluation_class))
    unknown = set(merged["evaluation_class"]) - _EVALUATION_CLASSES
    if unknown:
        msg = f"unknown evaluation classes {sorted(unknown)}"
        raise FrameError(msg)

    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    class_populations: dict[str, int] = {}
    for evaluation_class_name in (
        FRAME_CLASS_BENEFICIAL,
        FRAME_CLASS_HARMFUL,
        FRAME_CLASS_NEUTRAL,
        FRAME_CLASS_UNRESOLVED,
    ):
        stratum = merged[merged["evaluation_class"] == evaluation_class_name]
        class_populations[evaluation_class_name] = len(stratum)
        if evaluation_class_name not in _SAMPLED_CLASSES:
            continue
        if len(stratum) < min_per_class:
            msg = (
                f"Frame A class {evaluation_class_name!r} has {len(stratum)} candidates, "
                f"below the required minimum {min_per_class}; the design is underpowered, "
                "not silently salvageable by relaxing the floor"
            )
            raise FrameError(msg)
        by_doc = stratum.groupby("document_id")
        kept: list[pd.DataFrame] = []
        for _, document_rows in by_doc:
            if len(document_rows) > per_document_cap:
                kept.append(
                    document_rows.sample(
                        n=per_document_cap, random_state=int(rng.integers(0, 2**31 - 1))
                    )
                )
            else:
                kept.append(document_rows)
        sampled = pd.concat(kept, ignore_index=True)
        sampled = sampled.assign(
            source_population_stratum=f"{freeze.frame}:{evaluation_class_name}",
            evaluation_stratum=evaluation_class_name,
            sampling_weight=len(stratum) / len(sampled),
        )
        parts.append(sampled)

    frame_a = pd.concat(parts, ignore_index=True)
    table = frame_a.assign(frame=freeze.frame)[[*FRAME_A_CLASS_COLUMNS, "frame"]]
    return table, class_populations


def _label_only_columns(frame_table: pd.DataFrame) -> list[str]:
    return [
        c
        for c in (
            "outcome",
            "is_harmful",
            "evaluation_class",
            "d_before",
            "d_after",
            "labelable",
            "region_gt_status",
            "region_is_whitespace_only",
        )
        if c in frame_table.columns
    ]


def build_matched_pairs(frame_a: pd.DataFrame) -> pd.DataFrame:
    """Within-site beneficial/harmful pairs for the paired ranking endpoint.

    Both members come from the frozen pool by construction (Frame A rows are frozen
    candidates); a pair is every (beneficial, harmful) combination at a site, emitted
    deterministically in sorted order. The paired endpoint P(q(Y+) > q(Y-)) controls for
    everything the site shares: image, engine, document, layout, OCR quality.
    """
    needed = {"site_id", "document_id", "engine_id", "candidate_id", "evaluation_class"}
    missing = needed - set(frame_a.columns)
    if missing:
        msg = f"frame A table missing columns {sorted(missing)}"
        raise FrameError(msg)
    rows: list[dict[str, str]] = []
    for group_keys, group in frame_a.groupby(["site_id", "document_id", "engine_id"], sort=True):
        site_id, document_id, engine_id = (str(k) for k in group_keys)
        beneficial = sorted(
            group.loc[group["evaluation_class"] == FRAME_CLASS_BENEFICIAL, "candidate_id"]
        )
        harmful = sorted(
            group.loc[group["evaluation_class"] == FRAME_CLASS_HARMFUL, "candidate_id"]
        )
        for plus in beneficial:
            for minus in harmful:
                rows.append(
                    {
                        "site_id": site_id,
                        "document_id": document_id,
                        "engine_id": engine_id,
                        "plus_candidate_id": plus,
                        "minus_candidate_id": minus,
                    }
                )
    return pd.DataFrame(
        rows,
        columns=["site_id", "document_id", "engine_id", "plus_candidate_id", "minus_candidate_id"],
    )


def build_frame_b(labeled: pd.DataFrame, freeze: CandidateFreeze) -> pd.DataFrame:
    """The natural deployment stream (Frame B): every frozen candidate, unmodified.

    No stratification, no resampling, no class balancing -- reading prevalence off this
    table is the point. Labels are attached (they exist post-freeze) but influence
    nothing about its composition.
    """
    if freeze.frame != FRAME_NATURAL:
        msg = f"Frame B is the natural stream; freeze frame is {freeze.frame!r}"
        raise FrameError(msg)
    verify_frozen(labeled.drop(columns=_label_only_columns(labeled)), freeze)
    # The evaluation class is derived here rather than by the caller so that Frame A and
    # Frame B can never disagree about which outcome belongs to which class.
    return labeled.assign(
        evaluation_class=labeled["outcome"].map(evaluation_class),
        frame=FRAME_NATURAL,
        sampling_weight=1.0,
    )
