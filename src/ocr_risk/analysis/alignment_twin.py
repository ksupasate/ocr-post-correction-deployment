"""The CGV2 alignment twin audit (protocol §15) — Views A and B.

View A audits the OCR-to-GT structure: how many evaluable sites of each kind carry the
R-37 twin signature — an exact-text match, on the same page, to a ground-truth token the
aligner never anchored — and how the adjacent-pair population divides into eligible,
degenerate, ambiguous, and non-adjacent regions.

View B audits candidate-to-target projection: whether the region candidates the study
produced are attached to the region they claim, verified against the canonical sites
table rather than against the study's own bookkeeping.

This module reads immutable artifacts and computes counts. It does not relabel anything,
and it is the committed producer R-37 never had: the recovery phase measured the twin
signature in scripts and reported it in prose.
"""

from __future__ import annotations

import json
import unicodedata
from collections import defaultdict
from itertools import pairwise

import pandas as pd

from ocr_risk.align.confidence import (
    ConfidenceWeights,
    component_confidence,
    uniqueness_margin,
)
from ocr_risk.edits.outcome import classify_accepted, distance

__all__ = [
    "component_audit",
    "component_denominators",
    "projection_audit",
    "projection_summary",
    "region_census",
    "region_pair_audit",
    "site_eligibility_audit",
    "site_eligibility_denominators",
    "site_projection_audit",
    "twin_sites_table",
    "twin_table",
]


def _ids(value: object) -> tuple[str, ...]:
    """Normalize Arrow lists and JSON-in-CSV lists to a comparable tuple."""
    if value is None:
        return ()
    if isinstance(value, str):
        if not value:
            return ()
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return (value,)
        if isinstance(decoded, list):
            return tuple(str(item) for item in decoded)
        return (str(decoded),)
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list | tuple | set | frozenset):
        return tuple(str(item) for item in value)
    return (str(value),)


def _punctuation_boundary(*texts: str) -> bool:
    for text in texts:
        stripped = text.strip()
        if stripped and any(
            unicodedata.category(char).startswith("P") for char in (stripped[0], stripped[-1])
        ):
            return True
    return False


def _component_operation(n_ocr: int, n_gt: int) -> str:
    if n_ocr == 0 and n_gt > 0:
        return "insertion"
    if n_ocr > 0 and n_gt == 0:
        return "deletion"
    if n_ocr == 1 and n_gt == 1:
        return "substitution"
    if n_ocr == 1 and n_gt > 1:
        return "1_to_n_split"
    if n_ocr > 1 and n_gt == 1:
        return "n_to_1_merge"
    if n_ocr > 1 and n_gt > 1:
        return "n_to_m_mixed"
    return "empty_component"


def _component_state(status: str) -> str:
    return {
        "resolved": "eligible",
        "ambiguous": "ambiguous",
        "unresolved": "unresolved",
        "out_of_region": "excluded",
    }.get(status, "unresolved")


def _overlapping_interval_ids(
    intervals: dict[str, tuple[str, str, int, int]],
) -> dict[str, bool]:
    """Mark strict interval overlaps with one document-scoped sweep.

    Candidate spans on a page are nearly disjoint, so the active set stays tiny. This is
    linear after sorting instead of comparing every site in the corpus with every other
    site (the first audit implementation was accidentally quadratic).
    """
    grouped: dict[tuple[str, str], list[tuple[int, int, str]]] = defaultdict(list)
    for item_id, (document_id, engine_id, start, end) in intervals.items():
        grouped[(document_id, engine_id)].append((start, end, item_id))
    overlaps = dict.fromkeys(intervals, False)
    for group in grouped.values():
        active: list[tuple[int, str]] = []
        for start, end, item_id in sorted(group):
            active = [
                (active_end, active_id) for active_end, active_id in active if active_end > start
            ]
            if end > start and active:
                overlaps[item_id] = True
                for _active_end, active_id in active:
                    overlaps[active_id] = True
            if end > start:
                active.append((end, item_id))
    return overlaps


def component_audit(
    alignments: pd.DataFrame,
    gt_tokens: pd.DataFrame | None = None,
    *,
    min_align_confidence: float = 0.55,
    weights: ConfidenceWeights | None = None,
) -> pd.DataFrame:
    """View A, one row per OCR↔GT component with shape and explicit disposition.

    When the manifest token table is supplied, the uniqueness margin is recomputed
    against *every* other GT token on the page.  The historical aligner considered only
    the first 64 alternatives, so a near-tie later in reading order could be silently
    resolved.  This audit never rewrites the frozen alignment artifact: it records both
    values and conservatively changes the audit state to ``ambiguous`` when the complete
    margin moves a previously resolved component below the frozen confidence floor.
    """
    required = {
        "alignment_id",
        "document_id",
        "engine_id",
        "ocr_span_ids",
        "gt_token_ids",
        "status",
        "ocr_text",
        "gt_text",
    }
    missing = sorted(required - set(alignments.columns))
    if missing:
        raise ValueError(f"alignments table is missing required columns: {', '.join(missing)}")
    token_texts: dict[str, list[tuple[str, str]]] = {}
    if gt_tokens is not None and not gt_tokens.empty:
        token_required = {"gt_token_id", "document_id", "text"}
        token_missing = sorted(token_required - set(gt_tokens.columns))
        if token_missing:
            raise ValueError(
                "gt_tokens table is missing required columns: " + ", ".join(token_missing)
            )
        ordered = (
            gt_tokens.sort_values(["document_id", "index"])
            if "index" in gt_tokens.columns
            else gt_tokens.sort_values(["document_id", "gt_token_id"])
        )
        for token in ordered.itertuples(index=False):
            token_texts.setdefault(str(token.document_id), []).append(
                (str(token.gt_token_id), str(token.text))
            )

    audit_weights = weights or ConfidenceWeights()
    rows: list[dict[str, object]] = []
    for record in alignments.itertuples(index=False):
        ocr_ids = _ids(record.ocr_span_ids)
        gt_ids = _ids(record.gt_token_ids)
        ocr_text = str(record.ocr_text or "")
        gt_text = str(record.gt_text or "")
        status = str(record.status)
        recorded_margin = float(getattr(record, "uniqueness_margin", 1.0))
        recorded_confidence = float(getattr(record, "align_confidence", float("nan")))
        full_margin = recorded_margin
        full_confidence = recorded_confidence
        full_checked = bool(token_texts) and bool(ocr_text and gt_text)
        if full_checked:
            claimed = set(gt_ids)
            alternatives = [
                text
                for token_id, text in token_texts.get(str(record.document_id), ())
                if token_id not in claimed
            ]
            full_margin = uniqueness_margin(ocr_text, gt_text, alternatives)
            char_agreement = float(getattr(record, "char_agreement", 0.0))
            raw_geom = getattr(record, "geom_score", None)
            geom_score = None if raw_geom is None or pd.isna(raw_geom) else float(raw_geom)
            full_confidence, _diagnostics = component_confidence(
                char_agreement, geom_score, full_margin, audit_weights
            )

        recorded_state = _component_state(status)
        newly_ambiguous = bool(
            recorded_state == "eligible" and full_checked and full_confidence < min_align_confidence
        )
        state = "ambiguous" if newly_ambiguous else recorded_state
        audit_reason = (
            "full_uniqueness_below_confidence_floor"
            if newly_ambiguous
            else str(getattr(record, "reason_code", "") or "")
        )
        rows.append(
            {
                "alignment_id": str(record.alignment_id),
                "document_id": str(record.document_id),
                "dataset_id": str(getattr(record, "dataset_id", "")),
                "engine_id": str(record.engine_id),
                "relation": str(getattr(record, "relation", "")),
                "operation": _component_operation(len(ocr_ids), len(gt_ids)),
                "state": state,
                "recorded_state": recorded_state,
                "alignment_status": status,
                "reason_code": audit_reason,
                "n_ocr_spans": len(ocr_ids),
                "n_gt_tokens": len(gt_ids),
                "ocr_span_ids": json.dumps(list(ocr_ids), separators=(",", ":")),
                "gt_token_ids": json.dumps(list(gt_ids), separators=(",", ":")),
                "ocr_empty_anchor": len(ocr_ids) == 0,
                "gt_empty_anchor": len(gt_ids) == 0,
                "punctuation_boundary": _punctuation_boundary(ocr_text, gt_text),
                "whitespace_boundary": any(char.isspace() for char in ocr_text + gt_text),
                "align_confidence": recorded_confidence,
                "recorded_uniqueness_margin": recorded_margin,
                "full_uniqueness_margin": full_margin,
                "full_align_confidence": full_confidence,
                "full_uniqueness_checked": full_checked,
                "newly_ambiguous_full_uniqueness": newly_ambiguous,
            }
        )
    return pd.DataFrame(rows)


def component_denominators(audit: pd.DataFrame) -> pd.DataFrame:
    """Counts before and after ambiguity/exclusion, by engine and structural shape."""
    columns = [
        "engine_id",
        "operation",
        "n_components_before_exclusion",
        "n_eligible_after_exclusion",
        "n_ambiguous",
        "n_excluded",
        "n_unresolved",
        "eligible_share",
    ]
    if audit.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for (engine_id, operation), group in audit.groupby(["engine_id", "operation"], sort=True):
        states = group["state"].value_counts()
        total = len(group)
        eligible = int(states.get("eligible", 0))
        rows.append(
            {
                "engine_id": engine_id,
                "operation": operation,
                "n_components_before_exclusion": total,
                "n_eligible_after_exclusion": eligible,
                "n_ambiguous": int(states.get("ambiguous", 0)),
                "n_excluded": int(states.get("excluded", 0)),
                "n_unresolved": int(states.get("unresolved", 0)),
                "eligible_share": eligible / total if total else float("nan"),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _prepare(sites: pd.DataFrame) -> pd.DataFrame:
    frame = sites.copy()
    for column in ("ocr_text", "gt_text"):
        if column not in frame:
            msg = f"sites table is missing {column!r}"
            raise ValueError(msg)
        frame[column] = frame[column].fillna("")
    return frame


def site_eligibility_audit(
    sites: pd.DataFrame,
    components: pd.DataFrame,
    streams: dict[tuple[str, str], str] | None = None,
) -> pd.DataFrame:
    """Audit every site's View-A state and contiguous source projection.

    A GT-only alignment component has no OCR span.  Its historical ``[0, 0]`` range is a
    serialization default, not an insertion location, so it is unresolved until a real
    source anchor exists.  Likewise, a multi-span site's min/max envelope is not a valid
    editable source region when the exact stream slice differs from ``ocr_text``.  These
    states are decided without candidate outcomes and therefore form a reusable exclusion
    mask for all rungs.
    """
    required = {
        "site_id",
        "document_id",
        "engine_id",
        "site_kind",
        "evaluable",
        "alignment_ids",
        "ocr_span_ids",
        "ocr_text",
        "char_start",
        "char_end",
    }
    missing = sorted(required - set(sites.columns))
    if missing:
        raise ValueError(f"sites table is missing required columns: {', '.join(missing)}")
    component_required = {"alignment_id", "state", "reason_code"}
    component_missing = sorted(component_required - set(components.columns))
    if component_missing:
        raise ValueError(
            "component audit is missing required columns: " + ", ".join(component_missing)
        )

    component_lookup = {str(row.alignment_id): row for row in components.itertuples(index=False)}
    rows: list[dict[str, object]] = []
    for site in _prepare(sites).itertuples(index=False):
        site_id = str(site.site_id)
        document_id = str(site.document_id)
        engine_id = str(site.engine_id)
        alignment_ids = _ids(site.alignment_ids)
        span_ids = _ids(site.ocr_span_ids)
        linked = [component_lookup.get(item) for item in alignment_ids]
        components_found = bool(alignment_ids) and all(item is not None for item in linked)
        linked_states = [str(item.state) for item in linked if item is not None]
        full_uniqueness_ambiguous = any(
            bool(getattr(item, "newly_ambiguous_full_uniqueness", False))
            for item in linked
            if item is not None
        )
        stream_key = (document_id, engine_id)
        stream_present = streams is not None and stream_key in streams
        stream = (streams or {}).get(stream_key, "")
        start, end = int(str(site.char_start)), int(str(site.char_end))
        range_valid = 0 <= start <= end <= len(stream) if stream_present else False
        source_slice = stream[start:end] if range_valid else ""
        source_slice_projection = (
            bool(span_ids) and range_valid and source_slice == str(site.ocr_text)
        )
        empty_anchor = not span_ids

        if not bool(site.evaluable):
            state, reason = "ambiguous", "recorded_non_evaluable_site"
        elif not components_found:
            state, reason = "unresolved", "unknown_alignment_component"
        elif "unresolved" in linked_states:
            state, reason = "unresolved", "unresolved_alignment_component"
        elif "ambiguous" in linked_states:
            state, reason = (
                "ambiguous",
                "full_uniqueness_below_confidence_floor"
                if full_uniqueness_ambiguous
                else "ambiguous_alignment_component",
            )
        elif "excluded" in linked_states:
            state, reason = "excluded", "excluded_alignment_component"
        elif empty_anchor:
            state, reason = "unresolved", "empty_insertion_anchor"
        elif not stream_present:
            state, reason = "unresolved", "missing_stream"
        elif not range_valid:
            state, reason = "excluded", "source_range_out_of_bounds"
        elif not source_slice_projection:
            state, reason = "ambiguous", "noncontiguous_source_envelope"
        else:
            state, reason = "eligible", ""

        rows.append(
            {
                "site_id": site_id,
                "document_id": document_id,
                "dataset_id": str(getattr(site, "dataset_id", "")),
                "engine_id": engine_id,
                "site_kind": str(site.site_kind),
                "state": state,
                "reason": reason,
                "recorded_evaluable": bool(site.evaluable),
                "components_found": components_found,
                "n_alignment_components": len(alignment_ids),
                "full_uniqueness_ambiguous": full_uniqueness_ambiguous,
                "empty_anchor": empty_anchor,
                "stream_present": stream_present,
                "range_valid": range_valid,
                "source_slice_projection": source_slice_projection,
                "char_start": start,
                "char_end": end,
            }
        )
    return pd.DataFrame(rows)


def site_eligibility_denominators(audit: pd.DataFrame) -> pd.DataFrame:
    """Before/after site counts by engine and kind for the corrected audit mask."""
    columns = [
        "engine_id",
        "site_kind",
        "n_sites_before_exclusion",
        "n_eligible_after_exclusion",
        "n_ambiguous",
        "n_excluded",
        "n_unresolved",
        "eligible_share",
    ]
    if audit.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for (engine_id, site_kind), group in audit.groupby(["engine_id", "site_kind"], sort=True):
        states = group["state"].value_counts()
        total = len(group)
        eligible = int(states.get("eligible", 0))
        rows.append(
            {
                "engine_id": engine_id,
                "site_kind": site_kind,
                "n_sites_before_exclusion": total,
                "n_eligible_after_exclusion": eligible,
                "n_ambiguous": int(states.get("ambiguous", 0)),
                "n_excluded": int(states.get("excluded", 0)),
                "n_unresolved": int(states.get("unresolved", 0)),
                "eligible_share": eligible / total if total else float("nan"),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def twin_table(alignments: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    """Per engine x site kind: evaluable sites, twin-signature sites, share.

    The twin signature is the in-region form of what ADR-17 fixed on the orphan side: a
    site whose ground truth is empty (the OCR invented text) but whose exact OCR text
    appears among the ground-truth tokens the aligner never anchored on the same page —
    the engine read the text and the aligner lost it, so the "deletion needed" label is
    an alignment artifact.
    """
    if "reason_code" not in alignments:
        msg = "alignments table is missing 'reason_code'"
        raise ValueError(msg)
    frame = _prepare(sites)
    unanchored: dict[tuple[str, str], set[str]] = {}
    orphans = alignments[alignments["reason_code"] == "never_anchored"]
    for document_id, engine_id, gt_text in zip(
        orphans["document_id"], orphans["engine_id"], orphans["gt_text"].fillna(""), strict=True
    ):
        if gt_text.strip():
            unanchored.setdefault((str(document_id), str(engine_id)), set()).add(gt_text.strip())

    rows: list[dict[str, object]] = []
    for (engine_id, kind), group in frame[frame["evaluable"]].groupby(
        ["engine_id", "site_kind"], sort=True
    ):
        twins = sum(
            1
            for document_id, ocr_text, gt_text in zip(
                group["document_id"], group["ocr_text"], group["gt_text"], strict=True
            )
            if not gt_text.strip()
            and ocr_text.strip()
            and ocr_text.strip() in unanchored.get((str(document_id), str(engine_id)), ())
        )
        rows.append(
            {
                "engine_id": engine_id,
                "site_kind": kind,
                "n_evaluable_sites": len(group),
                "n_twin_signature": twins,
                "twin_share": twins / len(group) if len(group) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def region_pair_audit(
    sites: pd.DataFrame,
    streams: dict[tuple[str, str], str] | None = None,
    site_eligibility: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Audit the study's adjacent-evaluable pair universe and ambiguous boundaries.

    The generator pairs adjacent *evaluable* sites after filtering, not adjacent rows in
    the all-site table.  The primary population therefore mirrors that exact ordering and
    region id.  Raw boundaries touching a non-evaluable site are retained separately as
    diagnostics; they never inflate or shrink the study denominator.
    """
    from ocr_risk.experiments.cgv2_study import SiteCensusRecord, classify_site_pair

    frame = _prepare(sites)
    eligibility = (
        {
            str(row.site_id): (str(row.state), str(row.reason))
            for row in site_eligibility.itertuples(index=False)
        }
        if site_eligibility is not None and not site_eligibility.empty
        else {}
    )
    rows: list[dict[str, object]] = []
    for engine_id, group in frame.groupby("engine_id", sort=True):
        for _document_id, doc_sites in group.groupby("document_id", sort=True):
            ordered = doc_sites.sort_values(["char_start", "site_id"])
            raw_all = list(ordered.itertuples(index=False))

            for position, (left_raw, right_raw) in enumerate(pairwise(raw_all)):
                if bool(left_raw.evaluable) and bool(right_raw.evaluable):
                    continue
                rows.append(
                    {
                        "population": "diagnostic_non_evaluable_boundary",
                        "region_id": (
                            f"{_document_id}:{engine_id}:diagnostic-region:{position:05d}"
                        ),
                        "document_id": str(_document_id),
                        "dataset_id": str(getattr(left_raw, "dataset_id", "")),
                        "engine_id": str(engine_id),
                        "left_site_id": str(left_raw.site_id),
                        "right_site_id": str(right_raw.site_id),
                        "bucket": "ambiguous_member",
                        "source_bucket": "not_classified",
                        "state": "ambiguous",
                        "state_reason": "recorded_non_evaluable_member",
                        "left_evaluable": bool(left_raw.evaluable),
                        "right_evaluable": bool(right_raw.evaluable),
                        "empty_anchor": not str(left_raw.ocr_text) or not str(right_raw.ocr_text),
                        "punctuation_boundary": _punctuation_boundary(
                            str(left_raw.ocr_text), str(right_raw.ocr_text)
                        ),
                        "whitespace_boundary": False,
                    }
                )

            raw = [row for row in raw_all if bool(row.evaluable)]
            records = [
                SiteCensusRecord(
                    site_id=str(row.site_id),
                    document_id=str(row.document_id),
                    dataset_id=str(getattr(row, "dataset_id", "")),
                    engine_id=str(row.engine_id),
                    site_kind=str(row.site_kind),
                    d_before=int(str(row.d_before)),
                    gt_text=str(row.gt_text),
                    char_start=int(str(row.char_start)),
                    char_end=int(str(row.char_end)),
                    n_spans=1,
                    ocr_text=str(row.ocr_text),
                    min_align_confidence=1.0,
                )
                for row in raw
            ]
            stream = (streams or {}).get((str(_document_id), str(engine_id)), "")
            for position, (left, right) in enumerate(pairwise(records)):
                source_bucket, _ocr, _gap = classify_site_pair(left, right, stream)
                left_state, left_reason = eligibility.get(left.site_id, ("eligible", ""))
                right_state, right_reason = eligibility.get(right.site_id, ("eligible", ""))
                member_states = {left_state, right_state}
                if "unresolved" in member_states:
                    state, reason = "unresolved", left_reason or right_reason or "unresolved_member"
                elif "ambiguous" in member_states:
                    state, reason = "ambiguous", left_reason or right_reason or "ambiguous_member"
                elif "excluded" in member_states:
                    state, reason = "excluded", left_reason or right_reason or "excluded_member"
                elif source_bucket == "eligible":
                    state, reason = "eligible", ""
                elif source_bucket == "no_stream":
                    state, reason = "unresolved", "missing_stream"
                else:
                    state, reason = "excluded", source_bucket
                bucket = (
                    "ambiguous_member"
                    if state == "ambiguous"
                    else "unresolved_member"
                    if state == "unresolved" and source_bucket != "no_stream"
                    else "excluded_member"
                    if state == "excluded" and source_bucket == "eligible"
                    else source_bucket
                )
                rows.append(
                    {
                        "population": "study_evaluable_adjacency",
                        "region_id": f"{_document_id}:{engine_id}:region:{position:05d}",
                        "document_id": str(_document_id),
                        "dataset_id": str(getattr(raw[position], "dataset_id", "")),
                        "engine_id": str(engine_id),
                        "left_site_id": left.site_id,
                        "right_site_id": right.site_id,
                        "bucket": bucket,
                        "source_bucket": source_bucket,
                        "state": state,
                        "state_reason": reason,
                        "left_evaluable": bool(raw[position].evaluable),
                        "right_evaluable": bool(raw[position + 1].evaluable),
                        "empty_anchor": not left.ocr_text or not right.ocr_text,
                        "punctuation_boundary": _punctuation_boundary(
                            left.ocr_text, right.ocr_text
                        ),
                        "whitespace_boundary": bool(_ocr)
                        and _ocr.startswith(left.ocr_text)
                        and _ocr.endswith(right.ocr_text)
                        and _ocr[len(left.ocr_text) : len(_ocr) - len(right.ocr_text)].isspace(),
                    }
                )
    return pd.DataFrame(rows)


def region_census(
    sites: pd.DataFrame,
    streams: dict[tuple[str, str], str] | None = None,
    *,
    pair_audit: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per-engine pair denominators before/after ambiguity and exclusion."""
    audit = region_pair_audit(sites, streams) if pair_audit is None else pair_audit
    population = (
        audit[audit["population"] == "study_evaluable_adjacency"]
        if "population" in audit.columns
        else audit
    )
    diagnostic = (
        audit[audit["population"] == "diagnostic_non_evaluable_boundary"]
        if "population" in audit.columns
        else audit.iloc[0:0]
    )
    rows: list[dict[str, object]] = []
    for engine_id, group in population.groupby("engine_id", sort=True):
        counts = {
            bucket: int((group["bucket"] == bucket).sum())
            for bucket in (
                "eligible",
                "cross_line",
                "intervening_text",
                "degenerate",
                "ambiguous_member",
                "text_mismatch",
                "non_adjacent",
                "no_stream",
                "unresolved_member",
                "excluded_member",
            )
        }
        # The overlay bucket can preempt the classifier's source bucket (every degenerate
        # source pair has an empty-OCR member, which the member-state overlay buckets
        # first), so the source classification is reported alongside rather than lost.
        source_counts = {
            f"n_source_{bucket}": int((group["source_bucket"] == bucket).sum())
            for bucket in (
                "eligible",
                "cross_line",
                "intervening_text",
                "degenerate",
                "text_mismatch",
                "non_adjacent",
                "no_stream",
            )
            if "source_bucket" in group.columns
        }
        states = group["state"].value_counts()
        n_before = len(group)
        n_ambiguous = int(states.get("ambiguous", 0))
        n_unresolved = int(states.get("unresolved", 0))
        n_eligible = int(states.get("eligible", 0))
        n_excluded = int(states.get("excluded", 0))
        n_diagnostic = len(diagnostic[diagnostic["engine_id"] == engine_id])
        rows.append(
            {
                "engine_id": engine_id,
                "n_pairs_before_exclusion": n_before,
                "n_pairs_eligible_after_exclusion": n_eligible,
                "n_pairs_ambiguous": n_ambiguous,
                "n_pairs_excluded": n_excluded,
                "n_pairs_unresolved": n_unresolved,
                "n_diagnostic_ambiguous_boundaries": n_diagnostic,
                **counts,
                **source_counts,
            }
        )
    return pd.DataFrame(rows)


def _integer_field(record: object, name: str) -> int | None:
    value = getattr(record, name, None)
    if value is None or str(value) == "":
        return None
    try:
        return int(str(value))
    except ValueError:
        return None


def _candidate_label_valid(
    record: object, original: str, ground_truth: str, candidate: str
) -> bool:
    """Recompute every saved edit-distance label from strings, independently."""
    expected_before = distance(original, ground_truth)
    expected_after = distance(candidate, ground_truth)
    expected_outcome = classify_accepted(original, candidate, ground_truth).value
    recorded_before = _integer_field(record, "d_before")
    recorded_after = _integer_field(record, "d_after")
    recorded_delta = _integer_field(record, "delta")
    recorded_outcome = str(getattr(record, "outcome", "")).strip().lower()
    return bool(
        recorded_before == expected_before
        and recorded_after == expected_after
        and recorded_delta == expected_before - expected_after
        and recorded_outcome == expected_outcome
    )


def projection_audit(
    region_proposals: pd.DataFrame,
    sites: pd.DataFrame,
    streams: dict[tuple[str, str], str] | None = None,
    site_eligibility: pd.DataFrame | None = None,
    pair_audit: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """View B, one row per region candidate, reprojected from canonical sites.

    The candidate's own range, gap, member ids, and source-span provenance are checks,
    never inputs. Each row ends in an explicit eligibility state; a malformed or
    ambiguous projection can therefore never silently remain in the clean denominator.
    """
    columns = [
        "candidate_id",
        "region_id",
        "document_id",
        "engine_id",
        "operation_type",
        "state",
        "reason",
        "projection_correct",
        "label_valid",
        "pair_projection",
        "member_eligibility",
        "member_sites_found",
        "same_document_engine",
        "left_span_projection",
        "right_span_projection",
        "left_alignment_projection",
        "right_alignment_projection",
        "range_projection",
        "ocr_projection",
        "gt_projection",
        "gap_projection",
        "left_evaluable",
        "right_evaluable",
        "punctuation_boundary",
        "whitespace_boundary",
        "empty_anchor",
        "overlaps_other_region",
        "same_region_candidate_count",
    ]
    if region_proposals.empty:
        return pd.DataFrame(columns=columns)
    frame = _prepare(sites)
    site_lookup = {str(row.site_id): row for row in frame.itertuples(index=False)}
    site_states = (
        {
            str(row.site_id): (str(row.state), str(row.reason))
            for row in site_eligibility.itertuples(index=False)
        }
        if site_eligibility is not None and not site_eligibility.empty
        else {}
    )
    eligible_pairs = (
        {
            (str(row.left_site_id), str(row.right_site_id)): str(row.state)
            for row in pair_audit.itertuples(index=False)
            if not hasattr(row, "population") or str(row.population) == "study_evaluable_adjacency"
        }
        if pair_audit is not None and not pair_audit.empty
        else {}
    )
    region_intervals = {
        str(row.region_id): (
            str(getattr(row, "document_id", "")),
            str(row.engine_id),
            int(str(getattr(row, "region_char_start", 0) or 0)),
            int(str(getattr(row, "region_char_end", 0) or 0)),
        )
        for row in region_proposals.itertuples(index=False)
    }
    overlaps = _overlapping_interval_ids(region_intervals)
    region_counts = region_proposals.groupby("region_id").size().to_dict()
    rows: list[dict[str, object]] = []
    for record in region_proposals.itertuples(index=False):
        region_id = str(record.region_id)
        engine_id = str(record.engine_id)
        left = site_lookup.get(str(record.left_site_id))
        right = site_lookup.get(str(record.right_site_id))
        if left is None or right is None:
            rows.append(
                {
                    **dict.fromkeys(columns, False),
                    "candidate_id": str(getattr(record, "candidate_id", region_id)),
                    "region_id": region_id,
                    "document_id": str(getattr(record, "document_id", "")),
                    "engine_id": engine_id,
                    "operation_type": str(getattr(record, "operation_type", "")),
                    "state": "unresolved",
                    "reason": "unknown_member_site",
                    "same_region_candidate_count": int(region_counts.get(region_id, 0)),
                }
            )
            continue

        document_id = str(left.document_id)
        stream_key = (document_id, engine_id)
        stream_present = streams is not None and stream_key in streams
        stream = (streams or {}).get(stream_key, "")
        start, end = int(str(left.char_start)), int(str(right.char_end))
        expected_ocr = stream[start:end] if stream_present else ""
        left_text, right_text = str(left.ocr_text), str(right.ocr_text)
        middle = (
            expected_ocr[len(left_text) : len(expected_ocr) - len(right_text)]
            if expected_ocr.startswith(left_text) and expected_ocr.endswith(right_text)
            else None
        )
        expected_gt = " ".join(t for t in (str(left.gt_text), str(right.gt_text)) if t)
        same_document_engine = (
            str(left.document_id) == str(right.document_id)
            and str(left.engine_id) == str(right.engine_id) == engine_id
            and str(getattr(record, "document_id", document_id)) == document_id
        )
        left_span_projection = _ids(getattr(record, "left_source_span_ids", ())) == _ids(
            left.ocr_span_ids
        )
        right_span_projection = _ids(getattr(record, "right_source_span_ids", ())) == _ids(
            right.ocr_span_ids
        )
        left_alignment_projection = _ids(
            getattr(record, "left_alignment_component_ids", ())
        ) == _ids(left.alignment_ids)
        right_alignment_projection = _ids(
            getattr(record, "right_alignment_component_ids", ())
        ) == _ids(right.alignment_ids)
        range_projection = (
            int(str(getattr(record, "region_char_start", -1))) == start
            and int(str(getattr(record, "region_char_end", -1))) == end
        )
        ocr_projection = stream_present and expected_ocr == str(record.region_ocr)
        gt_projection = expected_gt == str(record.region_gt)
        gap_projection = middle is not None and len(middle) == int(str(record.gap_chars))
        kinds_match = str(left.site_kind) == str(record.left_kind) and str(right.site_kind) == str(
            record.right_kind
        )
        left_evaluable = bool(left.evaluable)
        right_evaluable = bool(right.evaluable)
        label_valid = gt_projection and _candidate_label_valid(
            record,
            expected_ocr,
            expected_gt,
            str(record.candidate_text),
        )
        left_state, left_reason = site_states.get(str(left.site_id), ("eligible", ""))
        right_state, right_reason = site_states.get(str(right.site_id), ("eligible", ""))
        member_eligibility = left_state == "eligible" and right_state == "eligible"
        pair_state = eligible_pairs.get((str(left.site_id), str(right.site_id)))
        pair_projection = pair_state == "eligible" if eligible_pairs else True
        correct = all(
            (
                same_document_engine,
                left_span_projection,
                right_span_projection,
                left_alignment_projection,
                right_alignment_projection,
                range_projection,
                ocr_projection,
                gt_projection,
                gap_projection,
                kinds_match,
                left_evaluable,
                right_evaluable,
                label_valid,
                member_eligibility,
                pair_projection,
            )
        )
        if not stream_present:
            state, reason = "unresolved", "missing_stream"
        elif "unresolved" in (left_state, right_state):
            state, reason = "unresolved", left_reason or right_reason or "unresolved_member"
        elif "ambiguous" in (left_state, right_state):
            state, reason = "ambiguous", left_reason or right_reason or "ambiguous_member"
        elif not left_evaluable or not right_evaluable:
            state, reason = "ambiguous", "ambiguous_member"
        elif not pair_projection:
            state, reason = "excluded", "pair_not_eligible"
        elif not label_valid:
            state, reason = "excluded", "invalid_candidate_label"
        elif correct:
            state, reason = "eligible", ""
        else:
            state, reason = "excluded", "projection_mismatch"
        rows.append(
            {
                "candidate_id": str(getattr(record, "candidate_id", region_id)),
                "region_id": region_id,
                "document_id": document_id,
                "engine_id": engine_id,
                "operation_type": str(getattr(record, "operation_type", "")),
                "state": state,
                "reason": reason,
                "projection_correct": correct,
                "label_valid": label_valid,
                "pair_projection": pair_projection,
                "member_eligibility": member_eligibility,
                "member_sites_found": True,
                "same_document_engine": same_document_engine,
                "left_span_projection": left_span_projection,
                "right_span_projection": right_span_projection,
                "left_alignment_projection": left_alignment_projection,
                "right_alignment_projection": right_alignment_projection,
                "range_projection": range_projection,
                "ocr_projection": ocr_projection,
                "gt_projection": gt_projection,
                "gap_projection": gap_projection,
                "left_evaluable": left_evaluable,
                "right_evaluable": right_evaluable,
                "punctuation_boundary": _punctuation_boundary(left_text, right_text),
                "whitespace_boundary": middle is not None and bool(middle) and middle.isspace(),
                "empty_anchor": not left_text or not right_text,
                "overlaps_other_region": bool(overlaps.get(region_id, False)),
                "same_region_candidate_count": int(region_counts.get(region_id, 0)),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def projection_summary(audit: pd.DataFrame) -> pd.DataFrame:
    """Compact per-engine denominators for the row-level region projection audit."""
    columns = [
        "engine_id",
        "n_region_candidates",
        "n_regions",
        "projection_correct",
        "projection_mismatch",
        "label_valid",
        "eligible",
        "ambiguous",
        "excluded",
        "unresolved",
    ]
    if audit.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for engine_id, group in audit.groupby("engine_id", sort=True):
        states = group["state"].value_counts()
        correct = int(group["projection_correct"].astype(bool).sum())
        rows.append(
            {
                "engine_id": engine_id,
                "n_region_candidates": len(group),
                "n_regions": group["region_id"].nunique(),
                "projection_correct": correct,
                "projection_mismatch": len(group) - correct,
                "label_valid": int(group["label_valid"].astype(bool).sum()),
                "eligible": int(states.get("eligible", 0)),
                "ambiguous": int(states.get("ambiguous", 0)),
                "excluded": int(states.get("excluded", 0)),
                "unresolved": int(states.get("unresolved", 0)),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def site_projection_audit(
    proposals: pd.DataFrame,
    sites: pd.DataFrame,
    streams: dict[tuple[str, str], str] | None = None,
    site_eligibility: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """View B for site candidates, including source-slice and label recomputation."""
    columns = [
        "candidate_id",
        "site_id",
        "document_id",
        "engine_id",
        "operation_type",
        "state",
        "reason",
        "projection_correct",
        "label_valid",
        "site_eligibility",
        "document_engine_projection",
        "range_projection",
        "source_slice_projection",
        "span_projection",
        "alignment_projection",
        "geometry_projection",
        "empty_anchor",
        "punctuation_boundary",
        "whitespace_boundary",
        "overlaps_other_site",
        "same_site_candidate_count",
    ]
    if proposals.empty:
        return pd.DataFrame(columns=columns)
    frame = _prepare(sites)
    site_lookup = {str(row.site_id): row for row in frame.itertuples(index=False)}
    site_states = (
        {
            str(row.site_id): (str(row.state), str(row.reason))
            for row in site_eligibility.itertuples(index=False)
        }
        if site_eligibility is not None and not site_eligibility.empty
        else {}
    )
    proposal_site_ids = set(proposals["site_id"].astype(str))
    site_intervals = {
        str(row.site_id): (
            str(row.document_id),
            str(row.engine_id),
            int(str(row.char_start)),
            int(str(row.char_end)),
        )
        for row in frame.itertuples(index=False)
        if str(row.site_id) in proposal_site_ids
    }
    overlaps = _overlapping_interval_ids(site_intervals)
    site_counts = proposals.groupby("site_id").size().to_dict()
    rows: list[dict[str, object]] = []
    for record in proposals.itertuples(index=False):
        site_id = str(record.site_id)
        site = site_lookup.get(site_id)
        if site is None:
            rows.append(
                {
                    **dict.fromkeys(columns, False),
                    "candidate_id": str(getattr(record, "candidate_id", site_id)),
                    "site_id": site_id,
                    "document_id": str(getattr(record, "document_id", "")),
                    "engine_id": str(getattr(record, "engine_id", "")),
                    "operation_type": str(getattr(record, "operation_type", "")),
                    "state": "unresolved",
                    "reason": "unknown_site",
                    "same_site_candidate_count": int(site_counts.get(site_id, 0)),
                }
            )
            continue
        document_engine = str(record.document_id) == str(site.document_id) and str(
            record.engine_id
        ) == str(site.engine_id)
        range_projection = int(str(record.source_char_start)) == int(str(site.char_start)) and int(
            str(record.source_char_end)
        ) == int(str(site.char_end))
        span_projection = _ids(record.source_span_ids) == _ids(site.ocr_span_ids)
        alignment_projection = _ids(record.alignment_component_ids) == _ids(site.alignment_ids)
        geometry_projection = str(record.source_geometry_id) == f"site:{site_id}"
        stream_key = (str(site.document_id), str(site.engine_id))
        stream_present = streams is not None and stream_key in streams
        stream = (streams or {}).get(stream_key, "")
        start, end = int(str(site.char_start)), int(str(site.char_end))
        span_ids = _ids(site.ocr_span_ids)
        source_slice_projection = bool(
            span_ids
            and stream_present
            and 0 <= start <= end <= len(stream)
            and stream[start:end] == str(site.ocr_text)
        )
        label_valid = (
            str(getattr(record, "original_ocr", "")) == str(site.ocr_text)
            and str(getattr(record, "gt_text", "")) == str(site.gt_text)
            and _candidate_label_valid(
                record,
                str(site.ocr_text),
                str(site.gt_text),
                str(getattr(record, "candidate_text", "")),
            )
        )
        audited_state, audited_reason = site_states.get(site_id, ("eligible", ""))
        site_is_eligible = audited_state == "eligible"
        correct = all(
            (
                document_engine,
                range_projection,
                span_projection,
                alignment_projection,
                geometry_projection,
                bool(site.evaluable),
                source_slice_projection,
                label_valid,
                site_is_eligible,
            )
        )
        if not bool(site.evaluable):
            state, reason = "ambiguous", "non_evaluable_site"
        elif audited_state == "unresolved" or not span_ids:
            state, reason = "unresolved", audited_reason or "empty_insertion_anchor"
        elif audited_state == "ambiguous":
            state, reason = "ambiguous", audited_reason or "ambiguous_site"
        elif audited_state == "excluded":
            state, reason = "excluded", audited_reason or "excluded_site"
        elif not label_valid:
            state, reason = "excluded", "invalid_candidate_label"
        elif correct:
            state, reason = "eligible", ""
        else:
            state, reason = "excluded", "projection_mismatch"
        original = str(getattr(record, "original_ocr", ""))
        rows.append(
            {
                "candidate_id": str(record.candidate_id),
                "site_id": site_id,
                "document_id": str(site.document_id),
                "engine_id": str(site.engine_id),
                "operation_type": str(record.operation_type),
                "state": state,
                "reason": reason,
                "projection_correct": correct,
                "label_valid": label_valid,
                "site_eligibility": site_is_eligible,
                "document_engine_projection": document_engine,
                "range_projection": range_projection,
                "source_slice_projection": source_slice_projection,
                "span_projection": span_projection,
                "alignment_projection": alignment_projection,
                "geometry_projection": geometry_projection,
                "empty_anchor": not span_ids,
                "punctuation_boundary": _punctuation_boundary(original),
                "whitespace_boundary": any(char.isspace() for char in original),
                "overlaps_other_site": bool(overlaps.get(site_id, False)),
                "same_site_candidate_count": int(site_counts.get(site_id, 0)),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def twin_sites_table(alignments: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    """Per evaluable site: the R-37 twin signature, so headline tables can exclude it.

    The aggregate twin table answers "how much"; this one answers "which sites", which is
    what a sensitivity variant of a per-engine contrast needs -- and what the protocol's
    pre-registered R-37 sensitivity requires to exist before the evaluate run fires.
    """
    if "reason_code" not in alignments:
        msg = "alignments table is missing 'reason_code'"
        raise ValueError(msg)
    frame = _prepare(sites)
    unanchored: dict[tuple[str, str], set[str]] = {}
    orphans = alignments[alignments["reason_code"] == "never_anchored"]
    for document_id, engine_id, gt_text in zip(
        orphans["document_id"], orphans["engine_id"], orphans["gt_text"].fillna(""), strict=True
    ):
        if str(gt_text).strip():
            unanchored.setdefault((str(document_id), str(engine_id)), set()).add(
                str(gt_text).strip()
            )
    rows: list[dict[str, object]] = []
    for row in frame.itertuples(index=False):
        engine_id = str(row.engine_id)
        ocr_text, gt_text = str(row.ocr_text), str(row.gt_text)
        rows.append(
            {
                "site_id": str(row.site_id),
                "document_id": str(row.document_id),
                "engine_id": engine_id,
                "site_kind": str(row.site_kind),
                "twin_signature": bool(
                    not gt_text.strip()
                    and ocr_text.strip()
                    and ocr_text.strip() in unanchored.get((str(row.document_id), engine_id), ())
                ),
            }
        )
    return pd.DataFrame(rows)
