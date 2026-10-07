"""The CGV2 decision gates (protocol §16), read from canonical artifacts only.

The generator gate's criteria were written into the protocol before any evaluate-role
result existed. This module applies them to the CSVs the study and tables commands
produced — it contains no thresholds of its own, and any criterion whose evidence file
is absent makes the gate INCONCLUSIVE rather than silently unmet.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ocr_risk.analysis import cgv2_tables
from ocr_risk.experiments.cgv2_study import HEADLINE_EXCLUDED_DATASETS
from ocr_risk.io.hashing import file_sha256

__all__ = ["evaluate_cgv2_gates"]

_H2_ACCEPTED_EDITS_FLOOR = 245
_H2_SAFE_COVERAGE_FLOOR = 0.05
_H2_CANDIDATE_FLOOR = 1000
_H2_DOCUMENT_FLOOR = 20
_H2_CLEAN_SHARE_FLOOR = 0.30
_H2_CLEAN_PROPOSAL_FLOOR = 100
_CLEAN_RATE_BOUND = 0.30
_HARM_RATIO_MULTIPLIER = 2.0
_ENGINES_REQUIRED = 3


@dataclass(frozen=True, slots=True)
class _Criterion:
    key: str
    question: str
    threshold: str
    met: bool | None
    observed: str


def _engines_passing(contrast: pd.DataFrame, metric: str) -> tuple[int, int, str]:
    rows = contrast[contrast["metric"] == metric]
    if rows.empty:
        return 0, 0, "no contrast rows"
    rejected = rows["reject_holm"].eq(True)
    passing = rows[(rows["ci_lower"] > 0) & rejected]
    degenerate = "degenerate_interval" in rows.columns

    def _interval(row: Any) -> str:
        ci = f"CI[{row.ci_lower:+.4f},{row.ci_upper:+.4f}]"
        if degenerate and bool(getattr(row, "degenerate_interval", False)):
            return f"{ci} (degenerate: zero-width resample distribution)"
        return ci

    return (
        len(passing),
        len(rows),
        "; ".join(
            f"{row.engine_id}: d={row.delta:+.4f} {_interval(row)} p_holm={row.p_holm:.4f}"
            for row in rows.itertuples(index=False)
        ),
    )


def _criterion_cg1(contrast: pd.DataFrame) -> _Criterion:
    n_passing, n_total, observed = _engines_passing(contrast, "site_availability")
    return _Criterion(
        key="CG1_availability",
        question="Does g6_union raise error-site availability over g3_edit_aware?",
        threshold="Holm-adjusted CI excluding 0, positive, on >= 3 of 4 engines",
        met=(n_passing >= _ENGINES_REQUIRED) if n_total else None,
        observed=f"{n_passing}/{n_total} engines; {observed}",
    )


def _criterion_cg2(quality: pd.DataFrame, preservation: pd.DataFrame) -> _Criterion:
    observations: list[str] = []
    met = True
    for engine_id in sorted(quality["engine_id"].unique()):
        ratio_row = quality[
            (quality["generator_id"] == "g6_union") & (quality["engine_id"] == engine_id)
        ]
        baseline_row = quality[
            (quality["generator_id"] == "g3_edit_aware") & (quality["engine_id"] == engine_id)
        ]
        clean_row = preservation[
            (preservation["generator_id"] == "g6_union") & (preservation["engine_id"] == engine_id)
        ]
        if ratio_row.empty or baseline_row.empty or clean_row.empty:
            met = False
            observations.append(f"{engine_id}: missing rows")
            continue
        ratio = float(ratio_row["harmful_beneficial_ratio"].iloc[0])
        baseline_ratio = float(baseline_row["harmful_beneficial_ratio"].iloc[0])
        clean_rate = float(clean_row["clean_span_proposal_rate"].iloc[0])
        ratio_ok = ratio <= _HARM_RATIO_MULTIPLIER * max(baseline_ratio, 1e-9) or (
            ratio == float("inf") and baseline_ratio == float("inf")
        )
        clean_ok = clean_rate <= _CLEAN_RATE_BOUND
        met = met and ratio_ok and clean_ok
        observations.append(
            f"{engine_id}: ratio {ratio:.2f} vs 2x{baseline_ratio:.2f}, clean rate {clean_rate:.3f}"
        )
    return _Criterion(
        key="CG2_harm_bounded",
        question="Is the structural pool's harmful burden bounded?",
        threshold="ratio <= 2x g3's per engine on 4/4, clean-span proposal rate <= 0.30 on 4/4",
        met=met if observations else None,
        observed="; ".join(observations),
    )


def _criterion_cg3(contrast: pd.DataFrame, opportunity: pd.DataFrame) -> _Criterion:
    n_passing, n_total, _ = _engines_passing(contrast, "structural_stratum_availability")
    insertion = opportunity[
        (opportunity["generator_id"] == "g6_union")
        & (opportunity["stratum"] == "deletion")
        & (opportunity["k"] == 4)
    ]
    insertion_positive = int((insertion["availability"] > 0).sum()) if not insertion.empty else 0
    insertion_total = len(insertion)
    structural_ok = n_total > 0 and n_passing >= _ENGINES_REQUIRED
    insertion_ok = insertion_total > 0 and insertion_positive >= _ENGINES_REQUIRED
    return _Criterion(
        key="CG3_structural_coverage",
        question="Does the structural stratum gain coverage, from exactly zero on insertions?",
        threshold="structural CI positive on >= 3/4 AND insertion availability > 0 on >= 3/4",
        met=(structural_ok and insertion_ok) if n_total and insertion_total else None,
        observed=(
            f"structural {n_passing}/{n_total}; insertion-repair availability > 0 on "
            f"{insertion_positive}/{insertion_total}: "
            + "; ".join(
                f"{row.engine_id}={row.availability:.4f}"
                for row in insertion.itertuples(index=False)
            )
        ),
    )


def _within_recorded_native_cap(frame: pd.DataFrame) -> pd.DataFrame:
    """Filter canonical rows to their configured pool; tolerate compact gate fixtures."""
    if frame.empty or not {"generator_rank", "generator_native_cap"} <= set(frame.columns):
        return frame
    rank = pd.to_numeric(frame["generator_rank"], errors="coerce")
    cap = pd.to_numeric(frame["generator_native_cap"], errors="coerce")
    return frame[rank.notna() & cap.notna() & rank.lt(cap)].copy()


def audit_lineage_mismatches(
    audit_gate: dict[str, object], current: dict[str, tuple[str | None, str | None]]
) -> list[str]:
    """Compare the audit gate's recorded upstream runs/hashes against the live store.

    ``current`` maps the audit's stage names to the store's latest ``(run_id, sha256)``
    for the audited table. Any stage whose run or table hash moved is a mismatch: the
    audit certified component states against artifacts that are no longer the ones the
    store would produce, so the overlay it licenses is stale.
    """
    recorded_runs = audit_gate.get("input_runs", {})
    recorded_inputs = audit_gate.get("inputs", {})
    table_by_stage = {
        "align": ("alignments", "alignments_sha256"),
        "sites": ("sites", "sites_sha256"),
        "canonicalize": ("spans", "spans_sha256"),
        "manifest": ("gt_tokens", "gt_tokens_sha256"),
    }
    if not isinstance(recorded_runs, dict) or not isinstance(recorded_inputs, dict):
        return ["alignment audit gate lacks input_runs/inputs lineage"]
    mismatches: list[str] = []
    for stage, (table, hash_key) in table_by_stage.items():
        run_id, table_sha = current.get(stage, (None, None))
        if run_id is None:
            mismatches.append(f"{stage}: no current {table} run for this experiment")
            continue
        if not isinstance(recorded_runs.get(stage), str) or recorded_runs[stage] != run_id:
            mismatches.append(
                f"{stage}: audit recorded run {recorded_runs.get(stage)!r}, store has {run_id!r}"
            )
            continue
        if (
            not isinstance(recorded_inputs.get(hash_key), str)
            or recorded_inputs[hash_key] != table_sha
        ):
            mismatches.append(f"{table}: table hash moved since the audit ({hash_key})")
    return mismatches


def _h2_criteria(
    oracle: pd.DataFrame,
    quality: pd.DataFrame,
    preservation: pd.DataFrame,
    census: pd.DataFrame,
    proposals: pd.DataFrame,
    region_proposals: pd.DataFrame,
    audit_dir: Path,
    analysis_dir: Path,
    role_suffix: str,
) -> list[_Criterion]:
    augmented = oracle[
        (oracle["generator_id"] == "g6_union") & (oracle["variant"] == "site_plus_regions")
    ]
    natural_sites = (
        proposals[proposals["pool"].fillna("").astype(str).str.lower() == "natural"]
        if "pool" in proposals.columns
        else proposals.iloc[0:0]
    )
    natural_sites = _within_recorded_native_cap(natural_sites)
    if "original_ocr" in natural_sites.columns:
        natural_sites = natural_sites[
            natural_sites["candidate_text"].astype(str) != natural_sites["original_ocr"].astype(str)
        ]
    natural_regions = (
        region_proposals[region_proposals["pool"].fillna("").astype(str).str.lower() == "natural"]
        if not region_proposals.empty and "pool" in region_proposals.columns
        else region_proposals.iloc[0:0]
    )
    natural_regions = _within_recorded_native_cap(natural_regions)
    if "region_ocr" in natural_regions.columns:
        natural_regions = natural_regions[
            natural_regions["candidate_text"].astype(str)
            != natural_regions["region_ocr"].astype(str)
        ]
    region_counts = (
        natural_regions.groupby("engine_id")["candidate_text"].count()
        if not natural_regions.empty
        else pd.Series(dtype=int)
    )
    site_counts = (
        natural_sites[natural_sites["generator_id"] == "g6_union"]
        .groupby("engine_id")["candidate_text"]
        .count()
    )
    n_documents = census["document_id"].nunique()
    clean_share = {
        str(engine_id): float((group["d_before"] == 0).mean())
        for engine_id, group in census.groupby("engine_id")
    }
    clean_proposed = {
        str(row.engine_id): int(str(row.n_clean_sites_proposed))
        for row in preservation[preservation["generator_id"] == "g6_union"].itertuples(index=False)
    }

    a_pass = int((augmented["oracle_accepted_edits"] >= _H2_ACCEPTED_EDITS_FLOOR).sum())
    b_pass = int((augmented["oracle_safe_coverage"] >= _H2_SAFE_COVERAGE_FLOOR).sum())
    n_engines = len(augmented)
    c_ok = (
        all(
            int(site_counts.get(engine_id, 0)) + int(region_counts.get(engine_id, 0))
            >= _H2_CANDIDATE_FLOOR
            for engine_id in augmented["engine_id"]
        )
        and n_documents >= _H2_DOCUMENT_FLOOR
    )
    d_ok = all(
        clean_share.get(str(engine_id), 0.0) >= _H2_CLEAN_SHARE_FLOOR
        and clean_proposed.get(str(engine_id), 0) >= _H2_CLEAN_PROPOSAL_FLOOR
        for engine_id in augmented["engine_id"]
    )
    audit_gate_path = audit_dir / "alignment_audit_gate.json"
    audit_gate: dict[str, object] = {}
    if audit_gate_path.exists():
        try:
            payload = json.loads(audit_gate_path.read_text(encoding="utf-8"))
            audit_gate = payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            audit_gate = {}
    required_audit_paths = [
        audit_dir / name
        for name in (
            "alignment_component_audit.csv",
            "alignment_component_denominators.csv",
            "alignment_site_audit.csv",
            "alignment_site_denominators.csv",
            "alignment_twin_sites.csv",
            "region_pair_audit.csv",
            "site_projection_audit.csv",
            "region_projection_audit.csv",
        )
    ]
    sensitivity_paths = [
        analysis_dir / f"cgv2_contrasts{role_suffix}__no_twin.csv",
        analysis_dir / f"cgv2_opportunity{role_suffix}__no_twin.csv",
        analysis_dir / f"cgv2_quality{role_suffix}__no_twin.csv",
        analysis_dir / f"cgv2_oracle{role_suffix}__no_twin.csv",
    ]
    sensitivity_manifest_path = analysis_dir / f"cgv2_analysis_manifest{role_suffix}__no_twin.json"
    site_path = analysis_dir / f"cgv2_proposals{role_suffix}.csv"
    region_path = analysis_dir / f"cgv2_region_proposals{role_suffix}.csv"
    audit_inputs = audit_gate.get("inputs", {})
    hashes_match = (
        isinstance(audit_inputs, dict)
        and site_path.is_file()
        and audit_inputs.get("site_proposals_sha256") == file_sha256(site_path)
        and (
            not region_path.is_file()
            or audit_inputs.get("region_proposals_sha256") == file_sha256(region_path)
        )
    )
    audit_outputs = audit_gate.get("outputs", {})
    audit_output_hashes_match = bool(
        isinstance(audit_outputs, dict)
        and all(
            isinstance(audit_outputs.get(path.name), dict)
            and audit_outputs[path.name].get("sha256") == file_sha256(path)
            for path in required_audit_paths
            if path.is_file()
        )
    )
    missing_audit = [path.name for path in required_audit_paths if not path.is_file()]
    missing_sensitivity = [path.name for path in sensitivity_paths if not path.is_file()]
    sensitivity_manifest_valid = False
    if sensitivity_manifest_path.exists():
        try:
            sensitivity_manifest = json.loads(sensitivity_manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            sensitivity_manifest = {}
        manifest_inputs = sensitivity_manifest.get("inputs", {})
        manifest_outputs = sensitivity_manifest.get("outputs", {})
        sensitivity_manifest_valid = bool(
            isinstance(manifest_inputs, dict)
            and isinstance(manifest_outputs, dict)
            and all(
                manifest_inputs.get(path.name) == file_sha256(path)
                for path in (
                    audit_dir / "alignment_twin_sites.csv",
                    audit_dir / "alignment_site_audit.csv",
                    audit_dir / "region_pair_audit.csv",
                    audit_dir / "site_projection_audit.csv",
                    audit_dir / "region_projection_audit.csv",
                )
                if path.is_file()
            )
            and all(
                isinstance(manifest_outputs.get(path.name), dict)
                and manifest_outputs[path.name].get("sha256") == file_sha256(path)
                for path in sensitivity_paths
                if path.is_file()
            )
        )
    f_ok = (
        bool(audit_gate.get("overall_pass"))
        and hashes_match
        and audit_output_hashes_match
        and not missing_audit
        and not missing_sensitivity
        and sensitivity_manifest_valid
    )
    pool_frames = [frame for frame in (proposals, region_proposals) if not frame.empty]
    pool_measurable = bool(pool_frames) and all("pool" in frame.columns for frame in pool_frames)
    all_natural = (
        all(
            frame["pool"].fillna("").astype(str).str.lower().eq("natural").all()
            for frame in pool_frames
        )
        if pool_measurable
        else None
    )
    pool_rows = sum(len(frame) for frame in pool_frames)

    observed_a = "; ".join(
        f"{row.engine_id}={int(str(row.oracle_accepted_edits))}"
        for row in augmented.itertuples(index=False)
    )
    observed_b = "; ".join(
        f"{row.engine_id}={row.oracle_safe_coverage:.4f}"
        for row in augmented.itertuples(index=False)
    )
    return [
        _Criterion(
            key="A_repair_opportunity",
            question="Does the pool admit enough beneficial edits to certify any target?",
            threshold=f">= {_H2_ACCEPTED_EDITS_FLOOR} oracle-accepted sites on >= 3/4 engines",
            met=(a_pass >= _ENGINES_REQUIRED) if n_engines else None,
            observed=f"{a_pass}/{n_engines} engines; {observed_a}",
        ),
        _Criterion(
            key="B_oracle_safe_coverage",
            question="Would a perfect verifier trace a risk-coverage curve worth reading?",
            threshold=f"oracle safe coverage >= {_H2_SAFE_COVERAGE_FLOOR} on >= 3/4 engines",
            met=(b_pass >= _ENGINES_REQUIRED) if n_engines else None,
            observed=f"{b_pass}/{n_engines} engines; {observed_b}",
        ),
        _Criterion(
            key="C_sample_size",
            question="Are there enough candidates and enough document clusters?",
            threshold=f">= {_H2_CANDIDATE_FLOOR} natural candidates per engine, "
            f">= {_H2_DOCUMENT_FLOOR} test documents",
            met=c_ok if n_engines else None,
            observed=f"documents={n_documents}; "
            + "; ".join(
                f"{engine_id}={int(site_counts.get(engine_id, 0))}"
                f"+{int(region_counts.get(engine_id, 0))}"
                for engine_id in augmented["engine_id"]
            ),
        ),
        _Criterion(
            key="D_overcorrection_observable",
            question="Can overcorrection still be measured on this pool?",
            threshold=f">= {int(_H2_CLEAN_SHARE_FLOOR * 100)}% sites clean and "
            f">= {_H2_CLEAN_PROPOSAL_FLOOR} clean-site proposals per engine",
            met=d_ok if n_engines else None,
            observed="; ".join(
                f"{engine_id}: clean share {clean_share.get(str(engine_id), 0):.3f}, "
                f"proposals {clean_proposed.get(str(engine_id), 0)}"
                for engine_id in augmented["engine_id"]
            ),
        ),
        _Criterion(
            key="E_natural_pool",
            question="Are the headline candidates naturally generated, not adversarial?",
            threshold="100% pool == natural",
            met=all_natural,
            observed=(
                f"{pool_rows}/{pool_rows} rows natural"
                if all_natural
                else (
                    "pool provenance missing"
                    if all_natural is None
                    else "one or more challenge/diagnostic rows present"
                )
            ),
        ),
        _Criterion(
            key="F_no_unresolved_confound",
            question="Has the alignment twin audit passed with its sensitivity lineage intact?",
            threshold=(
                "audit overall_pass, proposal and audit-output hashes match, structural "
                "audit files and hash-bound no-twin sensitivity manifest exist"
            ),
            met=f_ok if n_engines else None,
            observed=(
                f"{audit_gate_path}; input/output hashes matched; no-twin manifest verified"
                if f_ok
                else (
                    f"gate_pass={bool(audit_gate.get('overall_pass'))}; "
                    f"hashes_match={hashes_match}; missing_audit={missing_audit}; "
                    f"audit_output_hashes_match={audit_output_hashes_match}; "
                    f"missing_sensitivity={missing_sensitivity}; "
                    f"sensitivity_manifest_valid={sensitivity_manifest_valid}"
                )
            ),
        ),
    ]


def evaluate_cgv2_gates(source: Path, role: str = "evaluate") -> dict[str, object]:
    """Apply the frozen §16 criteria to the canonical CGV2 artifacts."""
    suffix = "" if role == "calibrate" else f"__{role}"
    integrity_path = source / "cgv2_integrity_gate.json"
    integrity_payload: dict[str, object] = {}
    if integrity_path.exists():
        try:
            parsed_integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
            integrity_payload = parsed_integrity if isinstance(parsed_integrity, dict) else {}
        except json.JSONDecodeError:
            integrity_payload = {}

    def _read(name: str) -> pd.DataFrame | None:
        path = source / f"{name}{suffix}.csv"
        if not path.exists():
            return None
        frame = pd.read_csv(path, keep_default_na=False)
        # keep_default_na=False keeps empty cells as "" strings, which turns float
        # columns object-dtype; coerce back so the comparisons below are numeric.
        for column in frame.columns:
            if column == "reject_holm":
                normalized = frame[column].astype(str).str.strip().str.lower()
                frame[column] = normalized.map({"true": True, "false": False})
            elif column.endswith(
                (
                    "_value",
                    "_lower",
                    "_upper",
                    "_holm",
                    "_rate",
                    "_share",
                    "_coverage",
                    "_ratio",
                    "_edits",
                    "_proposed",
                    "_sites",
                )
            ) or column in ("delta", "p_value", "availability", "n_candidates", "k", "d_before"):
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame

    contrast = _read("cgv2_contrasts")
    quality = _read("cgv2_quality")
    preservation = _read("cgv2_preservation")
    opportunity = _read("cgv2_opportunity")
    oracle = _read("cgv2_oracle")
    census = _read("cgv2_site_census")
    proposals = _read("cgv2_proposals")
    region_proposals = _read("cgv2_region_proposals")

    audit_dir = source / "audits" / role
    audit_paths = {
        "alignment_site_audit": audit_dir / "alignment_site_audit.csv",
        "site_projection_audit": audit_dir / "site_projection_audit.csv",
        "region_projection_audit": audit_dir / "region_projection_audit.csv",
    }
    audit_missing = [name for name, path in audit_paths.items() if not path.exists()]
    if (
        not audit_missing
        and census is not None
        and proposals is not None
        and region_proposals is not None
    ):
        census, proposals, cleaned_regions, _audit_counts = cgv2_tables.audit_clean_pool(
            census,
            proposals,
            region_proposals,
            pd.read_csv(audit_paths["alignment_site_audit"], keep_default_na=False),
            pd.read_csv(audit_paths["site_projection_audit"], keep_default_na=False),
            pd.read_csv(audit_paths["region_projection_audit"], keep_default_na=False),
        )
        region_proposals = cleaned_regions if cleaned_regions is not None else pd.DataFrame()

    def _headline(frame: pd.DataFrame | None) -> pd.DataFrame | None:
        if frame is None or frame.empty or "dataset_id" not in frame.columns:
            return frame
        return frame[~frame["dataset_id"].astype(str).isin(HEADLINE_EXCLUDED_DATASETS)]

    # Canonical study CSVs intentionally retain the OCR-D stress track. Gate inputs do
    # not: protocol section 4 excludes it from headline engine-transfer aggregates.
    census = _headline(census)
    proposals = _headline(proposals)
    region_proposals = _headline(region_proposals)

    missing = [
        name
        for name, frame in (
            ("contrasts", contrast),
            ("quality", quality),
            ("preservation", preservation),
            ("opportunity", opportunity),
            ("oracle", oracle),
            ("census", census),
            ("proposals", proposals),
        )
        if frame is None
    ]
    missing.extend(audit_missing)
    if region_proposals is None:
        region_proposals = pd.DataFrame()
    # The frozen per-rung candidate caps are gate evidence: without the recorded column
    # the pools cannot be bounded, and an unbounded pool is missing evidence, not a
    # measurably smaller one.
    for name, frame in (("proposals", proposals), ("region_proposals", region_proposals)):
        if frame is not None and not frame.empty and "generator_native_cap" not in frame.columns:
            missing.append(f"{name}: generator_native_cap")

    generator_criteria: list[_Criterion] = []
    if missing:
        generator_criteria.append(
            _Criterion(
                key="evidence",
                question="Are the study artifacts present to evaluate any criterion?",
                threshold="all canonical CSVs exist",
                met=None,
                observed=f"missing: {', '.join(missing)}",
            )
        )
        cg1 = cg2 = cg3 = None
        h2_criteria: list[_Criterion] = generator_criteria
    else:
        assert contrast is not None and quality is not None and preservation is not None
        assert opportunity is not None and oracle is not None and census is not None
        cg1 = _criterion_cg1(contrast)
        cg2 = _criterion_cg2(quality, preservation)
        cg3 = _criterion_cg3(contrast, opportunity)
        generator_criteria = [cg1, cg2, cg3]
        assert proposals is not None
        h2_criteria = _h2_criteria(
            oracle,
            quality,
            preservation,
            census,
            proposals,
            region_proposals,
            audit_dir,
            source,
            suffix,
        )

    integrity_met = bool(integrity_payload.get("overall_pass")) if integrity_payload else None
    raw_blocking_findings = integrity_payload.get("blocking_findings", [])
    blocking_findings = raw_blocking_findings if isinstance(raw_blocking_findings, list) else []
    integrity_observed = (
        f"overall_pass={integrity_met}; blocking_findings={len(blocking_findings)}; "
        f"confirmatory_status={integrity_payload.get('confirmatory_status', 'unknown')}"
        if integrity_payload
        else f"missing or invalid {integrity_path}"
    )
    generator_criteria.append(
        _Criterion(
            key="VALIDITY_integrity_review",
            question="Did the independent integrity review validate the evidence chain?",
            threshold="machine-readable integrity gate passes with no Critical/High blocker",
            met=integrity_met,
            observed=integrity_observed,
        )
    )
    h2_criteria.append(
        _Criterion(
            key="G_integrity_validity",
            question="Is this candidate pool valid evidence for a deployable H2 verifier?",
            threshold="integrity gate passes; natural-pool semantics and fresh evaluation hold",
            met=integrity_met,
            observed=integrity_observed,
        )
    )

    h2_met = [c for c in h2_criteria if c.met]
    h2_all_measurable = all(c.met is not None for c in h2_criteria)
    if not h2_all_measurable:
        h2_verdict = "H2 INCONCLUSIVE"
    elif len(h2_met) == len(h2_criteria):
        h2_verdict = "H2 READY"
    else:
        h2_verdict = "H2 NOT READY"

    criterion_a = next((c for c in h2_criteria if c.key == "A_repair_opportunity"), None)
    criterion_b = next((c for c in h2_criteria if c.key == "B_oracle_safe_coverage"), None)
    cg4 = _Criterion(
        key="CG4_h2_floor",
        question="Does the CGV2 pool meet the frozen H2 criteria A and B?",
        threshold="A and B met on >= 3/4 engines",
        met=(
            any(c.key == "A_repair_opportunity" and c.met for c in h2_criteria)
            and any(c.key == "B_oracle_safe_coverage" and c.met for c in h2_criteria)
        )
        if h2_all_measurable
        else None,
        observed=(
            f"A: {criterion_a.observed if criterion_a else 'not produced'}; "
            f"B: {criterion_b.observed if criterion_b else 'not produced'}; {h2_verdict}"
        ),
    )
    generator_criteria = [*generator_criteria, cg4]

    n_core_met = sum(
        1
        for c in generator_criteria
        if c.key.startswith("CG") and c.key != "CG4_h2_floor" and c.met
    )
    any_evidence_missing = any(c.met is None for c in generator_criteria)
    if any_evidence_missing:
        verdict = "INCONCLUSIVE"
    elif all(c.met for c in generator_criteria):
        verdict = "GENERATOR READY"
    elif integrity_met is False:
        # A produced-but-failed integrity review floors the verdict: a pool whose
        # evidence chain is invalid cannot be lifted to PARTIALLY READY by its numbers.
        verdict = "GENERATOR NOT READY"
    elif n_core_met >= 1:
        verdict = "GENERATOR PARTIALLY READY"
    else:
        verdict = "GENERATOR NOT READY"

    input_paths = [
        source / f"{name}{suffix}.csv"
        for name in (
            "cgv2_contrasts",
            "cgv2_quality",
            "cgv2_preservation",
            "cgv2_opportunity",
            "cgv2_oracle",
            "cgv2_site_census",
            "cgv2_proposals",
            "cgv2_region_proposals",
        )
    ]
    return {
        "gate": "cgv2_generator",
        "verdict": verdict,
        "role": role,
        "criteria": [
            {
                "key": c.key,
                "question": c.question,
                "threshold": c.threshold,
                "met": c.met,
                "observed": c.observed,
            }
            for c in generator_criteria
        ],
        "h2_readiness": {
            "gate": "cgv2_h2_readiness",
            "verdict": h2_verdict,
            "criteria": [
                {
                    "key": c.key,
                    "question": c.question,
                    "threshold": c.threshold,
                    "met": c.met,
                    "observed": c.observed,
                }
                for c in h2_criteria
            ],
        },
        "notes": [
            "CG1-CG4 are frozen in docs/cgv2/protocol.md section 16, written before any "
            "CGV2 evaluate-role artifact existed. VALIDITY_integrity_review, criterion G, "
            "and the verdict floor are amendment A7 (post-evaluate, review-forced); the "
            "integrity gate separately records whether the reused partition supports a "
            "confirmatory claim.",
            "The historical gates in results/generated/ are untouched; CGV2 writes only "
            "under results/generated/cgv2/.",
        ],
        "inputs": {path.name: file_sha256(path) for path in input_paths if path.is_file()},
    }
