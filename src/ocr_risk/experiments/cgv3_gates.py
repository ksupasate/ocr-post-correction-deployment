"""The CGV3 confirmatory gates (protocol section 14), from canonical artifacts only.

Three gates, one module, no thresholds of its own: the S1/S2 site-recall floors, K,
and the statistical plan are read from ``docs/cgv3/freeze2.json``; the H2 constants
come from their single CGV2 definition; every criterion reads a canonical artifact of
the confirmatory run and fails closed (``met = None`` -> INCONCLUSIVE) when its
evidence file is absent. The verdict logic mirrors ``cgv2_gates`` exactly: a failed
integrity review floors the verdict, evidence gaps make it INCONCLUSIVE, and nothing
here can outvote a missing artifact into a pass.

The H2 A-F criteria keep CGV2's thresholds unchanged and count them on the DISCOVERED
pool (the protocol's counted-on amendment): A and B from ``cgv3_oracle`` (g8, K=4),
C-E from the candidate and region-truth tables, F from the fresh alignment audit
gate whose lineage is re-verified against the live artifact store.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ocr_risk.experiments.cgv2_gates import (
    _CLEAN_RATE_BOUND,
    _ENGINES_REQUIRED,
    _H2_ACCEPTED_EDITS_FLOOR,
    _H2_CANDIDATE_FLOOR,
    _H2_CLEAN_PROPOSAL_FLOOR,
    _H2_CLEAN_SHARE_FLOOR,
    _H2_DOCUMENT_FLOOR,
    _H2_SAFE_COVERAGE_FLOOR,
    _HARM_RATIO_MULTIPLIER,
)
from ocr_risk.experiments.cgv3_confirmatory import read_frozen_csv

__all__ = [
    "evaluate_cgv3_generator_gate",
    "evaluate_cgv3_h2_readiness",
    "evaluate_cgv3_integrity",
]

_BENEFICIAL = ("true_correction", "partial_improvement")
_HARMFUL = ("miscorrection", "overcorrection")
_STRUCTURAL_CLASSES = ("deletion", "insertion", "segmentation")
_ENGINES = ("doctr", "easyocr", "paddleocr", "tesseract")


@dataclass(frozen=True, slots=True)
class _Criterion:
    key: str
    question: str
    threshold: str
    met: bool | None
    observed: str


def _read_frame(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    # DEFECT-3: pandas' default float parsing is 1-ULP lossy; a metric sitting on a
    # floor boundary must be read back at full precision or the gate could flip.
    return read_frozen_csv(path)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _engine_value(
    frame: pd.DataFrame, column: str, generator: str | None = None
) -> dict[str, float]:
    """Column values per engine, optionally restricted to one generator's rows."""
    if generator is not None and "generator_id" in frame.columns:
        frame = frame[frame["generator_id"] == generator]
    values: dict[str, float] = {}
    for engine_id, group in frame.groupby("engine_id"):
        numeric = pd.to_numeric(group[column], errors="coerce")
        numeric = numeric[numeric.notna()]
        if numeric.empty:
            continue
        values[str(engine_id)] = float(numeric.iloc[0])
    return values


def _class_recall(coverage: pd.DataFrame) -> dict[tuple[str, str], float]:
    """Site recall per (engine, site_kind) over eligible regions -- the S2 quantity."""
    eligible = coverage[coverage["audit_state"] == "eligible"]
    recalls: dict[tuple[str, str], float] = {}
    for (engine_id, site_kind), group in eligible.groupby(["engine_id", "site_kind"]):
        discovered = int((group["disposition"] == "discovered").sum())
        recalls[(str(engine_id), str(site_kind))] = (
            discovered / len(group) if len(group) else float("nan")
        )
    return recalls


def evaluate_cgv3_generator_gate(
    source: Path,
    freeze: dict[str, Any],
    integrity_pass: bool | None,
) -> dict[str, Any]:
    """Apply the frozen section-14 generator criteria to the confirmatory frames."""
    coverage = _read_frame(source / "cgv3_true_coverage.csv")
    candidates = _read_frame(source / "cgv3_candidates.csv")
    site_metrics = _read_frame(source / "cgv3_site_metrics.csv")
    harm = _read_frame(source / "cgv3_harm.csv")
    contrasts = _read_frame(source / "statistics" / "cgv3_contrasts.csv")
    region_truth = _read_frame(source / "cgv3_discovered_region_truth.csv")
    statistics_record = _read_json(source / "statistics" / "statistics_record.json")

    floors = freeze.get("site_recall_floors", {})
    s1_floor = float(floors.get("s1_overall", 0.0))
    s2_floors = {k: float(v) for k, v in floors.get("s2_by_class", {}).items()}

    criteria: list[_Criterion] = []

    if site_metrics is None or coverage is None:
        criteria.append(
            _Criterion(
                key="evidence",
                question="Are the confirmatory study artifacts present?",
                threshold="all canonical frames exist",
                met=None,
                observed=f"missing: {source}",
            )
        )
    else:
        recall = _engine_value(site_metrics, "site_recall")
        s1_met = bool(recall) and all(
            recall.get(engine, float("-inf")) >= s1_floor for engine in _ENGINES
        )
        criteria.append(
            _Criterion(
                key="S1_overall_site_recall",
                question="Does OCR-only discovery find enough repair sites on every engine?",
                threshold=f"site recall >= {s1_floor} on 4/4 engines (no 3/4 allowance)",
                met=s1_met if recall else None,
                observed="; ".join(
                    f"{engine}={recall.get(engine, float('nan')):.4f}" for engine in _ENGINES
                ),
            )
        )
        class_recalls = _class_recall(coverage)
        # An (engine, class) with no eligible regions is unmeasured, not passed:
        # NaN comparisons are False, so absent pairs must force INCONCLUSIVE explicitly.
        required_pairs = {(engine, kind) for engine in _ENGINES for kind in s2_floors}
        missing_pairs = sorted(required_pairs - set(class_recalls))
        failed_classes = [
            f"{engine}/{kind}={value:.4f}<{floor}"
            for engine in _ENGINES
            for kind, floor in sorted(s2_floors.items())
            if (value := class_recalls.get((engine, kind), float("nan"))) < floor
        ]
        s2_met = (not failed_classes and not missing_pairs) if class_recalls else None
        criteria.append(
            _Criterion(
                key="S2_structural_site_recall",
                question="Does discovery find enough sites in each structural class?",
                threshold="; ".join(f"{k}>={v}" for k, v in sorted(s2_floors.items()))
                + " on 4/4 engines (no 3/4 allowance)",
                met=s2_met,
                observed=(
                    "all engine x class floors met"
                    if s2_met
                    else (
                        f"{len(missing_pairs)} engine x class cells with no eligible regions: "
                        + "; ".join(f"{e}/{k}" for e, k in missing_pairs[:6])
                        if missing_pairs and not failed_classes
                        else f"{len(failed_classes)} below floor: " + "; ".join(failed_classes[:8])
                    )
                ),
            )
        )

    # CG1/CG3 read the frozen contrasts; absent statistics evidence fails closed.
    if contrasts is None:
        for key in ("CG1_availability_contrast", "CG3_structural_coverage"):
            criteria.append(
                _Criterion(
                    key=key,
                    question="Frozen section 14 generator criterion.",
                    threshold="see docs/cgv3/protocol.md section 14",
                    met=None,
                    observed="missing statistics/cgv3_contrasts.csv",
                )
            )
    else:
        primary = contrasts[contrasts["contrast"] == "g8_union_vs_g3_edit_aware__end_to_end"]
        passing = primary[
            (pd.to_numeric(primary["ci_lower"], errors="coerce") > 0)
            & (primary["reject_holm"].astype(bool))
        ]
        criteria.append(
            _Criterion(
                key="CG1_availability_contrast",
                question=(
                    "Does the structural union raise end-to-end opportunity over the incumbent?"
                ),
                threshold="Holm-adjusted CI excluding 0, positive, on >= 3/4 engines",
                met=(len(passing) >= _ENGINES_REQUIRED) if len(primary) else None,
                observed=f"{len(passing)}/{len(primary)} engines; "
                + "; ".join(
                    f"{engine}: d={float(delta):+.4f} "
                    f"CI[{float(lower):+.4f},{float(upper):+.4f}] "
                    f"p_holm={float(p_holm):.4f}"
                    for engine, delta, lower, upper, p_holm in zip(
                        primary["engine_id"].astype(str),
                        pd.to_numeric(primary["delta"], errors="coerce"),
                        pd.to_numeric(primary["ci_lower"], errors="coerce"),
                        pd.to_numeric(primary["ci_upper"], errors="coerce"),
                        pd.to_numeric(primary["p_holm"], errors="coerce"),
                        strict=True,
                    )
                ),
            )
        )
        structural = contrasts[
            contrasts["contrast"] == "g8_union_vs_g3_edit_aware__structural_strata"
        ]
        structural_passing = structural[
            (pd.to_numeric(structural["ci_lower"], errors="coerce") > 0)
            & (structural["reject_holm"].astype(bool))
        ]
        insertion_counts: dict[str, int] = {}
        if coverage is not None and candidates is not None and not candidates.empty:
            eligible = coverage[coverage["audit_state"] == "eligible"]
            # Protocol section 8: insertion-REPAIR regions are deletion-kind true
            # sites; an insertion at a GAP anchor is what repairs them.
            repair = eligible[eligible["site_kind"] == "deletion"]
            # g8 rows are within K by construction (the union is cap-forced at
            # freeze generators.union_cap; K=4 truncates nothing).
            union = candidates[candidates["generator_id"] == "g8_union"]
            beneficial = set(union[union["outcome"].isin(_BENEFICIAL)]["site_id"].astype(str))
            hits = repair[
                (repair["disposition"] == "discovered")
                & repair["matched_site_id"].astype(str).isin(beneficial)
            ]
            insertion_counts = {
                str(engine_id): len(group) for engine_id, group in hits.groupby("engine_id")
            }
        insertion_total = len(insertion_counts)
        insertion_positive = sum(1 for count in insertion_counts.values() if count > 0)
        structural_ok = len(structural) > 0 and len(structural_passing) >= _ENGINES_REQUIRED
        insertion_ok = insertion_total > 0 and insertion_positive >= _ENGINES_REQUIRED
        criteria.append(
            _Criterion(
                key="CG3_structural_coverage",
                question=(
                    "Does the structural stratum gain coverage, from exactly zero on insertions?"
                ),
                threshold="structural CI positive on >= 3/4 AND insertion-repair regions "
                "with a beneficial g8 candidate on >= 3/4",
                met=(
                    (structural_ok and insertion_ok)
                    if len(structural) and insertion_total
                    else None
                ),
                observed=(
                    f"structural {len(structural_passing)}/{len(structural)}; "
                    f"insertion-repair > 0 on {insertion_positive}/{insertion_total}: "
                    + "; ".join(
                        f"{engine}={count}" for engine, count in sorted(insertion_counts.items())
                    )
                ),
            )
        )

    # CG2 is CGV2's single conjunction: the harm ratio bound AND the clean-site
    # proposal bound, both per engine (CGV2-H3's two bounds on 4/4 engines).
    if (
        harm is None
        or harm.empty
        or region_truth is None
        or region_truth.empty
        or candidates is None
        or candidates.empty
    ):
        criteria.append(
            _Criterion(
                key="CG2_harm_bounded",
                question="Is the union's harmful burden bounded and its clean-site touch limited?",
                threshold=f"ratio <= {_HARM_RATIO_MULTIPLIER}x g3's and clean-span proposal "
                f"rate <= {_CLEAN_RATE_BOUND} per engine on 4/4",
                met=None,
                observed="missing harm, region-truth, or candidate frame",
            )
        )
    else:
        present = region_truth[region_truth["index_present"].astype(bool)]
        clean_by_engine = {
            str(engine_id): set(group.loc[group["d_before"] == 0, "site_id"].astype(str))
            for engine_id, group in present.groupby("engine_id")
        }
        union_rows = candidates[candidates["generator_id"] == "g8_union"]
        observations: list[str] = []
        cg2_met = True
        for engine_id in _ENGINES:
            union_row = harm[
                (harm["generator_id"] == "g8_union") & (harm["engine_id"] == engine_id)
            ]
            g3_row = harm[
                (harm["generator_id"] == "g3_edit_aware") & (harm["engine_id"] == engine_id)
            ]
            clean_ids = clean_by_engine.get(engine_id, set())
            engine_union = union_rows[union_rows["engine_id"] == engine_id]
            proposed_clean = engine_union.loc[
                engine_union["site_id"].astype(str).isin(clean_ids), "site_id"
            ].nunique()
            if union_row.empty or g3_row.empty or not clean_ids:
                cg2_met = False
                observations.append(f"{engine_id}: missing harm rows or no clean sites")
                continue
            ratio = float(union_row["harmful_beneficial_ratio"].iloc[0])
            baseline = float(g3_row["harmful_beneficial_ratio"].iloc[0])
            rate = proposed_clean / len(clean_ids)
            ratio_ok = ratio <= _HARM_RATIO_MULTIPLIER * max(baseline, 1e-9) or (
                ratio == float("inf") and baseline == float("inf")
            )
            cg2_met = cg2_met and ratio_ok and rate <= _CLEAN_RATE_BOUND
            observations.append(
                f"{engine_id}: ratio {ratio:.2f} vs 2x{baseline:.2f}, "
                f"clean-site proposal rate {rate:.3f}"
            )
        criteria.append(
            _Criterion(
                key="CG2_harm_bounded",
                question="Is the union's harmful burden bounded and its clean-site touch limited?",
                threshold=f"ratio <= {_HARM_RATIO_MULTIPLIER}x g3's and clean-span proposal "
                f"rate <= {_CLEAN_RATE_BOUND} per engine on 4/4",
                met=cg2_met,
                observed="; ".join(observations),
            )
        )

    h2 = evaluate_cgv3_h2_readiness(source, integrity_pass)
    h2_criteria = {c["key"]: c["met"] for c in h2["criteria"]}
    criteria.append(
        _Criterion(
            key="CG4_h2_floor",
            question="Does the discovered pool meet the frozen H2 criteria A and B?",
            threshold="A and B met on >= 3/4 engines",
            met=(
                bool(h2_criteria.get("A_repair_opportunity"))
                and bool(h2_criteria.get("B_oracle_safe_coverage"))
            )
            if h2_criteria
            else None,
            observed=f"H2 verdict: {h2['verdict']}",
        )
    )
    criteria.append(
        _Criterion(
            key="VALIDITY_integrity_review",
            question="Did the fresh integrity gate validate the evidence chain?",
            threshold="integrity gate passes with no unresolved Critical finding",
            met=integrity_pass,
            observed=f"integrity overall_pass={integrity_pass}",
        )
    )

    any_missing = any(c.met is None for c in criteria)
    core = [c for c in criteria if c.key.startswith(("CG", "S")) and c.key != "CG4_h2_floor"]
    n_core_met = sum(1 for c in core if c.met)
    if any_missing:
        verdict = "INCONCLUSIVE"
    elif all(c.met for c in criteria):
        verdict = "GENERATOR READY"
    elif integrity_pass is False:
        verdict = "GENERATOR NOT READY"
    elif n_core_met >= 1:
        verdict = "GENERATOR PARTIALLY READY"
    else:
        verdict = "GENERATOR NOT READY"

    return {
        "gate": "cgv3_generator",
        "verdict": verdict,
        "criteria": [asdict(c) for c in criteria],
        "statistics_record_present": statistics_record is not None,
        "inputs": {
            "frames": str(source),
            "freeze_sha256": freeze.get("sha256", ""),
        },
    }


def evaluate_cgv3_h2_readiness(source: Path, integrity_pass: bool | None) -> dict[str, Any]:
    """The frozen CGV2 A-F criteria, counted on the CGV3 discovered pool."""
    oracle = _read_frame(source / "cgv3_oracle.csv")
    candidates = _read_frame(source / "cgv3_candidates.csv")
    region_truth = _read_frame(source / "cgv3_discovered_region_truth.csv")
    audit_gate = _read_json(source / "audits" / "evaluate" / "alignment_audit_gate.json")

    criteria: list[_Criterion] = []
    if oracle is None:
        criteria.append(
            _Criterion(
                key="evidence",
                question="Are the confirmatory artifacts present to evaluate readiness?",
                threshold="oracle, candidates, region truth, coverage frames exist",
                met=None,
                observed=f"missing oracle frame under {source}",
            )
        )
        return {
            "gate": "cgv3_h2_readiness",
            "verdict": "H2 INCONCLUSIVE",
            "criteria": [asdict(c) for c in criteria],
        }

    augmented = (
        oracle[oracle["generator_id"] == "g8_union"] if "generator_id" in oracle.columns else oracle
    )
    a_pass = int(
        (pd.to_numeric(augmented["oracle_accepted_edits"]) >= _H2_ACCEPTED_EDITS_FLOOR).sum()
    )
    b_pass = int(
        (pd.to_numeric(augmented["oracle_safe_coverage"]) >= _H2_SAFE_COVERAGE_FLOOR).sum()
    )
    n_engines = len(augmented)
    criteria.append(
        _Criterion(
            key="A_repair_opportunity",
            question=(
                "Does the discovered pool admit enough beneficial edits to certify any target?"
            ),
            threshold=f">= {_H2_ACCEPTED_EDITS_FLOOR} oracle-accepted edits on >= 3/4 engines",
            met=(a_pass >= _ENGINES_REQUIRED) if n_engines else None,
            observed=f"{a_pass}/{n_engines} engines; "
            + "; ".join(
                f"{engine}={int(edits)}"
                for engine, edits in zip(
                    augmented["engine_id"].astype(str),
                    pd.to_numeric(augmented["oracle_accepted_edits"], errors="coerce"),
                    strict=True,
                )
            ),
        )
    )
    criteria.append(
        _Criterion(
            key="B_oracle_safe_coverage",
            question="Would a perfect verifier trace a risk-coverage curve worth reading?",
            threshold=f"oracle safe coverage >= {_H2_SAFE_COVERAGE_FLOOR} on >= 3/4 engines",
            met=(b_pass >= _ENGINES_REQUIRED) if n_engines else None,
            observed=f"{b_pass}/{n_engines} engines; "
            + "; ".join(
                f"{engine}={float(coverage_value):.4f}"
                for engine, coverage_value in zip(
                    augmented["engine_id"].astype(str),
                    pd.to_numeric(augmented["oracle_safe_coverage"], errors="coerce"),
                    strict=True,
                )
            ),
        )
    )

    natural_counts: dict[str, int] = {}
    if candidates is not None and not candidates.empty:
        union = candidates[candidates["generator_id"] == "g8_union"]
        natural_counts = {
            str(engine_id): len(group) for engine_id, group in union.groupby("engine_id")
        }
    # Document clusters counted on the discovered-site census (CGV2 counted the
    # site census, not the eligible-region table).
    n_documents = (
        int(region_truth["document_id"].nunique())
        if region_truth is not None and not region_truth.empty
        else 0
    )
    c_ok = (
        bool(natural_counts)
        and n_documents >= _H2_DOCUMENT_FLOOR
        and all(count >= _H2_CANDIDATE_FLOOR for count in natural_counts.values())
    )
    criteria.append(
        _Criterion(
            key="C_sample_size",
            question="Are there enough candidates and enough document clusters?",
            threshold=f">= {_H2_CANDIDATE_FLOOR} natural g8 candidates per engine, "
            f">= {_H2_DOCUMENT_FLOOR} documents",
            met=c_ok if natural_counts else None,
            observed=f"documents={n_documents}; "
            + "; ".join(f"{e}={c}" for e, c in sorted(natural_counts.items())),
        )
    )

    # DEFECT-4 (fresh-reviewer finding D-1, post-run hardening): D and E are
    # appended even when their evidence files are absent, with met=None -> H2
    # INCONCLUSIVE. Previously a missing region-truth table silently skipped D, so
    # A/B/C/F/G alone could return H2 READY without ever evaluating overcorrection
    # observability -- a fail-open path contradicting the module docstring. It did
    # not fire in this run (both tables present, D evaluated and failed 0/4).
    if region_truth is not None and not region_truth.empty and candidates is not None:
        present = region_truth[region_truth["index_present"].astype(bool)]
        clean_sites = {
            f"{row.site_id}" for row in present.itertuples(index=False) if row.d_before == 0
        }
        clean_share = {
            str(engine_id): float((group["d_before"] == 0).mean())
            for engine_id, group in present.groupby("engine_id")
        }
        union_rows = candidates[candidates["generator_id"] == "g8_union"]
        clean_proposed = {
            str(engine_id): int(
                group.loc[group["site_id"].astype(str).isin(clean_sites), "site_id"].nunique()
            )
            for engine_id, group in union_rows.groupby("engine_id")
        }
        d_ok = bool(clean_share) and all(
            clean_share.get(engine, 0.0) >= _H2_CLEAN_SHARE_FLOOR
            and clean_proposed.get(engine, 0) >= _H2_CLEAN_PROPOSAL_FLOOR
            for engine in clean_share
        )
        d_met: bool | None = d_ok if clean_share else None
        d_observed = "; ".join(
            f"{engine}: clean share {clean_share.get(engine, 0):.3f}, "
            f"proposals {clean_proposed.get(engine, 0)}"
            for engine in sorted(clean_share)
        )
    else:
        d_met = None
        d_observed = "region-truth table or candidate table absent -- not evaluable"
    criteria.append(
        _Criterion(
            key="D_overcorrection_observable",
            question="Can overcorrection still be measured on this pool?",
            threshold=f">= {int(_H2_CLEAN_SHARE_FLOOR * 100)}% clean anchors and "
            f">= {_H2_CLEAN_PROPOSAL_FLOOR} clean-site g8 proposals per engine",
            met=d_met,
            observed=d_observed,
        )
    )

    if candidates is not None:
        generator_ids = set(candidates["generator_id"].astype(str))
        e_met: bool | None = generator_ids <= {
            "b0_conf_only",
            "g3_edit_aware",
            "g7_structural_v2",
            "g8_union",
        }
        e_observed = f"generator ids: {sorted(generator_ids)}"
    else:
        e_met = None
        e_observed = "candidate table absent -- not evaluable"
    criteria.append(
        _Criterion(
            key="E_natural_pool",
            question="Are the candidates naturally generated, not adversarial?",
            threshold="only natural rungs (b0/g3/g7/g8) present; no challenge/oracle rows",
            met=e_met,
            observed=e_observed,
        )
    )

    if audit_gate is not None:
        # DEFECT-1: the CGV2 audit gate's overall_pass is structurally False without
        # CGV2 proposals files, and CGV3 has no CGV2 proposals layer -- its candidates
        # hang off discovered-site ids (a different namespace) and its projection
        # semantics are the A1 anchor-kind-strict coverage match, audited separately.
        # F therefore requires the audit to exist, record its four input runs, and
        # pass the three denominator reconciliation checks; the proposals-projection
        # checks are recorded as not applicable, and any failing applicable check or
        # missing lineage still fails F closed.
        recorded_runs = audit_gate.get("input_runs", {})
        lineage_ok = (
            isinstance(recorded_runs, dict)
            and {"align", "sites", "canonicalize", "manifest"} <= set(recorded_runs)
            and all(
                str(recorded_runs[name]) for name in ("align", "sites", "canonicalize", "manifest")
            )
        )
        checks = audit_gate.get("checks", {})
        if not isinstance(checks, dict):
            checks = {}
        applicable = (
            "component_denominators_reconcile",
            "site_denominators_reconcile",
            "region_denominators_reconcile",
        )
        applicable_values = {name: checks.get(name) for name in applicable}
        not_applicable = sorted(set(checks) - set(applicable))
        applicable_ok = all(value is True for value in applicable_values.values())
        criteria.append(
            _Criterion(
                key="F_no_unresolved_confound",
                question="Has the fresh alignment audit passed with its lineage intact?",
                threshold="audit gate present; align/sites/canonicalize/manifest runs "
                "recorded and matching the live store; the three denominator "
                "reconciliation checks true (CGV2 proposals-projection checks not "
                "applicable to CGV3 -- DEFECT-1)",
                met=lineage_ok and applicable_ok,
                observed=(
                    f"input_runs={recorded_runs}; applicable={applicable_values}; "
                    f"not_applicable={not_applicable} (no CGV2 proposals layer in CGV3)"
                ),
            )
        )
    else:
        criteria.append(
            _Criterion(
                key="F_no_unresolved_confound",
                question="Has the fresh alignment audit passed with its lineage intact?",
                threshold="alignment_audit_gate.json present and passing",
                met=None,
                observed=f"missing {source / 'audits' / 'evaluate' / 'alignment_audit_gate.json'}",
            )
        )
    criteria.append(
        _Criterion(
            key="G_integrity_validity",
            question="Is this pool valid evidence for a deployable H2 verifier?",
            threshold="fresh integrity gate passes; freshness, blindness, lineage all hold",
            met=integrity_pass,
            observed=f"integrity overall_pass={integrity_pass}",
        )
    )

    h2_met = [c for c in criteria if c.met]
    all_measurable = all(c.met is not None for c in criteria)
    if not all_measurable:
        verdict = "H2 INCONCLUSIVE"
    elif len(h2_met) == len(criteria):
        verdict = "H2 READY"
    else:
        verdict = "H2 NOT READY"
    return {
        "gate": "cgv3_h2_readiness",
        "verdict": verdict,
        "criteria": [asdict(c) for c in criteria],
        "reminder": (
            "H2 READY unlocks, never performs, H2; it is not evidence H2 would be supported."
        ),
    }


def evaluate_cgv3_integrity(source: Path, expected: dict[str, Any]) -> dict[str, Any]:
    """The fresh confirmatory integrity gate: every check from recorded artifacts.

    ``expected`` carries the run's own bindings (HEAD, seal and freeze hashes, the
    execution-complete manifest, the fingerprint census, the fold-corpus definition)
    so every check is a comparison between two independently recorded states, never
    a re-derivation from memory.
    """
    checks: dict[str, dict[str, Any]] = {}

    def record(key: str, passed: bool | None, observed: str) -> None:
        checks[key] = {"pass": passed, "observed": observed}

    snapshot = _read_json(source / "pre_access_snapshot.json")
    unlock = _read_json(source / "unlock_record.json")
    execution = _read_json(source / "execution_complete_manifest.json")
    r65 = _read_json(source / "r65_certificate.json")
    census = _read_json(source / "ocr_census.json")
    statistics_record = _read_json(source / "statistics" / "statistics_record.json")

    import hashlib

    def sha(path: Path) -> str | None:
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None

    record(
        "freshness_snapshot_present",
        snapshot is not None and bool(snapshot.get("all_previously_untouched")),
        f"snapshot={snapshot is not None}",
    )
    snapshot_sha_match = (
        unlock.get("pre_access_snapshot_sha256") == sha(source / "pre_access_snapshot.json")
        if unlock
        else None
    )
    # DEFECT-5 (fresh-reviewer finding D-6): the pass bit now includes the snapshot
    # hash match it already reports. Previously a tampered pre-access snapshot passed
    # this check with "matches=False" buried in the observed string.
    record(
        "unlock_record_present",
        unlock is not None and bool(unlock.get("unlocked")) and snapshot_sha_match is True,
        f"unlock={unlock is not None}; snapshot_sha matches={snapshot_sha_match}",
    )
    seal_paths_match = expected.get("sealed_sources_unchanged")
    record(
        "sealed_sources_unchanged",
        bool(seal_paths_match),
        f"sealed source hashes unchanged vs pre-confirmatory seal: {seal_paths_match}",
    )
    record(
        "ocr_census_complete",
        census is not None
        and census.get("expected_pairs") == census.get("present_pairs")
        and census.get("fingerprint_parity") is True,
        f"census={json.dumps(census)[:200] if census else None}",
    )
    record(
        "execution_manifest_binds_tables",
        execution is not None
        and sha(source / "cgv3_fresh_sites.csv") == execution.get("sites_sha256")
        and sha(source / "cgv3_fresh_candidates_pre_gt.csv") == execution.get("candidates_sha256"),
        "site and candidate table hashes bound by the execution-complete manifest",
    )
    record(
        "r65_fresh_certificate_closed",
        r65 is not None and r65.get("status") == "CLOSED",
        f"fresh R-65 certificate status={r65.get('status') if r65 else None}",
    )
    lineage_mismatches = expected.get("audit_lineage_mismatches")
    record(
        "audit_lineage_matches_live_store",
        lineage_mismatches == [],
        f"alignment-audit input runs vs live store: {lineage_mismatches}",
    )
    record(
        "fold_corpus_reserve_free",
        expected.get("fold_corpus_reserve_intersection") == [],
        f"fold corpus intersect reserve: {expected.get('fold_corpus_reserve_intersection')}",
    )
    record(
        "challenge_pool_absent",
        expected.get("generator_ids") is not None
        and set(expected["generator_ids"])
        <= {"b0_conf_only", "g3_edit_aware", "g7_structural_v2", "g8_union"},
        f"generator ids: {expected.get('generator_ids')}",
    )
    record(
        "statistics_plan_frozen",
        statistics_record is not None
        and statistics_record.get("n_resamples") == 10000
        and statistics_record.get("seed") == 7
        and statistics_record.get("unit") == "document"
        and statistics_record.get("multiplicity") == "holm_within_four_engine_family",
        f"record={statistics_record is not None}",
    )
    record(
        "confirmatory_role_labelled",
        expected.get("analysis_roles") is not None
        and set(expected["analysis_roles"]) == {"confirmatory"},
        f"analysis roles: {expected.get('analysis_roles')}",
    )
    unresolved_critical = expected.get("unresolved_critical_findings", [])
    record(
        "no_unresolved_critical_finding",
        not unresolved_critical,
        f"unresolved Critical findings: {unresolved_critical}",
    )

    overall = all(item["pass"] is True for item in checks.values()) and not unresolved_critical
    measurable = all(item["pass"] is not None for item in checks.values())
    return {
        "gate": "cgv3_integrity",
        "verdict": "PASS" if overall else ("INCONCLUSIVE" if not measurable else "FAIL"),
        "checks": checks,
        "confirmatory_status": "fresh_one_shot" if overall else "invalid_or_incomplete",
    }
