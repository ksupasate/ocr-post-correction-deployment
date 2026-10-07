"""The CGV2 integrity gate: mechanical validity checks plus recorded review findings.

The generator and H2 gates read numbers; this gate reads *validity*. It re-derives what
can be derived mechanically (pool provenance, role isolation, fold-scoped fitted
resources, selection scope, partition freshness) from the immutable study artifacts, and
merges what only an independent review can decide (whether the natural pool's site
topology is GT-blind enough to be deployable). Every check fails closed: an unrecorded
scope, a missing certificate, or an unresolved Critical finding keeps the gate closed,
and the downstream verdicts floor at NOT READY rather than trusting the numbers.

The findings live in a tracked review file so the gate is reproducible from the
repository alone: same artifacts plus same findings yield the same JSON, byte for byte.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from ocr_risk.io.hashing import file_sha256

__all__ = ["evaluate_cgv2_integrity"]

_BLOCKING_SEVERITIES = frozenset({"critical", "high"})
_DEFAULT_FINDINGS_PATH = Path("docs/cgv2/integrity_review.json")
_DEFAULT_PRIOR_STUDY = Path("results/generated/generators/generator_study__evaluate.json")

_FINDING_SCHEMA_VERSION = "cgv2-integrity-review-v1"
_GATE_SCHEMA_VERSION = "cgv2-integrity-gate-v1"


@dataclass(frozen=True, slots=True)
class _Finding:
    id: str
    severity: str
    summary: str
    evidence: tuple[str, ...]
    checks: tuple[str, ...]
    disposition: str


def _load_findings(path: Path) -> tuple[list[_Finding], dict[str, bool]]:
    if not path.exists():
        raise FileNotFoundError(f"integrity review file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != _FINDING_SCHEMA_VERSION:
        raise ValueError(
            f"{path} declares schema_version={payload.get('schema_version')!r}; "
            f"expected {_FINDING_SCHEMA_VERSION!r}"
        )
    findings: list[_Finding] = []
    for entry in payload.get("findings", []):
        findings.append(
            _Finding(
                id=str(entry["id"]),
                severity=str(entry["severity"]).strip().lower(),
                summary=str(entry["summary"]),
                evidence=tuple(str(item) for item in entry.get("evidence", [])),
                checks=tuple(str(item) for item in entry.get("checks", ())),
                disposition=str(entry.get("disposition", "open")),
            )
        )
    certifications = {
        str(key): bool(value) for key, value in payload.get("certifications", {}).items()
    }
    return findings, certifications


def _pool_marker_check(source: Path, suffix: str) -> bool:
    ok = True
    for name in (f"cgv2_proposals{suffix}.csv", f"cgv2_region_proposals{suffix}.csv"):
        path = source / name
        if not path.exists():
            ok = False
            continue
        frame = pd.read_csv(path, keep_default_na=False, usecols=["pool"])
        ok = ok and bool(frame["pool"].astype(str).str.lower().eq("natural").all())
    return ok


def _certificate_checks(source: Path, suffix: str) -> tuple[bool, bool, bool]:
    path = source / f"leakage_certificates{suffix}.json"
    if not path.exists():
        return False, False, False
    certificates = json.loads(path.read_text(encoding="utf-8"))
    per_engine = certificates.get("folds", certificates.get("certificates", certificates))
    if not isinstance(per_engine, dict) or not per_engine:
        return False, False, False
    sources_clean = True
    folds_pass = True
    engines_isolated = True
    for certificate in per_engine.values():
        sources_clean = sources_clean and bool(certificate.get("sources_clean"))
        folds_pass = folds_pass and bool(certificate.get("pass"))
        for rung in certificate.get("rungs", {}).values():
            engines_isolated = engines_isolated and bool(rung.get("clean"))
    return sources_clean, folds_pass, engines_isolated


def _selection_scope_check(source: Path, suffix: str) -> bool:
    # Unrecorded scope is unverifiable, not vacuously honored: A2's calibrate-role
    # selection against a fit_only contract is exactly the case this catches.
    for name in (f"leakage_certificates{suffix}.json", f"cgv2_study{suffix}.json"):
        path = source / name
        if not path.exists():
            continue
        value = json.loads(path.read_text(encoding="utf-8")).get("selection_scope")
        if value:
            return str(value) == "fit_only"
    return False


def _fresh_partition_check(source: Path, suffix: str, prior_study: Path) -> bool:
    study_path = source / f"cgv2_study{suffix}.json"
    if not study_path.exists():
        return False
    if not prior_study.exists():
        # Freshness is affirmative: without the pre-CGV2 study record to inspect, no
        # evidence exists that this experiment's evaluate documents were *not* already
        # scored, and the check fails closed rather than assuming it.
        return False
    study = json.loads(study_path.read_text(encoding="utf-8"))
    prior = json.loads(prior_study.read_text(encoding="utf-8"))
    analyzed_before = prior.get("role_scored") == "evaluate" and prior.get(
        "experiment"
    ) == study.get("experiment")
    # Fresh means no pre-CGV2 artifact scored this experiment's evaluate documents.
    return not analyzed_before


def evaluate_cgv2_integrity(
    source: Path,
    role: str = "evaluate",
    findings_path: Path | None = None,
    prior_evaluate_study: Path | None = None,
) -> dict[str, object]:
    """Re-derive the validity gate from artifacts plus the tracked review file."""
    findings_path = findings_path or _DEFAULT_FINDINGS_PATH
    prior_evaluate_study = (
        prior_evaluate_study if prior_evaluate_study is not None else _DEFAULT_PRIOR_STUDY
    )
    suffix = "" if role == "calibrate" else f"__{role}"
    findings, certifications = _load_findings(findings_path)

    sources_clean, folds_pass, engines_isolated = _certificate_checks(source, suffix)
    checks: dict[str, bool | None] = {
        "candidate_pool_marker_propagated": _pool_marker_check(source, suffix),
        "document_role_isolation": sources_clean,
        "fold_scoped_fitted_resources": folds_pass,
        "target_engine_isolation": engines_isolated,
        "fit_only_selection_scope_honored": _selection_scope_check(source, suffix),
        "fresh_confirmatory_evaluate_partition": _fresh_partition_check(
            source, suffix, prior_evaluate_study
        ),
        # A design-level property no artifact can certify: the review must affirm it,
        # and absent certification the gate stays closed.
        "gt_blind_natural_site_construction": certifications.get(
            "gt_blind_natural_site_construction"
        ),
    }
    for finding in findings:
        if finding.disposition.strip().lower() in {"resolved", "withdrawn"}:
            continue
        for check in finding.checks:
            checks[check] = False

    blocking = [
        finding
        for finding in findings
        if finding.severity in _BLOCKING_SEVERITIES
        and finding.disposition.strip().lower() not in {"resolved", "withdrawn"}
    ]
    unmeasurable = sorted(name for name, value in checks.items() if value is None)
    overall_pass = all(value is True for value in checks.values()) and not blocking
    confirmatory = (
        "confirmatory"
        if overall_pass
        and checks["fresh_confirmatory_evaluate_partition"]
        and checks["fit_only_selection_scope_honored"]
        else "exploratory_only"
    )
    return {
        "schema_version": _GATE_SCHEMA_VERSION,
        "role": role,
        "created_at": datetime.now(UTC).date().isoformat(),
        "inputs": {
            "findings_path": str(findings_path),
            "findings_sha256": file_sha256(findings_path),
            "leakage_certificates": f"leakage_certificates{suffix}.json",
            "study_record": f"cgv2_study{suffix}.json",
            "prior_evaluate_study": str(prior_evaluate_study),
        },
        "checks": checks,
        "unmeasurable_checks": unmeasurable,
        "blocking_findings": [
            {
                "id": finding.id,
                "severity": finding.severity,
                "summary": finding.summary,
                "evidence": list(finding.evidence),
                "disposition": finding.disposition,
            }
            for finding in blocking
        ],
        "findings_total": len(findings),
        "confirmatory_status": confirmatory,
        "overall_pass": overall_pass,
    }


def review_documented_on(findings_path: Path | None = None) -> date | None:
    """The review date recorded in the findings file, for report provenance."""
    payload = json.loads((findings_path or _DEFAULT_FINDINGS_PATH).read_text(encoding="utf-8"))
    recorded = payload.get("reviewed_at")
    return date.fromisoformat(str(recorded)) if recorded else None
