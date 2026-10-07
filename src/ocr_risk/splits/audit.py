"""Post-hoc leakage audit, re-derived from run records.

Deliberately independent of :mod:`ocr_risk.splits.plan`. The runtime guard checks the code
that is running; this checks what a *finished* run actually recorded. A bug in the guard
would pass its own tests and be caught here, and vice versa — which is the point of having
both.

It reads only ``SplitDescriptor`` fields from run records, so it works months later on
artifacts whose producing code has since changed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ocr_risk.schemas.run_record import RunRecord, SplitDescriptor

__all__ = ["AuditFinding", "LeakageAudit", "audit_records"]

_ALLOWED_SELECTION_SCOPES = frozenset({"fit_only", "dev_only"})
_INTENTIONALLY_CONTAMINATED = frozenset(
    {"diagnostic_doc_overlap", "in_engine_oracle", "matched_in_engine"}
)
# Contaminated on the ENGINE axis by design, and declared by the protocol name itself.
# The document axis is still checked for them, unconditionally.
_ENGINE_AXIS_BY_DESIGN = frozenset({"in_engine_oracle", "matched_in_engine"})


@dataclass(frozen=True, slots=True)
class AuditFinding:
    """One problem, with the vector it corresponds to."""

    severity: str
    vector: str
    fold_id: str
    message: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.vector} ({self.fold_id}): {self.message}"


@dataclass(slots=True)
class LeakageAudit:
    """Result of auditing a set of runs."""

    findings: list[AuditFinding] = field(default_factory=list)
    folds_checked: int = 0
    partition_hashes: set[str] = field(default_factory=set)

    @property
    def passed(self) -> bool:
        return not any(f.severity == "HIGH" for f in self.findings)

    def as_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "folds_checked": self.folds_checked,
            "n_findings": len(self.findings),
            "partition_hashes": sorted(self.partition_hashes),
            "findings": [
                {
                    "severity": f.severity,
                    "vector": f.vector,
                    "fold_id": f.fold_id,
                    "message": f.message,
                }
                for f in self.findings
            ],
        }


def _check_descriptor(split: SplitDescriptor, leaky_run: bool) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    intentional = split.protocol in _INTENTIONALLY_CONTAMINATED

    # L1/L2 — the held-out engine must not appear in any fitted scope, except in the two
    # protocols whose whole purpose is that it does. Both are named reference conditions,
    # never methods; the analysis layer keeps them out of headline method tables.
    if split.protocol not in _ENGINE_AXIS_BY_DESIGN:
        for role, engines in (
            ("fit", split.fit_engines),
            ("calibrate", split.calibrate_engines),
        ):
            shared = set(engines) & set(split.held_out_engines)
            if shared:
                is_few_shot = split.protocol == "loeo_few_shot_recal" and role == "calibrate"
                findings.append(
                    AuditFinding(
                        severity="LOW" if is_few_shot else "HIGH",
                        vector="L1/L2 held-out engine in a fitted scope",
                        fold_id=split.fold_id,
                        message=(
                            f"{role} engines {sorted(shared)} include the held-out engine"
                            + (" (expected under few-shot recalibration)" if is_few_shot else "")
                        ),
                    )
                )

    # The evaluated engine must be exactly the held-out one.
    if set(split.evaluate_engines) != set(split.held_out_engines):
        findings.append(
            AuditFinding(
                severity="HIGH",
                vector="protocol integrity",
                fold_id=split.fold_id,
                message=(
                    f"evaluate engines {sorted(split.evaluate_engines)} != held-out "
                    f"{sorted(split.held_out_engines)}"
                ),
            )
        )

    # L4/L11 — document sets must be disjoint. Compared by digest, because the audit does
    # not have the membership lists, only the hashes the run committed to.
    digests = {
        "fit": split.fit_documents_sha256,
        "calibrate": split.calibrate_documents_sha256,
        "evaluate": split.evaluate_documents_sha256,
    }
    # Partial overlap first: it is the shape a real bug takes, and it is invisible to a
    # digest comparison because two sets sharing some members still hash differently.
    for overlap_role, n_shared in (
        ("fit", split.n_fit_evaluate_shared_documents),
        ("calibrate", split.n_calibrate_evaluate_shared_documents),
    ):
        if n_shared and not (split.leaky or leaky_run):
            findings.append(
                AuditFinding(
                    severity="HIGH",
                    vector="L4 document split",
                    fold_id=split.fold_id,
                    message=(
                        f"{n_shared} document(s) are reachable in both the "
                        f"{overlap_role} and "
                        "evaluate roles. The benchmark is matched-source: holding out an "
                        "engine does not hold out the page."
                    ),
                )
            )

    # The document-axis check is UNCONDITIONAL. `intentional` marks a protocol contaminated
    # on the ENGINE axis by design -- in_engine_oracle is exactly that -- and suppressing
    # the document check for it meant the one arm whose document split most needs verifying
    # was the one arm not verified. Only diagnostic_doc_overlap is deliberately contaminated
    # on the document axis, and it is flagged `leaky` separately.
    document_axis_exempt = split.leaky or leaky_run
    for left, right in (("fit", "evaluate"), ("calibrate", "evaluate"), ("fit", "calibrate")):
        if digests[left] == digests[right] and not document_axis_exempt:
            findings.append(
                AuditFinding(
                    severity="HIGH",
                    vector="L4/L11 document split",
                    fold_id=split.fold_id,
                    message=(
                        f"{left} and {right} document sets have identical digests, so they "
                        "are the same set. The benchmark is matched-source: holding out an "
                        "engine does not hold out the page."
                    ),
                )
            )

    if split.n_evaluate_documents == 0:
        findings.append(
            AuditFinding(
                severity="HIGH",
                vector="protocol integrity",
                fold_id=split.fold_id,
                message="evaluation split is empty; the fold measures nothing",
            )
        )

    # L7 — a config tuned against fold results may not feed a headline table.
    if split.selection_scope not in _ALLOWED_SELECTION_SCOPES:
        findings.append(
            AuditFinding(
                severity="HIGH",
                vector="L7 selection scope",
                fold_id=split.fold_id,
                message=(
                    f"selection_scope={split.selection_scope!r}; model or hyperparameter "
                    "choices must be made within fit or dev scope only"
                ),
            )
        )

    # Deliberate contamination must be declared, so it cannot be mistaken for a clean run.
    if intentional and not (split.leaky or split.protocol in _ENGINE_AXIS_BY_DESIGN):
        findings.append(
            AuditFinding(
                severity="HIGH",
                vector="undeclared contamination",
                fold_id=split.fold_id,
                message=f"protocol {split.protocol!r} is contaminated by design but not flagged",
            )
        )
    if split.leaky and not leaky_run:
        findings.append(
            AuditFinding(
                severity="HIGH",
                vector="undeclared contamination",
                fold_id=split.fold_id,
                message="split is marked leaky but the run record is not",
            )
        )
    return findings


def audit_records(records: Sequence[RunRecord]) -> LeakageAudit:
    """Audit every fold represented in a set of run records."""
    audit = LeakageAudit()
    by_protocol: dict[str, set[str]] = {}

    for record in records:
        # Every fold, not only record.split. A run that executed twenty-eight folds and
        # recorded one would make the partition-stability check below vacuous: a
        # partition redrawn per fold would be reported as a single stable partition.
        descriptors = record.folds or ([record.split] if record.split is not None else [])
        for split in descriptors:
            audit.folds_checked += 1
            audit.partition_hashes.add(split.document_partition_sha256)
            by_protocol.setdefault(split.protocol, set()).add(split.document_partition_sha256)
            audit.findings.extend(_check_descriptor(split, leaky_run=record.leaky))

    # The partition must be identical across every fold of a protocol. A differing hash
    # means it was redrawn per fold, rotating documents between roles — which silently
    # undoes the document split even though each fold looks internally consistent.
    for protocol, hashes in sorted(by_protocol.items()):
        if len(hashes) > 1:
            audit.findings.append(
                AuditFinding(
                    severity="HIGH",
                    vector="L4 partition stability",
                    fold_id=f"protocol:{protocol}",
                    message=(
                        f"{len(hashes)} different document-partition hashes across folds of "
                        f"{protocol!r}. The partition must be drawn once and reused, or "
                        "documents rotate between roles across folds."
                    ),
                )
            )

    # Cross-EXPERIMENT partition agreement. by_protocol only compares folds within one
    # protocol, and the CLI audits one experiment at a time, so nothing asserted that the
    # two arms of the H1 contrast drew on the same partition -- the precise confound
    # docs/cross_engine_protocol.md says must not happen. Any two protocols sharing a
    # partition_id must agree.
    if len(audit.partition_hashes) > 1:
        audit.findings.append(
            AuditFinding(
                severity="HIGH",
                vector="L4 partition agreement across protocols",
                fold_id="all",
                message=(
                    f"{len(audit.partition_hashes)} distinct document partitions across the "
                    "audited records. Two protocols compared against each other must draw "
                    "on one partition, or the contrast is confounded by which documents "
                    "each arm happened to evaluate."
                ),
            )
        )

    return audit
