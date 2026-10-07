"""Document partitioning, leave-one-engine-out folds, and programmatic leakage guards.

Layer 7. The highest-stakes code in the repository: a leak here silently invalidates every
number in the paper, and the failure is invisible in the results — they simply look good.

Two axes, always applied together (see ``docs/cross_engine_protocol.md``):

- **engine** — leave one out, evaluate only on it;
- **document** — one global partition, identical across every engine and every fold.

The document axis exists because the benchmark is matched-source: every engine reads the
same page, so holding out an engine does not hold out the page.
"""

from __future__ import annotations

from ocr_risk.splits.audit import AuditFinding, LeakageAudit, audit_records
from ocr_risk.splits.document_partition import (
    DocumentPartition,
    PartitionSpec,
    build_partition,
    duplicate_groups,
)
from ocr_risk.splits.loeo import (
    PROTOCOLS,
    MatchedPair,
    few_shot_documents,
    loeo_folds,
    matched_pairs,
    pairwise_folds,
)
from ocr_risk.splits.plan import (
    CalibrationView,
    EvaluationView,
    FitView,
    LeakageError,
    SplitPlan,
    SplitView,
)

__all__ = [
    "PROTOCOLS",
    "AuditFinding",
    "CalibrationView",
    "DocumentPartition",
    "EvaluationView",
    "FitView",
    "LeakageAudit",
    "LeakageError",
    "MatchedPair",
    "PartitionSpec",
    "SplitPlan",
    "SplitView",
    "audit_records",
    "build_partition",
    "duplicate_groups",
    "few_shot_documents",
    "loeo_folds",
    "matched_pairs",
    "pairwise_folds",
]
