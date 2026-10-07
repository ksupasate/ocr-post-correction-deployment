"""The single definition site for every metric in this repository.

Layer 8. This package may **not** import ``verify``, ``candidates``, or ``engines``:
metrics must be computable from saved artifacts alone, months later, without the model
that produced them. Enforced by ``tests/architecture/test_layering.py``.

Never reimplement anything defined here — not in a script, not in a notebook, not "just
for a quick check". Two implementations of CER is how a paper ends up with two numbers
called CER that mean different things.
"""

from __future__ import annotations

from ocr_risk.metrics.calibration import (
    CalibrationReport,
    ReliabilityBin,
    brier_score,
    calibration_report,
    expected_calibration_error,
    murphy_decomposition,
    reliability_bins,
)
from ocr_risk.metrics.discrimination import (
    DiscriminationReport,
    average_precision,
    discrimination_report,
    resolution,
    roc_auc,
    score_separation,
)
from ocr_risk.metrics.edit_accounting import EditAccounting, EditDecision, account
from ocr_risk.metrics.generator import (
    GeneratorQuality,
    SiteProposals,
    beneficial_candidate_rate,
    clean_span_proposal_rate,
    error_repair_opportunity,
    generator_quality,
    harmful_candidate_rate,
    oracle_repair_recall,
    oracle_safe_coverage,
)
from ocr_risk.metrics.precision import (
    PrecisionRequirement,
    attainable_epsilon,
    minimum_accepted_edits,
    precision_table,
)
from ocr_risk.metrics.selective import (
    RiskCoverageCurve,
    RiskCoveragePoint,
    aurc,
    coverage_at_risk,
    risk_coverage_curve,
)
from ocr_risk.metrics.text import (
    ErrorRate,
    cer,
    character_error_counts,
    corpus_cer,
    corpus_wer,
    exact_match_rate,
    levenshtein,
    per_document_cer,
    wer,
    word_error_counts,
)

__all__ = [
    "CalibrationReport",
    "DiscriminationReport",
    "EditAccounting",
    "EditDecision",
    "ErrorRate",
    "GeneratorQuality",
    "PrecisionRequirement",
    "ReliabilityBin",
    "RiskCoverageCurve",
    "RiskCoveragePoint",
    "SiteProposals",
    "account",
    "attainable_epsilon",
    "aurc",
    "average_precision",
    "beneficial_candidate_rate",
    "brier_score",
    "calibration_report",
    "cer",
    "character_error_counts",
    "clean_span_proposal_rate",
    "corpus_cer",
    "corpus_wer",
    "coverage_at_risk",
    "discrimination_report",
    "error_repair_opportunity",
    "exact_match_rate",
    "expected_calibration_error",
    "generator_quality",
    "harmful_candidate_rate",
    "levenshtein",
    "minimum_accepted_edits",
    "murphy_decomposition",
    "oracle_repair_recall",
    "oracle_safe_coverage",
    "per_document_cer",
    "precision_table",
    "reliability_bins",
    "resolution",
    "risk_coverage_curve",
    "roc_auc",
    "score_separation",
    "wer",
    "word_error_counts",
]
