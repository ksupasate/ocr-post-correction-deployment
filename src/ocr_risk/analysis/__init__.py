"""Tables, figures, and reports derived only from immutable artifacts.

Layer 9. This package never recomputes upstream work: it reads what the pipeline wrote, so
every published number traces to bytes that were written once.
"""

from __future__ import annotations

from ocr_risk.analysis.figures import (
    FigureSpec,
    plot_coverage_at_risk,
    plot_reliability,
    plot_risk_coverage,
)
from ocr_risk.analysis.report import ReportResult, build_report
from ocr_risk.analysis.tables import (
    AnalysisInput,
    LeakyRunError,
    calibration_table,
    coverage_at_risk_table,
    harm_decomposition,
    risk_coverage_table,
    transfer_matrix,
)

__all__ = [
    "AnalysisInput",
    "FigureSpec",
    "LeakyRunError",
    "ReportResult",
    "build_report",
    "calibration_table",
    "coverage_at_risk_table",
    "harm_decomposition",
    "plot_coverage_at_risk",
    "plot_reliability",
    "plot_risk_coverage",
    "risk_coverage_table",
    "transfer_matrix",
]
