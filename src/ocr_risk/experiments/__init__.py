"""Pipeline orchestration, the leave-one-engine-out runner, and the pilot gate.

Layer 9.
"""

from __future__ import annotations

from ocr_risk.experiments.gate import GateReport, HypothesisVerdict, evaluate_gate
from ocr_risk.experiments.loeo_runner import FoldResult, MethodSpec, methods_from_config, run_fold
from ocr_risk.experiments.pipeline import PipelineState, run_experiment

__all__ = [
    "FoldResult",
    "GateReport",
    "HypothesisVerdict",
    "MethodSpec",
    "PipelineState",
    "evaluate_gate",
    "methods_from_config",
    "run_experiment",
    "run_fold",
]
