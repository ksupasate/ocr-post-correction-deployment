"""Typed configuration models, YAML include-DAG resolution, and canonical config hashing.

Layer 1.
"""

from __future__ import annotations

from ocr_risk.config.loader import (
    ConfigCycleError,
    ResolvedConfig,
    apply_overrides,
    config_hash,
    deep_merge,
    load_config,
    load_mapping,
    resolve_includes,
)
from ocr_risk.config.models import (
    AlignmentConfig,
    CalibrationConfig,
    CandidateConfig,
    DatasetConfig,
    EngineConfig,
    EvidenceConfig,
    ExperimentConfig,
    GeneratorSpec,
    RiskConfig,
    SiteConfig,
    SplitConfig,
    StatsConfig,
    VerifierSpec,
)

__all__ = [
    "AlignmentConfig",
    "CalibrationConfig",
    "CandidateConfig",
    "ConfigCycleError",
    "DatasetConfig",
    "EngineConfig",
    "EvidenceConfig",
    "ExperimentConfig",
    "GeneratorSpec",
    "ResolvedConfig",
    "RiskConfig",
    "SiteConfig",
    "SplitConfig",
    "StatsConfig",
    "VerifierSpec",
    "apply_overrides",
    "config_hash",
    "deep_merge",
    "load_config",
    "load_mapping",
    "resolve_includes",
]
