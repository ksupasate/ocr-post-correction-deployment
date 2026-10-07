"""Risk-coverage curves, threshold selection, and distribution-free risk control.

Layer 7. Thresholds are selected on the calibration split only; the risk-coverage *curve*
itself lives in :mod:`ocr_risk.metrics` because it is a metric computed from artifacts.
"""

from __future__ import annotations

from ocr_risk.risk.bounds import (
    bentkus_p_value,
    beta_binomial_upper,
    clopper_pearson_upper,
    hoeffding_p_value,
    risk_upper_bound,
    smallest_controlled_risk,
)
from ocr_risk.risk.cluster_bounds import (
    CLUSTER_RATIO_METHODS,
    FINITE_SAMPLE_METHODS,
    ClusterRatioBound,
    betting_mean_upper,
    cluster_ratio_bound,
    cluster_ratio_upper,
    cluster_robust_ratio_upper,
    design_effect,
    empirical_bernstein_mean_upper,
    hoeffding_mean_upper,
)
from ocr_risk.risk.controller import CONTROLLERS, ThresholdDecision, select_threshold
from ocr_risk.risk.policy import SITE_POLICIES, ScoredCandidate, SiteDecision, decide_sites
from ocr_risk.risk.prefix_control import (
    PREFIX_CONTROLS,
    PrefixDecision,
    assert_nested,
    prefix_thresholds,
    select_prefix,
)

__all__ = [
    "CLUSTER_RATIO_METHODS",
    "CONTROLLERS",
    "FINITE_SAMPLE_METHODS",
    "PREFIX_CONTROLS",
    "SITE_POLICIES",
    "ClusterRatioBound",
    "PrefixDecision",
    "ScoredCandidate",
    "SiteDecision",
    "ThresholdDecision",
    "assert_nested",
    "bentkus_p_value",
    "beta_binomial_upper",
    "betting_mean_upper",
    "clopper_pearson_upper",
    "cluster_ratio_bound",
    "cluster_ratio_upper",
    "cluster_robust_ratio_upper",
    "decide_sites",
    "design_effect",
    "empirical_bernstein_mean_upper",
    "hoeffding_mean_upper",
    "hoeffding_p_value",
    "prefix_thresholds",
    "risk_upper_bound",
    "select_prefix",
    "select_threshold",
    "smallest_controlled_risk",
]
