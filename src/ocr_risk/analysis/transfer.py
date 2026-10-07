"""The H1 measurement: what unseen-engine transfer costs.

H1 asks whether a correction-acceptance model calibrated on one set of OCR engines
degrades on a completely unseen one. That is a **paired contrast between two protocols**
on the same target documents, not a property of any single run:

``in_engine_oracle``
    the target engine is present in the fitting and calibration environment. An
    engine-aware reference condition, not a competing method and not an unrestricted
    oracle — the document partition still holds, so no test document influences fitting.

``loeo_zero_shot``
    the target engine contributes nothing to model fitting, calibrator fitting,
    confidence normalization, threshold selection, or hyperparameter selection.

The gap between them, per target engine, is the cost of engine shift.

Two measurement choices are load-bearing.

**Four metrics, and the hierarchy among them is narrower than it looks.** Brier
decomposes as ``calibration + refinement`` (DeGroot-Fienberg; ``refinement`` is the
within-bin outcome variance, so it already absorbs the uncertainty term). All four are
reported per engine.

The Murphy calibration term is *primary* only because H1 is worded about **calibration
specifically**. That is the whole of the justification, and it is worth stating what it is
NOT: an earlier version of this module claimed Brier was demoted because "engines differ in
base harm rate by construction, so a raw-Brier contrast partly measures base rates". That
argument is correct for a comparison of Brier LEVELS across engines -- which the gate used
to make -- and **inapplicable here**. This contrast is within-engine, on matched
candidates, so the two arms have bit-identical outcome vectors and the uncertainty term
cancels exactly. Preferring the calibration term does not remove a confound; it deliberately
narrows the question, and the refinement channel it excludes is where an effect may well be.

So ``brier`` and ``refinement_error`` are not decoration. A verdict read off the
calibration term alone answers "did the probabilities become less honest?" and is silent on
"did the model become less able to tell safe from harmful?" -- and the second is what caps
achievable coverage at a fixed risk.

**Per engine, never averaged.** Four OCR engines are four fixed, heterogeneous
experimental environments, not four draws from a population of engines. Document-level
resampling quantifies uncertainty *within* an engine's fold; nothing here licenses
inference about OCR engines in general, and no statistic is computed across folds.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from ocr_risk.analysis.tables import AnalysisInput, _harmful
from ocr_risk.metrics import calibration_report
from ocr_risk.schemas.enums import HarmPolicy
from ocr_risk.stats import (
    holm_bonferroni,
    minimum_detectable_effect,
    paired_cluster_bootstrap_multi,
)

__all__ = ["h1_transfer_table", "held_out_engine_of"]

# Higher is worse for these, so degradation under transfer is a positive delta.
_METRICS = (
    "brier",
    "calibration_error",
    # The other half of the decomposition. Brier = calibration + refinement, and the two
    # failure modes call for different responses: a miscalibrated model can be
    # recalibrated, an unrefined one cannot. Under engine shift they move independently.
    "refinement_error",
    # Both binning schemes, as the pre-registration requires. Equal-width ECE is
    # binning-biased in a way equal-mass is not, and reporting one alone is the choice
    # the rule exists to prevent.
    "ece_equal_mass",
    "ece_equal_width",
)

# Higher is worse for all of these, so degradation under transfer is a positive delta.
_ERROR_METRICS = frozenset(_METRICS)


def held_out_engine_of(fold_id: str) -> str:
    """``loeo_zero_shot:tesseract`` -> ``tesseract``."""
    return fold_id.rsplit(":", 1)[-1]


def _realized_bins(scores: np.ndarray, safe: np.ndarray, n_bins: int) -> int:
    """How many bins the equal-mass scheme actually produced.

    Not the requested count. Isotonic calibration collapses scores onto a few dozen
    distinct values, so ``np.unique`` on the equal-mass edges yields fewer bins than asked
    for — and a DIFFERENT number in each arm. The binning estimator of calibration error is
    upward-biased in the bin count, so an asymmetry here puts a small directional bias on
    the paired delta, measured at up to 38% of one engine's point estimate and pointing
    *against* detecting degradation on three of four engines. Below the interval widths, so
    it does not overturn anything, but it is unmodelled and it is not symmetric — which
    matters more for a null than it would for a positive.
    """
    if scores.size == 0:
        return 0
    return len(calibration_report(scores, safe, n_bins=n_bins).reliability)


def _metric_values(scores: np.ndarray, safe: np.ndarray, n_bins: int) -> dict[str, float]:
    """Calibration quantities for one arm, on one candidate set."""
    if scores.size == 0:
        return dict.fromkeys(_METRICS, float("nan"))
    report = calibration_report(scores, safe, n_bins=n_bins)
    return {
        "brier": report.brier,
        "calibration_error": report.calibration_term,
        # The other half of the decomposition. Brier = calibration + refinement, and the
        # two failure modes call for different responses: a miscalibrated model can be
        # recalibrated, an unrefined one cannot. Under engine shift they can move
        # independently, so reporting only the calibration half would answer a narrower
        # question than H1 asks.
        "refinement_error": report.refinement_term,
        "ece_equal_mass": report.ece_equal_mass,
        "ece_equal_width": report.ece_equal_width,
    }


def _arm_frame(data: AnalysisInput, policy: HarmPolicy) -> pd.DataFrame:
    """Predictions joined to outcomes, natural pool only, with harm resolved."""
    frame = data.predictions.merge(
        data.labels[["candidate_id", "outcome_if_accepted"]], on="candidate_id", how="inner"
    ).merge(
        data.candidates[["candidate_id", "is_synthetic_hard_negative"]],
        on="candidate_id",
        how="left",
    )
    frame = data.natural(frame)
    if frame.empty:
        return frame
    data.guard_headline(set(frame["fold_id"].unique()))
    frame = frame.assign(
        held_out_engine=frame["fold_id"].map(held_out_engine_of),
        safe=(~_harmful(frame, policy)).astype(np.float64),
        score=frame["calibrated_score"].fillna(frame["raw_score"]).astype(float),
    )
    return frame


def h1_transfer_table(
    zero_shot: AnalysisInput,
    oracle: AnalysisInput,
    policy: HarmPolicy,
    *,
    n_bins: int = 15,
    n_bootstrap: int = 10_000,
    ci_level: float = 0.95,
    seed: int = 7,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Per (target engine, verifier): the zero-shot minus oracle gap, with a paired CI.

    Candidates are intersected between the two conditions first, so the contrast is on
    identical rows and any difference is attributable to what the two protocols were
    allowed to fit on — not to a differing evaluation set.
    """
    left, right = _arm_frame(zero_shot, policy), _arm_frame(oracle, policy)
    if left.empty or right.empty:
        return pd.DataFrame()

    keys = ["held_out_engine", "verifier_id"]
    rows: list[dict[str, object]] = []

    for (engine, verifier), loeo_group in left.groupby(keys, sort=True):
        oracle_group = right[
            (right["held_out_engine"] == engine) & (right["verifier_id"] == verifier)
        ]
        if oracle_group.empty:
            continue

        # Matched candidates only: a paired contrast on different rows is not paired.
        shared = sorted(set(loeo_group["candidate_id"]) & set(oracle_group["candidate_id"]))
        if not shared:
            continue
        a = loeo_group.set_index("candidate_id").loc[shared]
        b = oracle_group.set_index("candidate_id").loc[shared]

        loeo_values = _metric_values(a["score"].to_numpy(), a["safe"].to_numpy(), n_bins)
        oracle_values = _metric_values(b["score"].to_numpy(), b["safe"].to_numpy(), n_bins)

        items_a = list(zip(a["document_id"], a["score"], a["safe"], strict=True))
        items_b = list(zip(b["document_id"], b["score"], b["safe"], strict=True))

        def all_metrics(items: Sequence[tuple[str, float, float]]) -> dict[str, float]:
            if not items:
                return dict.fromkeys(_METRICS, float("nan"))
            scores = np.array([s for _, s, _ in items], dtype=float)
            safe = np.array([v for _, _, v in items], dtype=float)
            return _metric_values(scores, safe, n_bins)

        # One bootstrap pass for all five metrics. They all fall out of a single
        # calibration_report, so bootstrapping them separately recomputed that report five
        # times per resample and threw away four fifths of it each time.
        paired_by_metric = paired_cluster_bootstrap_multi(
            items_a,
            items_b,
            lambda item: str(item[0]),
            all_metrics,
            n_resamples=n_bootstrap,
            ci_level=ci_level,
            seed=seed,
        )

        for metric in _METRICS:
            paired = paired_by_metric[metric]
            # A NaN interval means the resampling distribution was degenerate, which is
            # not evidence of anything. Without this guard a constant-score arm registers
            # as maximally significant on zero sampling variability.
            if not (np.isfinite(paired.lower) and np.isfinite(paired.upper)):
                degraded = False
            elif metric in _ERROR_METRICS:
                degraded = bool(paired.lower > 0.0)
            else:
                degraded = bool(paired.upper < 0.0)
            rows.append(
                {
                    "held_out_engine": engine,
                    "verifier_id": verifier,
                    "metric": metric,
                    "loeo_zero_shot": loeo_values[metric],
                    "in_engine_oracle": oracle_values[metric],
                    # zero-shot minus oracle: positive means transfer made the error worse.
                    "delta": paired.estimate,
                    "delta_ci_lower": paired.lower,
                    "delta_ci_upper": paired.upper,
                    "n_candidates": len(shared),
                    "n_documents": paired.n_clusters,
                    "p_value": paired.p_value_two_sided,
                    # Reported for EVERY row, significant or not. A null with a minimum
                    # detectable effect larger than the quantity being compared is
                    # uninformative, and the reader cannot tell which kind of null this
                    # is without the number.
                    "standard_error": paired.standard_error,
                    "minimum_detectable_effect": minimum_detectable_effect(
                        paired.standard_error, alpha=alpha, n_tests=4
                    ),
                    "degraded": degraded,
                    "harm_policy": policy.value,
                    # The measurement choices travel with the value. A calibration number
                    # without its bin count and scheme is not a reportable result, and
                    # recovering them from a config hash is not the same as recording them.
                    "n_bins": n_bins,
                    "binning": "equal_mass",
                    "n_bootstrap": n_bootstrap,
                    "ci_level": ci_level,
                    "degenerate_interval": bool(paired.degenerate_interval),
                    "realized_bins_zero_shot": _realized_bins(
                        a["score"].to_numpy(), a["safe"].to_numpy(), n_bins
                    ),
                    "realized_bins_oracle": _realized_bins(
                        b["score"].to_numpy(), b["safe"].to_numpy(), n_bins
                    ),
                    # Documents evaluated in this fold. The four folds do not see the same
                    # set -- an engine's own recognition failures remove its own pages -- so
                    # comparing LEVELS across engines compares measurements on slightly
                    # different document subsets. The paired within-engine contrast is
                    # unaffected; the cross-engine count is not.
                    "n_documents_evaluated": int(a["document_id"].nunique()),
                }
            )

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    # Multiplicity is controlled across the four TARGET ENGINES, within each
    # (metric, verifier) -- the family declared in docs/preregistration.md before the run.
    #
    # Not across engines AND verifiers together. The verifiers are the evidence-ablation
    # ladder: V3 against V6 is a different question from whether transfer costs anything,
    # and folding all seven into one family of 28 would apply a correction for comparisons
    # H1 is not making. It is also strictly more conservative than what was promised, and
    # choosing between the two after seeing results is exactly what declaring the family
    # in advance prevents.
    table["holm_adjusted_p"] = float("nan")
    table["degraded_after_holm"] = False
    for _key, family in table.groupby(["metric", "verifier_id"], sort=True):
        p_values = {
            str(row["held_out_engine"]): (
                float(row["p_value"]) if np.isfinite(float(row["p_value"])) else 1.0
            )
            for _, row in family.iterrows()
        }
        holm = {t.label: t for t in holm_bonferroni(p_values, alpha=alpha)}
        for index, row in family.iterrows():
            test = holm[str(row["held_out_engine"])]
            table.loc[index, "holm_adjusted_p"] = test.adjusted_p_value
            # Direction AND significance: a surviving p only says the difference is not
            # zero; which way it points is what makes it degradation.
            table.loc[index, "degraded_after_holm"] = bool(row["degraded"]) and test.significant

    return table
