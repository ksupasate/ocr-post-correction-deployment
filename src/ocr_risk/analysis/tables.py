"""Evaluation tables, computed from immutable artifacts only.

This layer never recomputes upstream work. It reads the decide-stage tables and turns them
into the numbers a paper would report, which is what makes every figure traceable to bytes
that were written once.

Two structural rules are enforced here rather than left to discipline:

- a run whose split is flagged ``leaky`` cannot enter a headline table;
- the adversarial challenge set is reported **separately** from the natural candidate
  pool, because pooling them measures a different question.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ocr_risk.edits.outcome import is_beneficial
from ocr_risk.metrics import (
    EditDecision,
    account,
    aurc,
    calibration_report,
    coverage_at_risk,
    risk_coverage_curve,
)
from ocr_risk.schemas.enums import (
    CoverageUnit,
    DecisionAction,
    HarmPolicy,
    OutcomeIfAccepted,
    OutcomeIfRejected,
    harmful_outcomes,
)
from ocr_risk.stats import cluster_bootstrap_indices

__all__ = [
    "LeakyRunError",
    "candidate_population",
    "confidence_reliability",
    "coverage_at_risk_table",
    "deployed_operating_table",
    "engine_diagnostics",
    "harm_decomposition",
    "noise_tertiles",
    "risk_coverage_table",
    "stratified_risk",
    "transfer_matrix",
]


class LeakyRunError(RuntimeError):
    """Raised when a deliberately contaminated run is used in a headline table."""


@dataclass(slots=True)
class AnalysisInput:
    """The joined artifact tables the analysis layer works from."""

    predictions: pd.DataFrame
    decisions: pd.DataFrame
    labels: pd.DataFrame
    candidates: pd.DataFrame
    sites: pd.DataFrame = field(default_factory=pd.DataFrame)
    """Site-level truth. ``d_before`` and whether a site was clean are properties of the
    SITE, not of whichever candidate happened to be accepted there — reading them off the
    accepted candidate makes every preserved site look clean."""
    leaky_folds: frozenset[str] = field(default_factory=frozenset)

    def natural(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Drop adversarial hard negatives from a candidate-level frame."""
        if "is_synthetic_hard_negative" not in frame.columns:
            return frame
        return frame.loc[~frame["is_synthetic_hard_negative"].astype(bool)]

    def challenge(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Keep only the adversarial hard negatives."""
        if "is_synthetic_hard_negative" not in frame.columns:
            return frame.iloc[0:0]
        return frame.loc[frame["is_synthetic_hard_negative"].astype(bool)]

    def guard_headline(self, fold_ids: set[str]) -> None:
        contaminated = fold_ids & self.leaky_folds
        if contaminated:
            msg = (
                f"folds {sorted(contaminated)} are contaminated by design and may not "
                "appear in a headline table. They exist to quantify the inflation that "
                "contamination causes, and are reported as diagnostics."
            )
            raise LeakyRunError(msg)


def _as_int(value: object) -> int:
    """Coerce a pandas cell to int. Missing is an error, not a zero.

    ``d_before == 0`` means "this site was already correct", so coercing a failed join to
    zero silently reclassifies every unmatched site as clean — which inflates the clean
    denominator, deflates the overcorrection rate, and drives ``n_sites_with_error`` to
    zero on a corpus that plainly has errors.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        msg = "missing integer field; a failed join must not be read as zero"
        raise ValueError(msg)
    return int(float(value))  # type: ignore[arg-type]


def _as_int_or(value: object, default: int) -> int:
    """``_as_int`` with an explicit, deliberate default for genuinely optional fields."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    return int(float(value))  # type: ignore[arg-type]


def _harmful(frame: pd.DataFrame, policy: HarmPolicy) -> np.ndarray:
    values = {o.value for o in harmful_outcomes(policy)}
    return frame["outcome_if_accepted"].isin(values).to_numpy(dtype=bool)


def risk_coverage_table(
    data: AnalysisInput, policy: HarmPolicy, *, pool: str = "natural"
) -> pd.DataFrame:
    """Risk-coverage summary per (fold, method), on one candidate pool."""
    frame = data.predictions.merge(
        data.labels[["candidate_id", "outcome_if_accepted", "d_before", "delta"]],
        on="candidate_id",
        how="inner",
    ).merge(
        data.candidates[["candidate_id", "is_synthetic_hard_negative"]],
        on="candidate_id",
        how="left",
    )
    frame = data.natural(frame) if pool == "natural" else data.challenge(frame)
    if frame.empty:
        return pd.DataFrame()

    data.guard_headline(set(frame["fold_id"].unique()) if pool == "natural" else set())

    rows = []
    for (fold_id, verifier_id), group in frame.groupby(["fold_id", "verifier_id"], sort=True):
        harmful = _harmful(group, policy)
        scores = group["calibrated_score"].fillna(group["raw_score"]).to_numpy(dtype=float)
        curve = risk_coverage_curve(scores, harmful)
        rows.append(
            {
                "fold_id": fold_id,
                "verifier_id": verifier_id,
                "pool": pool,
                "n_candidates": len(group),
                "base_harm_rate": float(harmful.mean()),
                "aurc": aurc(curve),
                "harm_policy": policy.value,
            }
        )
    return pd.DataFrame(rows)


def coverage_at_risk_table(
    data: AnalysisInput,
    policy: HarmPolicy,
    epsilons: tuple[float, ...],
    *,
    pool: str = "natural",
    n_bootstrap: int = 10_000,
    ci_level: float = 0.95,
    seed: int = 7,
) -> pd.DataFrame:
    """The headline table: coverage attainable at each harmful-edit tolerance.

    Confidence intervals resample **documents**, not candidates: candidates within a page
    are correlated, and candidate-level resampling would report intervals far too narrow.
    """
    frame = data.predictions.merge(
        data.labels[["candidate_id", "outcome_if_accepted"]], on="candidate_id", how="inner"
    ).merge(
        data.candidates[["candidate_id", "is_synthetic_hard_negative"]],
        on="candidate_id",
        how="left",
    )
    frame = data.natural(frame) if pool == "natural" else data.challenge(frame)
    if frame.empty:
        return pd.DataFrame()

    # This is the primary result, so the guard belongs here above all: a deliberately
    # contaminated fold reaching Coverage@Risk would be the one number most likely to be
    # quoted.
    if pool == "natural":
        data.guard_headline(set(frame["fold_id"].unique()))

    rows = []
    for (fold_id, verifier_id), group in frame.groupby(["fold_id", "verifier_id"], sort=True):
        harmful = _harmful(group, policy)
        scores = group["calibrated_score"].fillna(group["raw_score"]).to_numpy(dtype=float)
        documents = group["document_id"].to_numpy()
        curve = risk_coverage_curve(scores, harmful)

        for epsilon in epsilons:
            point = coverage_at_risk(curve, epsilon)
            coverage = point.coverage if point else 0.0

            def statistic(index: NDArray[np.intp]) -> float:
                if index.size == 0:
                    return 0.0
                resampled_point = coverage_at_risk(
                    risk_coverage_curve(scores[index], harmful[index]),  # noqa: B023
                    epsilon,  # noqa: B023
                )
                return resampled_point.coverage if resampled_point else 0.0

            interval = cluster_bootstrap_indices(
                [str(d) for d in documents],
                statistic,
                n_resamples=n_bootstrap,
                ci_level=ci_level,
                seed=seed,
            )
            rows.append(
                {
                    "fold_id": fold_id,
                    "verifier_id": verifier_id,
                    "pool": pool,
                    "epsilon": epsilon,
                    "coverage": coverage,
                    "coverage_ci_lower": interval.lower,
                    "coverage_ci_upper": interval.upper,
                    "n_documents": interval.n_clusters,
                    # NaN, not 0.0. Selective risk at zero coverage is undefined, and 57
                    # of 84 rows were reading "risk 0.0" for operating points that do not
                    # exist -- the best possible value for a method that reached nothing.
                    "risk_at_point": point.risk if point else float("nan"),
                    "feasible": point is not None,
                    # Flagged in the artifact, not only in the docstring and the prose. This
                    # threshold is re-derived on the evaluation sample subject to empirical
                    # risk <= epsilon, so "risk within tolerance" is true by construction
                    # here. A CSV consumer cannot otherwise tell this column from the
                    # deployed one, where the same phrase is a measurement that can fail.
                    "in_sample_upper_bound": True,
                    "harm_policy": policy.value,
                }
            )
    return pd.DataFrame(rows)


def harm_decomposition(data: AnalysisInput, policy: HarmPolicy) -> pd.DataFrame:
    """What the accepted edits actually did, per (fold, method, epsilon).

    Two joins, not one, and the distinction is the whole point:

    - ``d_before`` and ``outcome_if_rejected`` describe the **site**, and every decided
      site has them whether or not an edit was accepted there;
    - ``outcome_if_accepted`` and ``delta`` describe the **accepted candidate**, and exist
      only where something was accepted.

    Reading the site-level fields off the accepted candidate leaves them null at every
    preserved site, which reads as ``d_before == 0`` — "already correct". That silently
    reclassifies most of the corpus as clean and drives the error denominator to zero.
    """
    decisions = data.decisions
    if decisions.empty:
        return pd.DataFrame()

    # The headline guard and the natural-pool filter belong here too: this table feeds
    # the pilot gate, so a contaminated fold or an adversarial candidate pool reaching it
    # would move a published verdict.
    data.guard_headline(set(decisions["fold_id"].unique()))

    if not data.candidates.empty and "is_synthetic_hard_negative" in data.candidates.columns:
        adversarial = set(
            data.candidates.loc[
                data.candidates["is_synthetic_hard_negative"].astype(bool), "candidate_id"
            ]
        )
        if adversarial:
            accepted = decisions["accepted_candidate_id"]
            decisions = decisions.loc[~accepted.isin(adversarial)]

    site_columns = ["site_id", "d_before"]
    if data.sites.empty or not set(site_columns) <= set(data.sites.columns):
        msg = (
            "harm_decomposition needs the site table: d_before is a property of the site, "
            "and deriving it from the accepted candidate makes every preserved site look "
            "clean. Pass AnalysisInput(sites=...)."
        )
        raise ValueError(msg)

    decisions = decisions.merge(
        data.sites[site_columns].drop_duplicates("site_id"), on="site_id", how="left"
    ).merge(
        data.labels[["candidate_id", "outcome_if_accepted", "delta"]],
        left_on="accepted_candidate_id",
        right_on="candidate_id",
        how="left",
    )

    unmatched = int(decisions["d_before"].isna().sum())
    if unmatched:
        msg = f"{unmatched} decision(s) reference a site absent from the site table"
        raise ValueError(msg)

    # Beneficial candidates OFFERED per site, natural pool only. The denominator of
    # beneficial-edit coverage, which is what separates "the verifier rejected good
    # candidates" from "the generator produced none" -- two failures the pilot could not
    # tell apart because it reported neither.
    offered = data.natural(
        data.candidates[["candidate_id", "site_id"]].merge(
            data.labels[["candidate_id", "outcome_if_accepted"]], on="candidate_id"
        )
        if not data.candidates.empty
        else data.labels[["candidate_id", "site_id", "outcome_if_accepted"]]
    )
    beneficial_values = {o.value for o in OutcomeIfAccepted if is_beneficial(o, policy)}
    beneficial_per_site = (
        offered[offered["outcome_if_accepted"].isin(beneficial_values)]
        .groupby("site_id")
        .size()
        .to_dict()
        if not offered.empty
        else {}
    )

    rows = []
    for (fold_id, method_id, epsilon), group in decisions.groupby(
        ["fold_id", "method_id", "epsilon"], sort=True
    ):
        # Coerce once at the pandas boundary rather than casting at every field: the
        # frame's dtypes are Any as far as the type checker is concerned.
        edit_decisions = []
        for record in group.to_dict("records"):
            accepted = record["action"] == DecisionAction.CORRECT.value
            outcome = record.get("outcome_if_accepted")
            if accepted and (outcome is None or pd.isna(outcome)):
                # An accepted site whose candidate did not join: skip rather than invent
                # an outcome for it.
                continue
            d_before = _as_int(record.get("d_before"))
            edit_decisions.append(
                EditDecision(
                    site_id=str(record["site_id"]),
                    document_id=str(record["document_id"]),
                    accepted=accepted,
                    d_before=d_before,
                    outcome_if_accepted=OutcomeIfAccepted(str(outcome))
                    if outcome is not None and not pd.isna(outcome)
                    else None,
                    # Derived from the site, so a preserved clean site is PRESERVATION and
                    # a preserved broken site is MISSED_ERROR — the two are not the same
                    # outcome and collapsing them erases the cost of abstaining.
                    outcome_if_rejected=(
                        OutcomeIfRejected.PRESERVATION
                        if d_before == 0
                        else OutcomeIfRejected.MISSED_ERROR
                    ),
                    delta=_as_int_or(record.get("delta"), 0),
                    n_candidates=_as_int_or(record.get("n_candidates_considered"), 0),
                    n_beneficial_available=int(beneficial_per_site.get(str(record["site_id"]), 0)),
                )
            )
        summary = account(edit_decisions, policy, CoverageUnit.SITE)
        rows.append(
            {"fold_id": fold_id, "method_id": method_id, "epsilon": epsilon, **summary.as_dict()}
        )
    return pd.DataFrame(rows)


def transfer_matrix(data: AnalysisInput, policy: HarmPolicy, epsilon: float) -> pd.DataFrame:
    """Coverage per (held-out engine, method) — the cross-engine view.

    Per-fold rows, never a mean. With four engines a mean would summarize four points and
    hide precisely the variation the study is about.
    """
    table = coverage_at_risk_table(data, policy, (epsilon,))
    if table.empty:
        return table
    table = table.copy()
    table["held_out_engine"] = table["fold_id"].str.rsplit(":", n=1).str[-1]
    return table.pivot_table(
        index="held_out_engine", columns="verifier_id", values="coverage", aggfunc="first"
    ).reset_index()


def calibration_table(data: AnalysisInput, policy: HarmPolicy, n_bins: int = 10) -> pd.DataFrame:
    """Calibration quality per (fold, method) on the evaluation split.

    The direct H1 measurement: a calibrator fitted on the training engines, assessed on
    the held-out one.
    """
    frame = data.predictions.merge(
        data.labels[["candidate_id", "outcome_if_accepted"]], on="candidate_id", how="inner"
    ).merge(
        data.candidates[["candidate_id", "is_synthetic_hard_negative"]],
        on="candidate_id",
        how="left",
    )
    frame = data.natural(frame)
    if not frame.empty:
        data.guard_headline(set(frame["fold_id"].unique()))

    rows = []
    for (fold_id, verifier_id), group in frame.groupby(["fold_id", "verifier_id"], sort=True):
        scores = group["calibrated_score"].fillna(group["raw_score"]).to_numpy(dtype=float)
        safe = (~_harmful(group, policy)).astype(np.float64)
        report = calibration_report(scores, safe, n_bins=n_bins)
        rows.append(
            {
                "fold_id": fold_id,
                "verifier_id": verifier_id,
                "n": report.n,
                "brier": report.brier,
                "ece_equal_mass": report.ece_equal_mass,
                "ece_equal_width": report.ece_equal_width,
                "calibration_term": report.calibration_term,
                "refinement_term": report.refinement_term,
                "base_rate": report.base_rate,
            }
        )
    return pd.DataFrame(rows)


def deployed_operating_table(
    data: AnalysisInput,
    policy: HarmPolicy,
    thresholds: pd.DataFrame,
    *,
    n_bootstrap: int = 10_000,
    ci_level: float = 0.95,
    seed: int = 7,
) -> pd.DataFrame:
    """What the frozen system actually did on the held-out engine.

    This is the deployable measurement, and it differs from
    :func:`coverage_at_risk_table` in the way that matters. That function rebuilds the
    risk-coverage curve on the evaluation predictions and reads off the best coverage
    whose *empirical* risk on that same sample is within tolerance — a constrained
    in-sample optimum, so ``risk <= epsilon`` holds there by construction and measures
    nothing. Here the threshold is the one the controller certified on the calibration
    split, applied unchanged, and the realized risk is free to exceed the tolerance.

    It being free to exceed is the point. The distribution-free guarantee assumes
    exchangeability between calibration and test, and cross-engine shift violates that by
    construction — so the gap between ``realized_risk`` and ``risk_upper_bound`` is a
    measurement of how far the guarantee degrades, which is a result rather than a bug.
    """
    if thresholds.empty:
        return pd.DataFrame()

    frame = data.predictions.merge(
        data.labels[["candidate_id", "outcome_if_accepted"]], on="candidate_id", how="inner"
    ).merge(
        data.candidates[["candidate_id", "is_synthetic_hard_negative"]],
        on="candidate_id",
        how="left",
    )
    frame = data.natural(frame)
    if frame.empty:
        return pd.DataFrame()
    data.guard_headline(set(frame["fold_id"].unique()))

    # method_id is "<verifier>|<evidence>|<calibrator>"; the verifier is what identifies
    # the arm in the prediction table.
    certified = thresholds.assign(verifier_id=thresholds["method_id"].str.split("|").str[0])

    rows = []
    for (fold_id, verifier_id), group in frame.groupby(["fold_id", "verifier_id"], sort=True):
        harmful = _harmful(group, policy)
        scores = group["calibrated_score"].fillna(group["raw_score"]).to_numpy(dtype=float)
        documents = group["document_id"].to_numpy()

        arm = certified[
            (certified["fold_id"] == fold_id) & (certified["verifier_id"] == verifier_id)
        ]
        for record in arm.to_dict("records"):
            tau = float(record["tau"])
            epsilon = float(record["epsilon"])
            accepted = scores >= tau if np.isfinite(tau) else np.zeros(len(scores), dtype=bool)
            n_accepted = int(accepted.sum())
            n_harmful = int(harmful[accepted].sum())
            realized = float(n_harmful / n_accepted) if n_accepted else float("nan")

            document_labels = [str(d) for d in documents]

            def realized_risk(index: NDArray[np.intp]) -> float:
                taken = harmful[index][accepted[index]]  # noqa: B023
                return float(taken.mean()) if taken.size else float("nan")

            def realized_coverage(index: NDArray[np.intp]) -> float:
                return float(accepted[index].mean()) if index.size else 0.0  # noqa: B023

            risk_ci = cluster_bootstrap_indices(
                document_labels,
                realized_risk,
                n_resamples=n_bootstrap,
                ci_level=ci_level,
                seed=seed,
            )
            coverage_ci = cluster_bootstrap_indices(
                document_labels,
                realized_coverage,
                n_resamples=n_bootstrap,
                ci_level=ci_level,
                seed=seed,
            )
            bound = float(record.get("risk_upper_bound", float("nan")))
            rows.append(
                {
                    "fold_id": fold_id,
                    "verifier_id": verifier_id,
                    "epsilon": epsilon,
                    "tau": tau,
                    "n_candidates": len(group),
                    "n_accepted": n_accepted,
                    "n_harmful_accepted": n_harmful,
                    "realized_risk": realized,
                    "realized_risk_ci_lower": risk_ci.lower,
                    "realized_risk_ci_upper": risk_ci.upper,
                    "coverage": float(accepted.mean()) if len(accepted) else 0.0,
                    "coverage_ci_lower": coverage_ci.lower,
                    "coverage_ci_upper": coverage_ci.upper,
                    # The stable denominator: selective risk vanishes as tau -> 1, exactly
                    # where a method looks safest, so the joint rate is reported beside it.
                    # Tri-valued, not boolean. A system that accepted nothing has not
                    # respected the bound -- there is nothing to respect, and recording
                    # True scores total abstention with the best possible value. On this
                    # pilot every one of 168 rows had n_accepted == 0, so a boolean column
                    # would have read "bound respected" across the board while the
                    # exchangeability measurement this table exists for was unmeasurable.
                    "joint_harm_rate": (
                        float(n_harmful / len(group)) if n_accepted else float("nan")
                    ),
                    "risk_upper_bound_on_calibration": bound,
                    "bound_respected": bool(realized <= bound) if n_accepted else None,
                    "tolerance_respected": bool(realized <= epsilon) if n_accepted else None,
                    "n_documents": risk_ci.n_clusters,
                    "in_sample_upper_bound": False,
                    "degenerate_interval": bool(
                        risk_ci.degenerate_interval or coverage_ci.degenerate_interval
                    ),
                    "harm_policy": policy.value,
                }
            )
    return pd.DataFrame(rows)


def engine_diagnostics(store: object, experiment: str) -> pd.DataFrame:
    """Per-engine properties that could be mistaken for a transfer effect.

    Reported beside the risk numbers, not in a separate diagnostics file. Three
    quantities differ across engines for reasons that have nothing to do with recognition
    quality, and each of them moves the site population that Coverage and Risk are
    denominated in:

    ``ambiguity_rate``
        the share of alignment components the aligner declined to resolve. It is not
        uniform across engines, and a fold whose held-out engine aligns worse will look
        different for that reason alone.
    ``n_sites`` / ``n_clean_sites``
        clean sites are the only place overcorrection is observable, so an engine
        contributing few of them looks safer than it is.
    ``granularity_split_rate``
        the share of spans cut out of a coarser engine detection. PaddleOCR detects
        lines; normalization equalizes this, and the rate records how much work the
        normalization did for each engine.
    """
    from ocr_risk.io.artifacts import ArtifactStore
    from ocr_risk.schemas.enums import StageName

    assert isinstance(store, ArtifactStore)
    align_run = store.latest(StageName.ALIGN, experiment)
    sites_run = store.latest(StageName.SITES, experiment)
    spans_run = store.latest(StageName.CANONICALIZE, experiment)
    if align_run is None or sites_run is None or spans_run is None:
        return pd.DataFrame()

    alignments = store.read_table(align_run, "alignments").to_pandas()
    sites = store.read_table(sites_run, "sites").to_pandas()
    spans = store.read_table(spans_run, "spans").to_pandas()

    rows = []
    for engine_id, group in alignments.groupby("engine_id", sort=True):
        engine_sites = sites[sites["engine_id"] == engine_id]
        engine_spans = spans[spans["engine_id"] == engine_id]
        resolved = (group["status"] == "resolved").sum()
        two_sided = group[
            group["ocr_span_ids"].map(len).gt(0) & group["gt_token_ids"].map(len).gt(0)
        ]
        rows.append(
            {
                "engine_id": engine_id,
                "n_components": len(group),
                "resolved_rate": float(resolved / len(group)) if len(group) else 0.0,
                # Over the components where AMBIGUOUS is REACHABLE. One-sided and orphan
                # components return a status before the confidence floor is consulted, so
                # including them pads the denominator by each engine's own insertion,
                # deletion and out-of-region volume -- which ranges from 30% to 60% of
                # components and differs enough to REVERSE the ranking. Tesseract reads as
                # the least ambiguous engine on the padded denominator and is joint-worst
                # on this one. The padded figure is kept beside it, clearly named.
                "ambiguity_rate": (
                    float((two_sided["status"] == "ambiguous").mean()) if len(two_sided) else 0.0
                ),
                "ambiguity_rate_all_components": float((group["status"] == "ambiguous").mean()),
                "n_two_sided_components": len(two_sided),
                "one_sided_share": float(1.0 - len(two_sided) / len(group)) if len(group) else 0.0,
                "unresolved_rate": float((group["status"] == "unresolved").mean()),
                "out_of_region_rate": float((group["status"] == "out_of_region").mean()),
                "n_spans": len(engine_spans),
                "granularity_split_rate": (
                    float(engine_spans["granularity_split"].mean())
                    if "granularity_split" in engine_spans and len(engine_spans)
                    else 0.0
                ),
                "n_sites": len(engine_sites),
                "n_clean_sites": int((engine_sites["d_before"] == 0).sum()),
                "n_evaluable_sites": (
                    int(engine_sites["evaluable"].sum())
                    if "evaluable" in engine_sites
                    else len(engine_sites)
                ),
                "mean_site_chars": (
                    float(engine_sites["ocr_text"].str.len().mean()) if len(engine_sites) else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def candidate_population(data: AnalysisInput, policy: HarmPolicy) -> pd.DataFrame:
    """What the candidate pool looks like, per engine, before any model is fitted.

    Different engines produce different candidate populations, and the population is the
    denominator of every risk and coverage number. An engine whose pool happens to be
    less harmful will look safer under any verifier, so the composition has to be visible
    beside the results rather than inferred from them.

    Reported on the natural pool only. The adversarial challenge set is ~99% harmful by
    construction, and pooling it here would describe the generator rather than the data.
    """
    if data.candidates.empty:
        return pd.DataFrame()

    frame = data.candidates.merge(
        data.labels[["candidate_id", "outcome_if_accepted", "d_before", "d_after", "delta"]],
        on="candidate_id",
        how="inner",
    )
    frame = data.natural(frame)
    if frame.empty:
        return pd.DataFrame()

    harmful_values = {o.value for o in harmful_outcomes(policy)}
    beneficial = {
        OutcomeIfAccepted.TRUE_CORRECTION.value,
        OutcomeIfAccepted.PARTIAL_IMPROVEMENT.value,
    }

    rows = []
    for engine_id, group in frame.groupby("engine_id", sort=True):
        outcomes = group["outcome_if_accepted"]
        n_sites = group["site_id"].nunique()
        clean = group[group["d_before"] == 0]
        broken = group[group["d_before"] > 0]
        rows.append(
            {
                "engine_id": engine_id,
                "n_candidates": len(group),
                "n_sites_with_a_candidate": n_sites,
                "candidates_per_site": float(len(group) / n_sites) if n_sites else 0.0,
                "harmful_fraction": float(outcomes.isin(harmful_values).mean()),
                "beneficial_fraction": float(outcomes.isin(beneficial).mean()),
                "true_correction_fraction": float(
                    (outcomes == OutcomeIfAccepted.TRUE_CORRECTION.value).mean()
                ),
                "lateral_fraction": float(
                    (outcomes == OutcomeIfAccepted.LATERAL_CHANGE.value).mean()
                ),
                # Where the two harms can even occur: overcorrection needs a clean site,
                # miscorrection needs a broken one. An engine with few clean sites cannot
                # overcorrect much, which is a property of its segmentation rather than
                # of its safety.
                #
                # There is deliberately no "overcorrection rate given a clean site" here.
                # At a clean site O equals G, so any candidate that changes the text has
                # d_after > 0 = d_before and is an overcorrection by definition: the rate
                # is structurally 1.0 and measures nothing. It was reported for one run
                # before that was noticed, which is how a column that looks like a
                # measurement and is not gets into a table.
                "overcorrection_opportunity": float(len(clean) / len(group)),
                "miscorrection_opportunity": float(len(broken) / len(group)),
                "mean_d_before": float(group["d_before"].mean()),
                "mean_edit_distance_proposed": float(
                    (group["d_before"] - group["delta"]).abs().mean()
                ),
                "median_delta": float(group["delta"].median()),
                "harm_policy": policy.value,
            }
        )
    return pd.DataFrame(rows)


def noise_tertiles(
    documents: pd.DataFrame, per_document_cer: Mapping[tuple[str, str], float]
) -> dict[tuple[str, str], str]:
    """Assign each (document, engine) a noise tertile **within its (corpus, engine) cell**.

    Within the cell, not globally. A global tertile over a benchmark whose corpora differ
    in difficulty would put nearly all of one corpus in the high-noise band and nearly all
    of another in the low, so the "noise" stratum would be a corpus label wearing a
    different name -- and the low-noise stratum H4 is about would stop being about noise.
    """
    dataset_of = dict(zip(documents["document_id"], documents["dataset_id"], strict=True))
    by_cell: dict[tuple[str, str], list[tuple[float, str]]] = {}
    for (document_id, engine_id), rate in per_document_cer.items():
        dataset_id = dataset_of.get(document_id)
        if dataset_id is None:
            continue
        by_cell.setdefault((str(dataset_id), engine_id), []).append((rate, document_id))

    tertile_of: dict[tuple[str, str], str] = {}
    for (_, engine_id), entries in by_cell.items():
        entries.sort()
        n = len(entries)
        for position, (_, document_id) in enumerate(entries):
            # Rank-based thirds rather than value cuts: CER distributions are skewed, and
            # equal-width bands would leave the top tertile nearly empty.
            third = 0 if position * 3 < n else 1 if position * 3 < 2 * n else 2
            tertile_of[(document_id, engine_id)] = ("low", "medium", "high")[third]
    return tertile_of


def stratified_risk(
    data: AnalysisInput,
    policy: HarmPolicy,
    stratum_of: Mapping[str, str],
    stratum_name: str,
) -> pd.DataFrame:
    """Base harm rate and pool composition per (fold, verifier, stratum).

    A pre-registered secondary analysis. Strata are supplied rather than derived here so
    the binning rule is fixed in one place and cannot be chosen after seeing the result.
    """
    frame = data.predictions.merge(
        data.labels[["candidate_id", "outcome_if_accepted"]], on="candidate_id", how="inner"
    ).merge(
        data.candidates[["candidate_id", "is_synthetic_hard_negative"]],
        on="candidate_id",
        how="left",
    )
    frame = data.natural(frame)
    if frame.empty:
        return pd.DataFrame()
    data.guard_headline(set(frame["fold_id"].unique()))

    frame = frame.assign(stratum=frame["document_id"].map(stratum_of))
    frame = frame[frame["stratum"].notna()]
    if frame.empty:
        return pd.DataFrame()

    rows = []
    for (fold_id, verifier_id, stratum), group in frame.groupby(
        ["fold_id", "verifier_id", "stratum"], sort=True
    ):
        harmful = _harmful(group, policy)
        scores = group["calibrated_score"].fillna(group["raw_score"]).to_numpy(dtype=float)
        curve = risk_coverage_curve(scores, harmful)
        rows.append(
            {
                "fold_id": fold_id,
                "verifier_id": verifier_id,
                "stratum_kind": stratum_name,
                "stratum": stratum,
                "n_candidates": len(group),
                "n_documents": group["document_id"].nunique(),
                "base_harm_rate": float(harmful.mean()),
                "aurc": aurc(curve),
                "harm_policy": policy.value,
            }
        )
    return pd.DataFrame(rows)


def confidence_reliability(store: object, experiment: str, n_bins: int = 10) -> pd.DataFrame:
    """Native OCR confidence against empirical span correctness, per engine.

    Purely descriptive and computed before any verifier exists. Each span is correct when
    the alignment component containing it matched its ground-truth token exactly; the
    confidence is the engine's own value mapped onto [0, 1] by its declared scale, never
    rescaled to make engines agree.

    Whether these curves have the same shape across engines is the question underneath
    H1: a verifier that learned what "0.8" means for one engine has to relearn it for
    another if they differ. Reported as a diagnostic, not as a test of the hypothesis.
    """
    from ocr_risk.canonical.confidence import normalize_confidence
    from ocr_risk.io.artifacts import ArtifactStore
    from ocr_risk.schemas.enums import StageName

    assert isinstance(store, ArtifactStore)
    align_run = store.latest(StageName.ALIGN, experiment)
    spans_run = store.latest(StageName.CANONICALIZE, experiment)
    if align_run is None or spans_run is None:
        return pd.DataFrame()

    spans = store.read_table(spans_run, "spans").to_pandas()
    alignments = store.read_table(align_run, "alignments").to_pandas()

    # A span is "correct" when the resolved component containing it has zero edit
    # distance. Unresolved and ambiguous components are excluded rather than counted as
    # wrong: the aligner declining to place a span says nothing about the recognition.
    correct_spans: set[str] = set()
    seen_spans: set[str] = set()
    for record in alignments.itertuples():
        if str(record.status) != "resolved":
            continue
        # `or []` is wrong on a numpy array: its truth value is ambiguous when empty.
        ids = [] if record.ocr_span_ids is None else list(record.ocr_span_ids)
        seen_spans.update(ids)
        if int(record.edit_distance) == 0:
            correct_spans.update(ids)

    rows = []
    for engine_id, group in spans.groupby("engine_id", sort=True):
        observations: list[tuple[float, int]] = []
        for record in group.itertuples():
            if record.span_id not in seen_spans:
                continue
            value = normalize_confidence(record.native_conf_recognition, record.conf_scale)
            if value is None:
                continue
            observations.append((value, int(record.span_id in correct_spans)))
        if not observations:
            continue
        for index in range(n_bins):
            low, high = index / n_bins, (index + 1) / n_bins
            members = [
                c
                for v, c in observations
                if (low <= v < high or (index == n_bins - 1 and v == 1.0))
            ]
            if not members:
                continue
            rows.append(
                {
                    "engine_id": engine_id,
                    "confidence_bin": (low + high) / 2.0,
                    "bin_low": low,
                    "bin_high": high,
                    "n": len(members),
                    "empirical_correct": float(sum(members) / len(members)),
                }
            )
    return pd.DataFrame(rows)
