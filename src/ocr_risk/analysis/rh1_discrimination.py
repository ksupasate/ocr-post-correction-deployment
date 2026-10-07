"""RH1: the matched-handicap contrast, on discrimination endpoints.

Deliberately a separate module from ``transfer.py``. That one measures H1 — a *calibration*
contrast against ``in_engine_oracle``, whose pre-registered verdict was NOT SUPPORTED and
is not revisited. This one measures RH1: whether transfer costs the model its ability to
**rank** beneficial edits above harmful ones, against a reference arm matched on training
volume, class balance, and feature availability.

Three differences from H1 that change what the number means:

1. **The reference is ``matched_in_engine``**, not ``in_engine_oracle``. Four of the five
   ways the H1 arms differed were training opportunity rather than engine exposure; see
   ``docs/h1_recovery/amendment4_confound.md``.
2. **The primary endpoint is ROC AUC on the RAW verifier score.** ROC AUC is invariant to
   a *strictly* monotone transform, which is the property the protocol invokes — but
   isotonic regression is a step function, and it collapsed 2 487-2 882 distinct raw scores
   to 28-83 distinct calibrated levels, by a different amount in each arm because the two
   calibrators are fitted on different engine mixes. Measured on the pilot predictions, the
   zero-shot-minus-reference delta moved by up to 0.0041 between raw and calibrated scores,
   on deltas of magnitude 0.002-0.036 — a factor of two on one pair. Scoring the raw output
   restores the property the endpoint was chosen for; scoring the calibrated output would
   have measured how finely each arm's step function happened to resolve the range.
3. **Every donor substitution is run and averaged over.** With four engines the reference
   arm's fit set depends on which engine the target replaces, and that choice moves the fit
   population by up to 13%. Aggregating over donors marginalizes it instead of fixing it
   arbitrarily.

Per target engine, never averaged across engines: four OCR engines are four fixed
environments, not four draws from a population of engines.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ocr_risk.analysis.tables import AnalysisInput, _harmful
from ocr_risk.metrics.calibration import murphy_decomposition
from ocr_risk.metrics.discrimination import (
    average_precision,
    resolution,
    roc_auc,
    score_separation,
)
from ocr_risk.schemas.enums import HarmPolicy
from ocr_risk.stats import (
    cluster_bootstrap_indices,
    holm_bonferroni,
    minimum_detectable_effect,
    paired_cluster_bootstrap_multi,
)

__all__ = ["RH1_METRICS", "RH1_PRIMARY", "rh1_discrimination_table"]

RH1_PRIMARY = "roc_auc"


# Higher is BETTER for every one of these, so degradation under transfer is a NEGATIVE
# delta — the opposite sign convention from transfer.py's error metrics. Keeping them in
# separate modules is partly to keep the two conventions from being confused.
RH1_METRICS = (
    "roc_auc",
    "average_precision",
    "resolution",
    "score_separation",
)


def _pair_of(fold_id: str) -> tuple[str, str]:
    """``matched_in_engine:tesseract<-doctr`` -> ``("tesseract", "doctr")``."""
    _, _, pair = fold_id.partition(":")
    target, _, donor = pair.partition("<-")
    return target, donor


def _values(scores: np.ndarray, safe: np.ndarray, n_bins: int) -> dict[str, float]:
    """Every RH1 endpoint for one arm on one candidate set."""
    if scores.size == 0:
        return dict.fromkeys(RH1_METRICS, float("nan"))
    base_rate = float(np.mean(safe))
    _, refinement, _ = murphy_decomposition(scores, safe, n_bins=n_bins, binning="equal_mass")
    return {
        "roc_auc": roc_auc(scores, safe),
        "average_precision": average_precision(scores, safe),
        "resolution": resolution(base_rate, refinement),
        "score_separation": score_separation(scores, safe),
    }


def _arm_frame(data: AnalysisInput, policy: HarmPolicy, arm: str) -> pd.DataFrame:
    """One arm's predictions, natural pool only, with harm and the pair resolved."""
    frame = data.predictions.merge(
        data.labels[["candidate_id", "outcome_if_accepted"]], on="candidate_id", how="inner"
    ).merge(
        data.candidates[["candidate_id", "is_synthetic_hard_negative"]],
        on="candidate_id",
        how="left",
    )
    frame = data.natural(frame)
    frame = frame[frame["fold_id"].str.startswith(f"{arm}:")]
    if frame.empty:
        return frame
    data.guard_headline(set(frame["fold_id"].unique()))
    pairs = [_pair_of(f) for f in frame["fold_id"]]
    return frame.assign(
        held_out_engine=[t for t, _ in pairs],
        donor_engine=[d for _, d in pairs],
        safe=(~_harmful(frame, policy)).astype(np.float64),
        # RAW, not calibrated: see the module docstring. Ranking is a property of the
        # verifier's output, and the calibrator is fitted on a different engine mix in
        # each arm, which is a dimension the match cannot equalize.
        score=frame["raw_score"].astype(float),
        calibrated=frame["calibrated_score"].fillna(frame["raw_score"]).astype(float),
    )


def _pooled_rows(
    left: pd.DataFrame,
    right: pd.DataFrame,
    engine: str,
    verifier: str,
    policy: HarmPolicy,
    n_bins: int,
    n_bootstrap: int,
    ci_level: float,
    seed: int,
    alpha: float,
) -> list[dict[str, object]]:
    """The per-engine row the verdict is read from: the donor substitution marginalized out.

    The protocol says "averaged over all three donor substitutions per target engine". The
    three pairs have *different* zero-shot models — each pair's subsample is drawn against
    its own partner — so the three contrasts are three genuine paired comparisons, and the
    average is over deltas rather than over scores. Resampling is joint: documents are
    drawn once and all three deltas are recomputed on the same draw, so the interval
    reflects that the three contrasts share an evaluation set rather than treating them as
    independent replicates.
    """
    # Donors of THIS engine. Taking the union over the whole table includes the engine
    # itself, which is never its own donor -- the first version did, found an empty frame,
    # and bailed out of every pooled row in the study.
    engine_left = left[left["held_out_engine"] == engine]
    engine_right = right[right["held_out_engine"] == engine]
    donors = sorted(set(engine_left["donor_engine"]) & set(engine_right["donor_engine"]))
    if not donors:
        return []

    per_donor: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    shared: set[str] | None = None
    for donor in donors:
        a = engine_left[engine_left["donor_engine"] == donor]
        b = engine_right[engine_right["donor_engine"] == donor]
        if a.empty or b.empty:
            return []
        ids = set(a["candidate_id"]) & set(b["candidate_id"])
        shared = ids if shared is None else shared & ids
        per_donor.append((a, b))
    if not shared:
        return []

    order = sorted(shared)
    frames = [
        (a.set_index("candidate_id").loc[order], b.set_index("candidate_id").loc[order])
        for a, b in per_donor
    ]
    # The per-donor loop above asserts the two arms agree on which candidates are safe;
    # the pooled path builds from those same frames, but the assertion is cheap and the
    # pairing is the whole design, so it is checked here too rather than trusted to the
    # loop's ordering.
    for a, b in frames:
        if not np.array_equal(a["safe"].to_numpy(), b["safe"].to_numpy()):
            msg = (
                f"pooled arms for {engine} disagree about which of the same "
                f"{len(order)} candidates are safe; the pairing is broken"
            )
            raise ValueError(msg)
    documents = list(frames[0][0]["document_id"])
    safe = list(frames[0][0]["safe"])
    # One document label per candidate, so a single cluster resample drives every donor.
    document_labels = [str(d) for d in documents]
    safe_array = np.asarray([float(v) for v in safe], dtype=np.float64)
    # (zero-shot, reference) score columns per donor, so a resample is an index into
    # arrays rather than a rebuilt list of tuples.
    scores = [
        (a["score"].to_numpy(dtype=np.float64), b["score"].to_numpy(dtype=np.float64))
        for a, b in frames
    ]

    # One bootstrap pass for all four metrics, not one pass each. `_values` computes the
    # whole set from a single sweep, so asking it per metric threw away three quarters of
    # its work on every resample -- and this statistic evaluates it six times (three
    # donors x two arms), which made the pooled rows the dominant cost of the analysis.
    cache: dict[bytes, dict[str, float]] = {}

    def mean_deltas(indices: NDArray[np.intp]) -> dict[str, float]:
        labels = safe_array[indices]
        totals = dict.fromkeys(RH1_METRICS, 0.0)
        for zero_shot, reference in scores:
            left = _values(zero_shot[indices], labels, n_bins)
            right = _values(reference[indices], labels, n_bins)
            for metric in RH1_METRICS:
                totals[metric] += left[metric] - right[metric]
        return {metric: totals[metric] / len(scores) for metric in RH1_METRICS}

    def statistic_for(metric: str) -> Callable[[NDArray[np.intp]], float]:
        def statistic(indices: NDArray[np.intp]) -> float:
            key = indices.tobytes()
            if key not in cache:
                cache.clear()  # one resample is live at a time: a memo, not a store
                cache[key] = mean_deltas(indices)
            return cache[key][metric]

        return statistic

    rows: list[dict[str, object]] = []
    for metric in RH1_METRICS:
        result = cluster_bootstrap_indices(
            document_labels,
            statistic_for(metric),
            n_resamples=n_bootstrap,
            ci_level=ci_level,
            seed=seed,
            # A difference of ranking statistics lives in [-1, 1] for AUC and AP and is
            # unbounded for Cohen's d, so no fabricated bound is offered on a collapse.
            bounds=None,
        )
        zero_shot_level = float(
            np.mean(
                [
                    _values(a["score"].to_numpy(), a["safe"].to_numpy(), n_bins)[metric]
                    for a, _ in frames
                ]
            )
        )
        reference_level = float(
            np.mean(
                [
                    _values(b["score"].to_numpy(), b["safe"].to_numpy(), n_bins)[metric]
                    for _, b in frames
                ]
            )
        )
        finite = np.isfinite(result.lower) and np.isfinite(result.upper)
        rows.append(
            {
                "held_out_engine": engine,
                "donor_engine": "ALL",
                "aggregation": "pooled",
                "n_donors": len(frames),
                "verifier_id": verifier,
                "metric": metric,
                "loeo_zero_shot": zero_shot_level,
                "matched_in_engine": reference_level,
                "delta": result.estimate,
                "delta_ci_lower": result.lower,
                "delta_ci_upper": result.upper,
                "n_candidates": len(order),
                "n_documents": result.n_clusters,
                "p_value": result.p_value_two_sided,
                "standard_error": result.standard_error,
                "minimum_detectable_effect": minimum_detectable_effect(
                    result.standard_error, alpha=alpha, n_tests=4
                ),
                "degraded": bool(finite and result.upper < 0.0),
                "harm_policy": policy.value,
                "n_bins": n_bins,
                "binning": "equal_mass",
                "n_bootstrap": n_bootstrap,
                "ci_level": ci_level,
                "degenerate_interval": bool(result.degenerate_interval),
                "base_rate_safe": float(np.mean(safe)),
                "score_channel": "raw",
            }
        )
    return rows


def rh1_discrimination_table(
    data: AnalysisInput,
    policy: HarmPolicy,
    *,
    verifiers: Sequence[str] | None = None,
    n_bins: int = 15,
    n_bootstrap: int = 10_000,
    ci_level: float = 0.95,
    seed: int = 7,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Per target engine: the pooled contrast, plus one diagnostic row per donor.

    Both arms live in one run, so they are joined from one artifact rather than across two.
    Candidates are intersected before anything is computed: a paired contrast on different
    rows is not paired, and the whole design rests on the two arms seeing the same edits.

    Holm is applied across the four target engines within each ``(metric, verifier)`` on the
    **pooled** rows — the family named in the frozen protocol. The per-donor rows are
    diagnostic and carry no adjusted p: correcting across them would make each family three
    engines wide instead of four, which is weaker than what was pre-registered.
    """
    left = _arm_frame(data, policy, "loeo_zero_shot")
    right = _arm_frame(data, policy, "matched_in_engine")
    if left.empty or right.empty:
        return pd.DataFrame()
    if verifiers:
        left = left[left["verifier_id"].isin(verifiers)]
        right = right[right["verifier_id"].isin(verifiers)]

    rows: list[dict[str, object]] = []
    keys = ["held_out_engine", "donor_engine", "verifier_id"]
    for (engine, donor, verifier), zero_shot in left.groupby(keys, sort=True):
        reference = right[
            (right["held_out_engine"] == engine)
            & (right["donor_engine"] == donor)
            & (right["verifier_id"] == verifier)
        ]
        if reference.empty:
            continue
        shared = sorted(set(zero_shot["candidate_id"]) & set(reference["candidate_id"]))
        if not shared:
            continue
        a = zero_shot.set_index("candidate_id").loc[shared]
        b = reference.set_index("candidate_id").loc[shared]

        # The two arms must be scored on bit-identical outcomes, or the contrast has a
        # second moving part. Checked rather than assumed: the arms are built separately.
        if not np.array_equal(a["safe"].to_numpy(), b["safe"].to_numpy()):
            msg = (
                f"arms for {engine}<-{donor} disagree about which of the same "
                f"{len(shared)} candidates are safe; the pairing is broken"
            )
            raise ValueError(msg)

        zero_shot_values = _values(a["score"].to_numpy(), a["safe"].to_numpy(), n_bins)
        reference_values = _values(b["score"].to_numpy(), b["safe"].to_numpy(), n_bins)

        items_a = list(zip(a["document_id"], a["score"], a["safe"], strict=True))
        items_b = list(zip(b["document_id"], b["score"], b["safe"], strict=True))

        def every_metric(items: Sequence[tuple[str, float, float]]) -> dict[str, float]:
            if not items:
                return dict.fromkeys(RH1_METRICS, float("nan"))
            return _values(
                np.array([s for _, s, _ in items], dtype=float),
                np.array([v for _, _, v in items], dtype=float),
                n_bins,
            )

        paired_by_metric = paired_cluster_bootstrap_multi(
            items_a,
            items_b,
            lambda item: str(item[0]),
            every_metric,
            n_resamples=n_bootstrap,
            ci_level=ci_level,
            seed=seed,
        )

        for metric in RH1_METRICS:
            paired = paired_by_metric[metric]
            finite = np.isfinite(paired.lower) and np.isfinite(paired.upper)
            rows.append(
                {
                    "held_out_engine": engine,
                    "donor_engine": donor,
                    "aggregation": "per_donor",
                    "n_donors": 1,
                    "verifier_id": verifier,
                    "metric": metric,
                    "loeo_zero_shot": zero_shot_values[metric],
                    "matched_in_engine": reference_values[metric],
                    "delta": paired.estimate,
                    "delta_ci_lower": paired.lower,
                    "delta_ci_upper": paired.upper,
                    "n_candidates": len(shared),
                    "n_documents": paired.n_clusters,
                    "p_value": paired.p_value_two_sided,
                    "standard_error": paired.standard_error,
                    "minimum_detectable_effect": minimum_detectable_effect(
                        paired.standard_error, alpha=alpha, n_tests=4
                    ),
                    "degraded": bool(finite and paired.upper < 0.0),
                    "harm_policy": policy.value,
                    "n_bins": n_bins,
                    "binning": "equal_mass",
                    "n_bootstrap": n_bootstrap,
                    "ci_level": ci_level,
                    "degenerate_interval": bool(paired.degenerate_interval),
                    "base_rate_safe": float(a["safe"].mean()),
                    "score_channel": "raw",
                }
            )

    for verifier in sorted(set(left["verifier_id"]) & set(right["verifier_id"])):
        for engine in sorted(set(left["held_out_engine"]) & set(right["held_out_engine"])):
            rows.extend(
                _pooled_rows(
                    left[left["verifier_id"] == verifier],
                    right[right["verifier_id"] == verifier],
                    engine,
                    verifier,
                    policy,
                    n_bins,
                    n_bootstrap,
                    ci_level,
                    seed,
                    alpha,
                )
            )

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    # Multiplicity across the four TARGET ENGINES within each (metric, verifier), on the
    # POOLED rows -- the family declared in docs/h1_recovery/h2_readiness_protocol.md
    # section 4. Adding the donor to the key would make each family three engines wide,
    # which is weaker than what was pre-registered; the per-donor rows are diagnostic and
    # are left unadjusted rather than corrected in a family nobody declared.
    table["holm_adjusted_p"] = float("nan")
    table["degraded_after_holm"] = False
    pooled = table[table["aggregation"] == "pooled"]
    for _key, family in pooled.groupby(["metric", "verifier_id"], sort=True):
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
            table.loc[index, "degraded_after_holm"] = bool(row["degraded"]) and test.significant

    return table
