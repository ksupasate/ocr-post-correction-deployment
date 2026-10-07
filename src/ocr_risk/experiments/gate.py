"""The pilot go/no-go gate.

Evaluates H1-H4 against pre-registered criteria and emits a machine-readable verdict. It
exists so that a negative result is a *reportable outcome of running the gate*, not an
awkward silence — and so nobody has to decide after the fact what would have counted as
success.

Every verdict is one of:

``GO``            the criterion was met
``NO_GO``         the criterion was tested and failed — a real finding, with a pivot
``INCONCLUSIVE``  the data could not decide (too few folds, no feasible operating point)

``INCONCLUSIVE`` is deliberately distinct from ``NO_GO``. "We looked and it did not help"
and "we could not tell" call for different next steps, and collapsing them would let a
weak experiment masquerade as evidence of absence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

__all__ = ["GateReport", "HypothesisVerdict", "evaluate_gate"]

GO = "GO"
NO_GO = "NO_GO"
INCONCLUSIVE = "INCONCLUSIVE"

# H1 is judged on its own four-way vocabulary, fixed in docs/preregistration.md before
# the results were seen. "Engine-dependent" is a distinct scientific finding from "no
# effect", and collapsing it into GO/NO_GO would discard the heterogeneity that is the
# most useful thing four fixed engines can tell us.
H1_SUPPORTED = "H1 SUPPORTED"
H1_PARTIALLY_SUPPORTED = "H1 PARTIALLY SUPPORTED"
H1_NOT_SUPPORTED = "H1 NOT SUPPORTED"
H1_INCONCLUSIVE = "H1 INCONCLUSIVE"


@dataclass(frozen=True, slots=True)
class HypothesisVerdict:
    """The outcome for one hypothesis, with the evidence that produced it."""

    hypothesis: str
    question: str
    verdict: str
    criterion: str
    observed: str
    implication: str
    evidence: dict[str, float | int | str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "hypothesis": self.hypothesis,
            "question": self.question,
            "verdict": self.verdict,
            "criterion": self.criterion,
            "observed": self.observed,
            "implication": self.implication,
            "evidence": self.evidence,
        }


@dataclass(slots=True)
class GateReport:
    """The full pilot verdict."""

    experiment: str
    synthetic: bool
    harm_policy: str = ""
    """Which outcomes counted as harmful. The H1 verdict is policy-dependent -- the same
    candidates give 0/4 degraded engines under one policy and 2/4 under another -- so a
    report that does not name its policy is not interpretable."""
    verdicts: list[HypothesisVerdict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def any_go(self) -> bool:
        return any(v.verdict == GO for v in self.verdicts)

    def as_dict(self) -> dict[str, object]:
        return {
            "experiment": self.experiment,
            "synthetic": self.synthetic,
            "harm_policy": self.harm_policy,
            "verdicts": [v.as_dict() for v in self.verdicts],
            "notes": self.notes,
            "summary": {v.hypothesis: v.verdict for v in self.verdicts},
        }

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n", "utf-8")
        return path


def evaluate_gate(
    experiment: str,
    *,
    synthetic: bool,
    coverage_table: pd.DataFrame,
    calibration: pd.DataFrame,
    harm: pd.DataFrame,
    epsilon: float,
    min_folds: int = 2,
    transfer: pd.DataFrame | None = None,
    paired_h2: pd.DataFrame | None = None,
    primary_verifier: str = "v6_full",
    harm_policy: str = "",
) -> GateReport:
    """Judge H1-H4 from the analysis tables.

    ``transfer`` is the paired oracle-vs-zero-shot table that H1 is defined on. Without
    it H1 reports ``H1 INCONCLUSIVE`` rather than substituting a within-run proxy: a
    spread across folds is a different quantity from a transfer gap, and reporting one as
    the other would answer a question nobody asked.
    """
    report = GateReport(experiment=experiment, synthetic=synthetic, harm_policy=harm_policy)
    if synthetic:
        report.notes.append(
            "SYNTHETIC RUN. These verdicts validate that the gate executes and can return "
            "each outcome; they are not evidence about the hypotheses."
        )

    at_epsilon = (
        coverage_table[coverage_table["epsilon"] == epsilon]
        if not coverage_table.empty and "epsilon" in coverage_table
        else pd.DataFrame()
    )
    n_folds = at_epsilon["fold_id"].nunique() if not at_epsilon.empty else 0

    report.verdicts.append(
        _h1(
            transfer if transfer is not None else pd.DataFrame(),
            primary_verifier=primary_verifier,
        )
    )
    report.verdicts.append(_h2(at_epsilon, epsilon, n_folds, min_folds, paired_h2))
    report.verdicts.append(_h3(at_epsilon, n_folds, min_folds))
    report.verdicts.append(_h4(harm, epsilon))
    return report


def _h1(
    transfer: pd.DataFrame,
    *,
    primary_metric: str = "calibration_error",
    primary_verifier: str = "v6_full",
    min_engines: int = 2,
) -> HypothesisVerdict:
    """Cross-engine calibration shift, from the paired oracle-vs-zero-shot contrast.

    Reads ``analysis.transfer.h1_transfer_table``: per target engine, the same candidates
    scored under both protocols, with a paired document-level bootstrap on the
    difference. The primary metric is the Murphy **calibration term** rather than raw
    Brier — engines differ in base harm rate by construction, so a raw-Brier contrast
    would partly measure base rates rather than calibration.
    """
    criterion = (
        "for the primary evidence configuration, per target engine: loeo_zero_shot minus "
        "in_engine_oracle in Murphy calibration error on matched candidates, with a paired "
        "document-level bootstrap interval excluding zero after Holm adjustment across "
        "the four engines"
    )
    question = "Does calibration degrade under unseen-engine shift?"

    if transfer.empty or "metric" not in transfer.columns:
        return HypothesisVerdict(
            hypothesis="H1",
            question=question,
            verdict=H1_INCONCLUSIVE,
            criterion=criterion,
            observed="no paired transfer comparison available",
            implication=(
                "H1 is a contrast between two protocols. Run both loeo_zero_shot and "
                "in_engine_oracle; a single run cannot answer it."
            ),
        )

    primary = transfer[transfer["metric"] == primary_metric]
    if primary.empty:
        return HypothesisVerdict(
            hypothesis="H1",
            question=question,
            verdict=H1_INCONCLUSIVE,
            criterion=criterion,
            observed=f"metric {primary_metric!r} absent from the transfer table",
            implication="The primary metric was not computed; H1 cannot be judged.",
        )

    # ONE verifier, then per engine. Taking `.any()` across the seven evidence
    # configurations gave each engine seven uncorrected chances to register degradation,
    # inflating the false-positive rate roughly sevenfold — and the multiplicity control
    # is declared over ENGINES, not over the ablation ladder, which is H2's question.
    #
    # The pre-registration did not name a primary verifier; that is a gap in it, recorded
    # as an amendment. `v6_full` is named because it is the configuration with every
    # evidence channel — the project's proposed method by construction, which is true
    # independently of any result. Every verifier's per-engine count is reported in the
    # evidence below, so the reader can see what a different choice would have given.
    by_verifier = primary.groupby("verifier_id")["degraded_after_holm"].sum().sort_index().to_dict()
    # Every metric's count for the primary verifier. The verdict is read off ONE endpoint,
    # and burying the pre-registered endpoints that disagree would make the verdict a
    # selection rather than a finding. On this pilot the calibration term and Brier point
    # opposite ways, and a reader shown only the first has been misled.
    same_verifier = transfer[transfer["verifier_id"] == primary_verifier]
    by_metric = {
        str(metric): int(group["degraded_after_holm"].sum())
        for metric, group in same_verifier.groupby("metric", sort=True)
    }
    n_engines_total = int(same_verifier["held_out_engine"].nunique()) or 1
    disagreeing = sorted(
        metric
        for metric, count in by_metric.items()
        if metric != primary_metric and count == n_engines_total
    )
    arm = primary[primary["verifier_id"] == primary_verifier]
    if arm.empty:
        return HypothesisVerdict(
            hypothesis="H1",
            question=question,
            verdict=H1_INCONCLUSIVE,
            criterion=criterion,
            observed=f"the primary verifier {primary_verifier!r} is absent from the table",
            implication="H1 is judged on one named configuration; that one did not run.",
        )
    per_engine = arm.groupby("held_out_engine")["degraded_after_holm"].any()
    n_engines = int(per_engine.size)
    if n_engines < min_engines:
        return HypothesisVerdict(
            hypothesis="H1",
            question=question,
            verdict=H1_INCONCLUSIVE,
            criterion=criterion,
            observed=f"only {n_engines} target engine(s) with a paired comparison",
            implication=(
                "Too few target engines to distinguish consistent degradation from an "
                "engine-specific effect."
            ),
        )

    n_degraded = int(per_engine.sum())
    usable = primary["delta"].notna().sum()
    if usable == 0:
        verdict = H1_INCONCLUSIVE
    elif n_degraded == 0:
        verdict = H1_NOT_SUPPORTED
    elif n_degraded == n_engines:
        verdict = H1_SUPPORTED
    else:
        verdict = H1_PARTIALLY_SUPPORTED

    degraded_engines = sorted(per_engine[per_engine].index)
    # A null is only informative if the design could have seen the effect. Comparing the
    # minimum detectable effect to the reference arm's own error says which kind of null
    # this is, and the verdict text carries it so the number cannot be read without it.
    mde_ratio = float("nan")
    if "minimum_detectable_effect" in arm.columns:
        reference = arm["in_engine_oracle"].replace(0.0, float("nan"))
        mde_ratio = float((arm["minimum_detectable_effect"] / reference).median())

    implications = {
        H1_SUPPORTED: (
            "Every target engine shows degraded calibration under zero-shot transfer. "
            "Risk control for OCR post-correction must contend with engine shift, and "
            "the source-grounded verification question is worth pursuing."
        ),
        H1_PARTIALLY_SUPPORTED: (
            "Transfer degradation is real but engine-dependent. The effect exists and is "
            "not universal, so any claim must be stated per engine; report which engines "
            "and look for what distinguishes them before building on it."
        ),
        H1_NOT_SUPPORTED: (
            f"No target engine shows {primary_metric} degradation this design could "
            "detect. Two qualifications belong with that sentence, not after it. First, "
            f"the median minimum detectable effect is {mde_ratio:.1f}x the reference "
            "arm's own error, so the result rules out a degradation of roughly that size "
            "and not a smaller one -- a null whose detectable effect approaches the "
            "quantity being compared is weak evidence of absence. Second"
            + (
                f", the pre-registered endpoint(s) {', '.join(disagreeing)} degraded on "
                f"ALL {n_engines_total} engines for the same verifier on the same rows, "
                "so the verdict reflects which endpoint was named primary rather than an "
                "absence of any transfer effect."
                if disagreeing
                else ", no other pre-registered endpoint contradicts it."
            )
        ),
        H1_INCONCLUSIVE: "The comparison ran but produced no usable difference estimate.",
    }
    return HypothesisVerdict(
        hypothesis="H1",
        question=question,
        verdict=verdict,
        criterion=criterion,
        observed=(
            f"{n_degraded} of {n_engines} target engines show degraded {primary_metric} "
            f"after Holm" + (f" ({', '.join(degraded_engines)})" if degraded_engines else "")
        ),
        implication=implications[verdict],
        evidence={
            "primary_metric": primary_metric,
            "primary_verifier": primary_verifier,
            "n_engines": n_engines,
            "n_engines_degraded": n_degraded,
            "degraded_engines": ", ".join(degraded_engines) if degraded_engines else "none",
            # Per engine, not a median. Four fixed environments summarized by one number
            # is the thing the protocol forbids, even when it is not a CI or a p-value.
            "delta_by_engine": ", ".join(
                f"{row['held_out_engine']}={row['delta']:+.5f}"
                for _, row in arm.sort_values("held_out_engine").iterrows()
            ),
            "minimum_detectable_effect_by_engine": (
                ", ".join(
                    f"{row['held_out_engine']}={row['minimum_detectable_effect']:.5f}"
                    for _, row in arm.sort_values("held_out_engine").iterrows()
                )
                if "minimum_detectable_effect" in arm.columns
                else "not computed"
            ),
            "mde_as_multiple_of_reference_error": mde_ratio,
            "sensitivity_adequate": bool(mde_ratio < 1.0),
            "engines_degraded_by_metric": ", ".join(
                f"{metric}={count}/{n_engines_total}" for metric, count in by_metric.items()
            ),
            "endpoints_degraded_on_every_engine": ", ".join(disagreeing) or "none",
            # What every other evidence configuration would have given, so the effect of
            # naming one is visible rather than hidden.
            "engines_degraded_by_verifier": ", ".join(
                f"{verifier}={int(count)}/{n_engines}" for verifier, count in by_verifier.items()
            ),
        },
    )


def _h2(
    at_epsilon: pd.DataFrame,
    epsilon: float,
    n_folds: int,
    min_folds: int,
    paired: pd.DataFrame | None = None,
) -> HypothesisVerdict:
    """Incremental value of pixels: does V6 beat V3?

    Judged on the **paired document-level bootstrap interval** for the per-fold V6-minus-
    V3 difference, as pre-registered. A bare mean comparison would return GO on a
    difference of 1e-9, and would let one outlier fold outvote three folds that went the
    other way.
    """
    criterion = (
        "per fold, coverage(V6) - coverage(V3) at the primary tolerance, with a paired "
        "document-level bootstrap interval excluding zero; GO requires a majority of "
        "folds favouring V6 and none favouring V3 with an interval excluding zero"
    )
    question = "Do source pixels help beyond text and confidence?"

    if paired is None or paired.empty:
        return HypothesisVerdict(
            hypothesis="H2",
            question=question,
            verdict=INCONCLUSIVE,
            criterion=criterion,
            observed="no paired V3-vs-V6 comparison was computed",
            implication=(
                "H2 is defined on a paired interval, not on a mean. Without it the "
                "comparison is not made rather than made and failed."
            ),
        )

    at_primary = paired[paired["epsilon"] == epsilon]
    if at_primary.empty or at_primary["delta"].isna().all():
        return HypothesisVerdict(
            hypothesis="H2",
            question=question,
            verdict=INCONCLUSIVE,
            criterion=criterion,
            observed=(
                f"no fold produced a feasible operating point for both V3 and V6 at "
                f"epsilon={epsilon}"
            ),
            implication="Neither arm reached the tolerance; the comparison is undefined.",
        )

    favours_v6 = at_primary["ci_lower"] > 0.0
    favours_v3 = at_primary["ci_upper"] < 0.0
    n_total = len(at_primary)
    n_v6, n_v3 = int(favours_v6.sum()), int(favours_v3.sum())
    met = n_v6 > n_total / 2 and n_v3 == 0

    return HypothesisVerdict(
        hypothesis="H2",
        question=question,
        verdict=GO if met else NO_GO,
        criterion=criterion,
        observed=(
            f"{n_v6}/{n_total} folds favour V6 with an interval excluding zero; {n_v3} favour V3"
        ),
        implication=(
            "Source-grounded verification adds coverage over text plus confidence."
            if met
            else "Pixels did not add coverage here. V4 (image alone) disambiguates whether "
            "the image channel carries no signal in this corpus, or carries signal that "
            "text and confidence already capture."
        ),
        evidence={
            "n_folds": n_total,
            "n_folds_favouring_v6": n_v6,
            "n_folds_favouring_v3": n_v3,
            "median_delta": float(at_primary["delta"].median()),
            "epsilon": epsilon,
        },
    )


def _h3(at_epsilon: pd.DataFrame, n_folds: int, min_folds: int) -> HypothesisVerdict:
    """Verify-vs-rewrite. The rewriting baseline is not implemented in this bootstrap."""
    return HypothesisVerdict(
        hypothesis="H3",
        question="Does generate-then-verify beat direct multimodal rewriting?",
        verdict=INCONCLUSIVE,
        criterion="coverage(verify pipeline) > coverage(multimodal_rewriter) at equal risk",
        observed="the multimodal rewriting baseline is declared but not implemented",
        implication=(
            "Cannot be judged in this bootstrap, and is reported as untested rather than "
            "as an absent effect. Requires the direct-rewriting baseline (next milestone)."
        ),
    )


def _h4(harm: pd.DataFrame, epsilon: float | None = None) -> HypothesisVerdict:
    """Low-noise reliability: does risk control protect already-good OCR?

    Two things this deliberately does not do. It does not average the overcorrection rate
    across tolerances, folds and methods and compare that to the *mean* tolerance — an
    arm breaking 20% of clean spans at a strict tolerance can otherwise be outvoted by
    twenty near-abstaining arms. And it does not compare the overcorrection rate to
    epsilon as though they shared a denominator: epsilon bounds ``harmful accepted /
    accepted edits``, while the overcorrection rate is over *clean sites*. The
    accepted-edit risk is what epsilon governs, so that is what is tested; the
    overcorrection rate is reported beside it as the H4-specific quantity.
    """
    criterion = (
        "at the primary tolerance, every operating point that accepted any edit keeps "
        "accepted-edit risk within tolerance, and the overcorrection rate on clean sites "
        "is reported per operating point rather than averaged"
    )
    question = "Does risk control protect already-high-quality OCR?"
    required = {"overcorrection_rate", "accepted_edit_risk", "n_accepted", "epsilon"}
    if harm.empty or not required <= set(harm.columns):
        return HypothesisVerdict(
            hypothesis="H4",
            question=question,
            verdict=INCONCLUSIVE,
            criterion=criterion,
            observed="no harm decomposition available",
            implication="Run the decide stage before judging this.",
        )

    scoped = harm if epsilon is None else harm[harm["epsilon"] == epsilon]
    accepted = scoped[scoped["n_accepted"] > 0]
    if accepted.empty:
        return HypothesisVerdict(
            hypothesis="H4",
            question=question,
            verdict=INCONCLUSIVE,
            criterion=criterion,
            observed="every method abstained; no edits were accepted anywhere",
            implication=(
                "Total abstention is safe but uninformative: with no accepted edits there "
                "is no overcorrection to measure. Either the tolerance is unreachable on "
                "this candidate pool or the verifiers carry too little signal."
            ),
        )

    # Each operating point against its OWN tolerance, then require all of them.
    within = accepted["accepted_edit_risk"] <= accepted["epsilon"]
    n_within, n_points = int(within.sum()), len(accepted)
    met = n_within == n_points
    worst = accepted.loc[(accepted["accepted_edit_risk"] - accepted["epsilon"]).idxmax()]

    return HypothesisVerdict(
        hypothesis="H4",
        question=question,
        verdict=GO if met else NO_GO,
        criterion=criterion,
        observed=(
            f"{n_within}/{n_points} operating points hold accepted-edit risk within their "
            f"own tolerance; worst is {worst['accepted_edit_risk']:.4f} against "
            f"{worst['epsilon']:g}"
        ),
        implication=(
            "Clean spans were preserved and the tolerance held at every operating point."
            if met
            else "At least one operating point broke the tolerance it was certified for, "
            "which is the failure mode H4 predicts matters most on low-noise input."
        ),
        evidence={
            "n_operating_points": n_points,
            "n_within_tolerance": n_within,
            "max_accepted_edit_risk": float(accepted["accepted_edit_risk"].max()),
            "max_overcorrection_rate": float(accepted["overcorrection_rate"].max()),
            "median_overcorrection_rate": float(accepted["overcorrection_rate"].median()),
        },
    )
