"""The leave-one-engine-out experiment.

Per fold, in this order and no other:

1. fit the featurizer and verifier on ``train engines x D_fit``;
2. fit the calibrator on ``train engines x D_cal``;
3. select tau(epsilon) on the same calibration scope;
4. **freeze all three**;
5. score and decide on ``held-out engine x D_test``.

Steps 1-3 never see the held-out engine or the test documents. That is enforced upstream by
``SplitPlan`` handing out role-scoped views, and re-verified downstream by the post-hoc
audit reading the run record — the runner itself does not get to be trusted.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from ocr_risk.calibrate import build_calibrator
from ocr_risk.config.models import ExperimentConfig
from ocr_risk.evidence import EvidenceMask
from ocr_risk.evidence.features_conf import ConfidenceFeaturizer
from ocr_risk.experiments.matched import FoldMatch
from ocr_risk.risk import ScoredCandidate, decide_sites, select_threshold
from ocr_risk.schemas.decisions import Decision
from ocr_risk.schemas.enums import DecisionAction, HarmPolicy, SplitRole, harmful_outcomes
from ocr_risk.schemas.evidence import EvidenceBundle
from ocr_risk.schemas.predictions import Prediction
from ocr_risk.schemas.run_record import SplitDescriptor
from ocr_risk.splits import LeakageError, SplitPlan
from ocr_risk.verify import VerificationInput, build_verifier

__all__ = ["FoldResult", "MethodSpec", "run_fold"]


@dataclass(frozen=True, slots=True)
class _EvalRow:
    """The evaluation-frame columns the runner reads, with real types."""

    site_id: str
    document_id: str
    dataset_id: str
    engine_id: str
    generator_rank: int


@dataclass(frozen=True, slots=True)
class ConfSample:
    """The raw confidence readings behind one candidate, kept out of the bundle.

    A z-score needs a population, and the admissible population is fold-dependent: the
    fit engines of one fold include the engine another fold holds out. So the bundle
    carries no z-score and the fold recomputes it from these raw readings.
    """

    engine_id: str
    native_confidences: tuple[float | None, ...]
    conf_scale: str | None


@dataclass(frozen=True, slots=True)
class MethodSpec:
    """One (verifier, evidence configuration, calibrator) combination to evaluate."""

    verifier_id: str
    kind: str
    evidence_config: str
    calibration_method: str
    params: dict[str, object] = field(default_factory=dict)

    @property
    def method_id(self) -> str:
        return f"{self.verifier_id}|{self.evidence_config}|{self.calibration_method}"


@dataclass(slots=True)
class FoldResult:
    """Everything one fold produced, ready to be written as artifacts."""

    fold_id: str
    method_id: str
    descriptor: SplitDescriptor
    predictions: list[Prediction] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    thresholds: list[dict[str, object]] = field(default_factory=list)
    calibrator_id: str = ""
    diagnostics: dict[str, object] = field(default_factory=dict)


def _harmful_flags(frame: pd.DataFrame, policy: HarmPolicy) -> np.ndarray:
    """Derive harm from the outcome enum and the configured policy.

    Never read from a stored boolean: harmfulness is policy-dependent, and materializing
    it would let the stored value drift from the taxonomy it came from.
    """
    harmful_values = {o.value for o in harmful_outcomes(policy)}
    return frame["outcome_if_accepted"].isin(harmful_values).to_numpy(dtype=bool)


def _selection_mask(frame: pd.DataFrame, config: ExperimentConfig) -> np.ndarray:
    """Rows eligible for threshold selection and headline evaluation.

    Excludes synthesized hard negatives unless the config opts them in. They are still
    fitted on and still evaluated — as a separately reported challenge set.
    """
    if config.candidates.hard_negatives_in_evaluation:
        return np.ones(len(frame), dtype=bool)
    if "is_synthetic_hard_negative" not in frame.columns:
        return np.ones(len(frame), dtype=bool)
    return ~frame["is_synthetic_hard_negative"].to_numpy(dtype=bool)


def _inputs(
    frame: pd.DataFrame,
    bundles: dict[str, EvidenceBundle],
    mask: EvidenceMask,
    crops: dict[str, Path],
    featurizer: ConfidenceFeaturizer | None = None,
    conf_samples: dict[str, ConfSample] | None = None,
    drop_zscore: bool = False,
) -> list[VerificationInput]:
    """Mask each bundle before the verifier ever sees it.

    The fold's own confidence featurizer is applied first, so the z-score a verifier
    reads was estimated on this fold's fit engines and no others. An engine absent from
    that fit — the held-out one — yields ``None``, which is the honest answer: its
    confidence scale was never observed.
    """
    items: list[VerificationInput] = []
    for candidate_id in frame["candidate_id"]:
        bundle = bundles.get(candidate_id)
        if bundle is None:
            continue
        if featurizer is not None and conf_samples is not None:
            sample = conf_samples.get(candidate_id)
            if sample is not None:
                features = featurizer.transform(
                    sample.native_confidences,
                    sample.conf_scale,
                    engine_id=sample.engine_id,
                )
                if drop_zscore and features.conf_zscore_within_engine is not None:
                    # The matched handicap. A zero-shot arm structurally cannot standardize
                    # an engine it never saw, so a reference arm that can is being compared
                    # on one extra feature rather than on what it learned. Discarding it
                    # here -- after the transform, so the two arms take the same code path
                    # -- leaves both with an imputed 0.0 and the same missing indicator.
                    features = features.model_copy(update={"conf_zscore_within_engine": None})
                bundle = bundle.model_copy(update={"conf_features": features})
        masked = mask.apply(bundle)
        # The crop is withheld too, not merely unused: an ablation that could still open
        # the file would depend on the featurizer's restraint.
        crop = crops.get(candidate_id) if masked.crop_recipe_sha256 else None
        items.append(VerificationInput(bundle=masked, crop=crop))
    return items


def run_fold(
    plan: SplitPlan,
    method: MethodSpec,
    config: ExperimentConfig,
    bundles: dict[str, EvidenceBundle],
    crops: dict[str, Path] | None = None,
    conf_samples: dict[str, ConfSample] | None = None,
    match: FoldMatch | None = None,
    fold_id: str | None = None,
) -> FoldResult:
    """Run one fold end to end under one method.

    ``match`` is the matched-handicap design: it cuts the fit and calibration views down to
    the sample this fold's partner arm can also supply, and ablates the target engine's
    confidence z-score so neither arm has a feature the other structurally cannot compute.
    It never touches the evaluation view.

    ``fold_id`` overrides the plan's, and must reach the prediction and decision rows too.
    One zero-shot plan is shared by a target engine's three donor pairs, so relabelling
    only the result object left every prediction carrying the plan's id — three runs
    indistinguishable in the artifact, and a paired join that silently collapses them.
    """
    crops = crops or {}
    conf_samples = conf_samples or {}
    mask = EvidenceMask.from_key(method.evidence_config)
    policy = config.risk.harm_policy
    label = fold_id or plan.fold_id

    fit_view = plan.view(SplitRole.FIT)
    calibrate_view = plan.view(SplitRole.CALIBRATE)
    evaluate_view = plan.view(SplitRole.EVALUATE)
    drop_zscore = match is not None and match.ablate_target_engine_zscore
    if match is not None:
        fit_view = replace(fit_view, frame=match.restrict(fit_view.frame, SplitRole.FIT))
        calibrate_view = replace(
            calibrate_view, frame=match.restrict(calibrate_view.frame, SplitRole.CALIBRATE)
        )

    # The descriptor carries the id too. `plan.descriptor()` reads `plan.fold_id`, and one
    # zero-shot plan is shared by a target engine's three donor pairs -- so the run record
    # held 168 folds under 16 distinct descriptor ids, one descriptor standing for 21 runs.
    # `replace` rather than mutating the plan: SplitPlan is not frozen and the pairs share
    # the object, so an in-place write would corrupt its siblings.
    result = FoldResult(
        fold_id=label,
        method_id=method.method_id,
        descriptor=plan.descriptor().model_copy(update={"fold_id": label}),
    )
    result.diagnostics = {
        "n_fit": len(fit_view),
        "n_calibrate": len(calibrate_view),
        "n_evaluate": len(evaluate_view),
        "harm_policy": policy.value,
        "evidence_config": method.evidence_config,
        "matched_handicap": match is not None,
        "target_engine_zscore_ablated": drop_zscore,
    }
    if evaluate_view.is_empty:
        result.diagnostics["skipped"] = "evaluation view is empty"
        return result

    # 0. Fit the confidence featurizer on the fit scope only — this fold's fit ENGINES
    #    as well as its fit documents. Filtering on documents alone would estimate the
    #    held-out engine's own mean and standard deviation and then standardize that
    #    engine's confidences against them, normalizing away the cross-engine confidence
    #    shift that H1 exists to measure (vectors L2/L3).
    featurizer = ConfidenceFeaturizer()
    featurizer.fit(
        [
            (sample.engine_id, value, sample.conf_scale)
            for candidate_id in fit_view.frame["candidate_id"]
            if (sample := conf_samples.get(str(candidate_id))) is not None
            for value in sample.native_confidences
        ]
    )
    held_out_seen = set(featurizer.engines) & plan.held_out_engines
    if held_out_seen and not plan.include_held_out_in_fitting:
        msg = (
            f"the confidence featurizer was fitted on held-out engine(s) "
            f"{sorted(held_out_seen)}; the zero-shot protocol forbids it (vector L3)"
        )
        raise LeakageError(msg)
    result.diagnostics["featurizer_engines"] = list(featurizer.engines)

    # 1. Fit the verifier on the fit scope only.
    verifier = build_verifier(
        method.kind,
        verifier_id=method.verifier_id,
        evidence_config=method.evidence_config,
        **method.params,
    )
    fit_inputs = _inputs(
        fit_view.frame, bundles, mask, crops, featurizer, conf_samples, drop_zscore
    )
    verifier.fit(fit_inputs, _harmful_flags(fit_view.frame, policy).tolist())

    # 2. Fit the calibrator on the calibration scope only. The event being predicted is
    #    "accepting this edit is safe", matching the verifier's orientation.
    calibration_inputs = _inputs(
        calibrate_view.frame, bundles, mask, crops, featurizer, conf_samples, drop_zscore
    )
    calibration_scores = verifier.score(calibration_inputs).scores
    calibration_safe = (~_harmful_flags(calibrate_view.frame, policy)).astype(np.float64)

    calibrator = build_calibrator(method.calibration_method)
    calibrator.fit(calibration_scores, calibration_safe)
    result.calibrator_id = calibrator.identity()
    calibrated_calibration = calibrator.transform(calibration_scores)

    # 3. Select tau(epsilon) on the calibration scope. Nothing from D_test participates.
    #
    # Threshold selection uses the *natural* candidate pool by default. Adversarial hard
    # negatives are ~99% harmful and can outnumber real candidates several to one, so
    # including them would set the threshold against a population the deployed system
    # never sees, and would report a tolerance met on the wrong distribution. They remain
    # in fitting, and are evaluated as a separate challenge set.
    selection_mask = _selection_mask(calibrate_view.frame, config)
    calibration_harmful = _harmful_flags(calibrate_view.frame, policy)
    result.diagnostics["n_threshold_selection"] = int(selection_mask.sum())
    result.diagnostics["hard_negatives_in_evaluation"] = (
        config.candidates.hard_negatives_in_evaluation
    )

    decisions_by_epsilon: dict[float, float] = {}
    for epsilon in config.risk.epsilon_grid:
        decision = select_threshold(
            calibrated_calibration[selection_mask],
            calibration_harmful[selection_mask],
            epsilon=epsilon,
            delta=config.risk.delta,
            controller=config.risk.controller,
            n_grid=config.risk.n_threshold_grid,
        )
        decisions_by_epsilon[epsilon] = decision.tau
        result.thresholds.append(
            {**decision.as_dict(), "fold_id": label, "method_id": method.method_id}
        )

    # 4-5. Frozen. Score and decide on the held-out engine's test documents.
    evaluate_inputs = _inputs(
        evaluate_view.frame, bundles, mask, crops, featurizer, conf_samples, drop_zscore
    )
    raw = verifier.score(evaluate_inputs)
    calibrated = calibrator.transform(raw.scores)

    # A typed row map, built once: pandas' itertuples() is loosely typed, and casting at
    # each use would scatter the noise across the function.
    by_candidate: dict[str, _EvalRow] = {
        str(row.candidate_id): _EvalRow(
            site_id=str(row.site_id),
            document_id=str(row.document_id),
            dataset_id=str(row.dataset_id),
            engine_id=str(row.engine_id),
            generator_rank=int(getattr(row, "generator_rank", 0)),
        )
        for row in evaluate_view.frame.itertuples()
    }
    for candidate_id, raw_score, calibrated_score in zip(
        raw.candidate_ids, raw.scores, calibrated, strict=True
    ):
        row = by_candidate[candidate_id]
        result.predictions.append(
            Prediction(
                candidate_id=candidate_id,
                site_id=row.site_id,
                document_id=row.document_id,
                dataset_id=row.dataset_id,
                engine_id=row.engine_id,
                verifier_id=method.verifier_id,
                evidence_config=method.evidence_config,
                fold_id=label,
                raw_score=float(raw_score),
                calibrated_score=float(calibrated_score),
                calibrator_id=result.calibrator_id,
            )
        )

    # Site decisions are taken over the SAME pool the threshold was certified on. tau was
    # chosen against the natural candidate distribution; letting adversarial hard
    # negatives compete for acceptance would apply that promise to a population that is
    # ~99% harmful by construction, and the measured risk would be computed on a
    # different distribution from the one the guarantee was made about (vectors L6/L10).
    eval_selection = _selection_mask(evaluate_view.frame, config)
    decidable = set(evaluate_view.frame.loc[eval_selection, "candidate_id"])
    result.diagnostics["n_evaluate_decidable"] = len(decidable)
    result.diagnostics["n_evaluate_excluded_hard_negative"] = int((~eval_selection).sum())

    scored = [
        ScoredCandidate(
            candidate_id=candidate_id,
            site_id=by_candidate[candidate_id].site_id,
            document_id=by_candidate[candidate_id].document_id,
            score=float(score),
            generator_rank=by_candidate[candidate_id].generator_rank,
        )
        for candidate_id, score in zip(raw.candidate_ids, calibrated, strict=True)
        if candidate_id in decidable
    ]

    # A preserved site has no accepted candidate to read ids from, so the dataset and
    # engine are looked up from the SITE. Falling back to row 0 of the fold was correct for
    # engine_id -- a fold is single-engine by construction -- and wrong for dataset_id,
    # because a fold spans three corpora. Since every decision in this pilot was PRESERVE,
    # all 109,305 rows were labelled with whichever corpus happened to sort first.
    site_dataset: dict[str, str] = dict(
        zip(
            evaluate_view.frame["site_id"].astype(str),
            evaluate_view.frame["dataset_id"].astype(str),
            strict=True,
        )
    )
    default_engine_id = str(evaluate_view.frame["engine_id"].iloc[0])

    for epsilon, tau in decisions_by_epsilon.items():
        for site_decision in decide_sites(scored, tau, config.risk.site_policy):
            accepted_row = by_candidate.get(site_decision.accepted_candidate_id or "")
            dataset_id = (
                accepted_row.dataset_id
                if accepted_row
                else site_dataset.get(site_decision.site_id, "")
            )
            engine_id = accepted_row.engine_id if accepted_row else default_engine_id
            result.decisions.append(
                Decision(
                    site_id=site_decision.site_id,
                    document_id=site_decision.document_id,
                    dataset_id=dataset_id,
                    engine_id=engine_id,
                    method_id=method.method_id,
                    fold_id=label,
                    epsilon=epsilon,
                    tau=tau if np.isfinite(tau) else 1.0,
                    action=(
                        DecisionAction.CORRECT
                        if site_decision.accepted
                        else DecisionAction.PRESERVE
                    ),
                    accepted_candidate_id=site_decision.accepted_candidate_id,
                    accepted_score=site_decision.accepted_score,
                    n_candidates_considered=site_decision.n_candidates_considered,
                )
            )

    return result


def methods_from_config(config: ExperimentConfig) -> list[MethodSpec]:
    """Every enabled verifier, paired with the configured calibrator."""
    return [
        MethodSpec(
            verifier_id=spec.id,
            kind=spec.kind,
            evidence_config=spec.evidence_config,
            calibration_method=config.calibration.method,
            params=dict(spec.params),
        )
        for spec in config.verifiers
        if spec.enabled
    ]


def engine_ids(plans: Sequence[SplitPlan]) -> list[str]:
    return sorted({engine for plan in plans for engine in plan.all_engines})
