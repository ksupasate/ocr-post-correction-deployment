#!/usr/bin/env python3
"""SGV3 development phase 1: can the correction system estimate its own decision reliability?

Two negatives set this stage up. Phase 6 (`docs/sgv1/domain_generalized_risk_calibration.md`,
SGV1-DG1) found the harm *ranking* does not survive an unseen engine and that no monotone
recalibration can move the achievable frontier. SGV2 (`docs/sgv2/shift_aware_reliability.md`,
SGV2-R1) then found that detecting the shift and rejecting the anomalous rows does not
recover safety -- and that the association runs the other way: on all four held-out engines
the anomalous rows carried *less* harm and *more* benefit than the typical ones.

So the question is no longer "is this row from a strange engine". It is whether the system
can say how unsure it is about *this decision*, with no engine label anywhere.

**The hypothesis is recorded as SGV3-U1, not H1.** `docs/sgv1/protocol.md` binds SGV1-H1
through SGV1-H4 to preregistered claims and states a frozen ID is never reused; SGV1-H1 is
already "does source-image evidence improve discrimination". SGV2 restarted the family under
its own root for the same reason, and so does this stage.

    SGV3-U1: a self-aware uncertainty model combining aleatoric, epistemic and decision
    uncertainty improves selective OCR correction -- more repairs captured under a bounded
    harmful-accept risk -- over OCR confidence alone, over the harm-aware utility model, and
    over shift-based rejection, on an engine held out of fitting.

Five things decide what the numbers below can mean. Four were settled before measuring.

**1. The endpoint is a property of a ranking, so the uncertainty term earns nothing unless it
reorders.** Repair recall at bounded harm is read off `argsort(-score)`. `U = P(benefit) -
l1*P(harm) - l2*uncertainty` at `l2 = 0` is exactly the harm-aware baseline, which makes
ablation E an identity rather than an experiment -- and `tests/leakage` asserts it as one.
Every arm is compared to the baseline at MATCHED coverage, never at whatever coverage it
happened to land on.

**2. Two of the four calibration methods cannot move the achievable frontier, by
construction.** Temperature and isotonic are monotone in their input, so they leave
`argsort` fixed; Phase 6 proved the point for calibration transfer and it applies unchanged
here. They can still move a *deployed* threshold, which is a different question and is
reported separately. Only ensemble averaging (not a monotone map of any single model's
output) and conformal truncation (which removes rows rather than reordering them) can move
the frontier, and the invariance is measured rather than asserted.

**3. lambda2's sign is free.** SGV2 measured the association between anomaly and harm running
opposite to the assumed direction on all four engines. A grid of non-negative lambda2 could
only ever test the direction this hypothesis assumes, and would report a null where the data
may hold a reversal. The grid is symmetric; negative lambda2 means "prefer the uncertain
rows"; zero means the selection declined to use uncertainty at all.

**4. CORRECT / PRESERVE / ABSTAIN is two-way on text and three-way on review budget.** Both
PRESERVE and ABSTAIN leave `O` in place, so both give `d_after = d_before` and neither is
harmful nor beneficial under `edits/outcome.py`. The benchmark has no human-review channel,
so the third action cannot be scored on text outcomes. What *is* measurable is the
composition of the two sets -- whether ABSTAIN concentrates the repairs a reviewer could
still capture while PRESERVE concentrates the rows that were fine left alone. That is the
claim the three-way split has to earn, and it is the one this stage measures.

**5. No engine identity, anywhere in the method.** SGV2's strongest detector was a supervised
domain classifier over engine labels. This stage's constraint forbids that: every uncertainty
signal is computed from the row plus models frozen on the fit engines, and a behavioural test
permutes the engine column and asserts every signal is bit-identical. SGV2's arm is still run
as a *baseline*, with the detector it selected recorded, because a baseline may use what the
proposed method may not.

    --features     stage 1: the uncertainty feature framework, documented signal by signal
    --scores       fit every fold once; write the row-level table every later stage reads
    --calibration  stage 3: temperature / isotonic / conformal / ensemble, beyond ECE
    --curves       the risk-coverage curves, in-domain and cross-engine
    --actions      stage 4: CORRECT / PRESERVE / ABSTAIN and what the split is worth
    --transfer     stage 5: per-engine evaluation against all four mandatory baselines
    --ablation     ablations A-E: which uncertainty source actually contributes
    --figures      the figures, with the manifest that hashes their inputs
    --decide       the machine-readable finding

DEVELOPMENT ONLY. The CONFIRMATORY reserve stays locked and is absent from every artifact.
Every fold holds out an engine AND holds out documents; neither axis is relaxed anywhere.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sgv1_candidate_conditioned as cc
import sgv1_domain_generalization as dg
import sgv1_risk_policy as policy
import sgv1_verifier_pilot as pilot
import sgv2_reliability_layer as rl
from ocr_risk.calibrate.calibrators import build_calibrator
from ocr_risk.io.hashing import file_sha256
from ocr_risk.metrics.calibration import (
    brier_score,
    expected_calibration_error,
    murphy_decomposition,
)
from ocr_risk.metrics.discrimination import roc_auc
from ocr_risk.metrics.selective import aurc, coverage_at_risk, risk_coverage_curve
from ocr_risk.risk.controller import CONTROLLERS, select_threshold
from ocr_risk.stats.bootstrap import cluster_bootstrap_indices

REPO = pilot.REPO
OUT = REPO / "results/generated/sgv3/self_aware"
FEATURE_SPEC = OUT / "uncertainty_features.json"
CALIBRATION_RESULTS = OUT / "calibration_results.json"
CURVE_RESULTS = OUT / "risk_coverage_curves.json"
ACTION_RESULTS = OUT / "action_selection.json"
TRANSFER_RESULTS = OUT / "engine_transfer_results.json"
ABLATION_RESULTS = OUT / "ablation_results.json"
DECISION = OUT / "research_decision.json"
FIT_RECORD = OUT / "fit_record.json"
SCORES = OUT / "selfaware_scores.parquet"
SELECTION_RECORD = OUT / "selection_record.json"
FIGURE_DIR = OUT / "figures"
FIGURE_MANIFEST = OUT / "figure_manifest.json"

EPSILONS = policy.EPSILONS
PRIMARY_EPSILON = dg.PRIMARY_EPSILON
DELTA = dg.DELTA
BOOTSTRAP_RESAMPLES = dg.BOOTSTRAP_RESAMPLES
ECE_BINS = dg.ECE_BINS
BINNINGS = dg.BINNINGS

# The endpoint has exactly one implementation in this repository and it is Phase 5's. Phase 6
# imported it, SGV2 imported it, and so does this stage; `tests/leakage` asserts all four
# names are the same function object. Two functions called "repair recall" that disagree by a
# rounding convention is how a paper ends up with two headline numbers.
achievable_repair_recall = rl.achievable_repair_recall
harm_at_matched_coverage = rl.harm_at_matched_coverage
harm_at_matched_repair_recall = rl.harm_at_matched_repair_recall
build_fold = rl.build_fold
Fold = rl.Fold

MATCHED_COVERAGES = rl.MATCHED_COVERAGES

# lambda1 trades benefit against harm; the grid is Phase 5's, unchanged, so the harm-aware
# baseline here is Phase 5's baseline and not a re-tuned version of it.
LAMBDA1_GRID = policy.LAMBDAS
# lambda2 weights the uncertainty penalty. SYMMETRIC about zero: see point 3 of the module
# docstring. The composite uncertainty is mapped to a fit percentile in [0, 1], so lambda2 is
# on the same scale as a probability and 1.0 already lets uncertainty dominate the utility.
LAMBDA2_GRID = (-1.0, -0.5, -0.25, -0.10, -0.05, 0.0, 0.05, 0.10, 0.25, 0.50, 1.00)

ENSEMBLE_MEMBERS = 12
ENSEMBLE_ROW_FRACTION = 0.5
ENSEMBLE_FEATURE_FRACTION = 0.75
DROPOUT_DRAWS = 12
DROPOUT_RATE = 0.25
MAHALANOBIS_RIDGE = rl.MAHALANOBIS_RIDGE
CONFORMAL_ALPHAS = (0.05, 0.10, 0.20)
CALIBRATION_METHODS = ("identity", "temperature", "isotonic", "ensemble", "conformal")
MONOTONE_METHODS = ("identity", "temperature", "isotonic")
ABSTAIN_PERCENTILES = (0.80, 0.90, 0.95)
CURVE_POINTS = 200


class PhaseError(RuntimeError):
    """A freeze, role, split, or selection invariant failed."""


# ------------------------------------------------------------------ stage 1: the signals

# Each signal is one row of the framework the brief asks for. `higher_is_more_uncertain` is
# the sign convention every one of them is coerced into before aggregation, so a reader never
# has to remember which of `conf_normalized` and `1 - conf_normalized` a given line meant.
# `source` names where the number physically comes from, because "documented feature source"
# is the requirement and "it is in the design matrix" is not an answer.


@dataclass(frozen=True, slots=True)
class SignalSpec:
    name: str
    block: str
    tags: tuple[str, ...]
    source: str
    definition: str
    rationale: str


SIGNALS: tuple[SignalSpec, ...] = (
    SignalSpec(
        name="alea_ocr_confidence",
        block="aleatoric",
        tags=("confidence",),
        source="engine recognition confidence, `conf_normalized` (design matrix, conf_ family)",
        definition=(
            "1 - conf_normalized, with the fit-row median of conf_normalized substituted "
            "wherever conf_missing marks the engine as reporting no confidence"
        ),
        rationale=(
            "the engine's own statement of how sure it was about the glyphs under the edit; "
            "the median substitution is frozen from fit rows, never recomputed at scoring "
            "time, and the substitution rate is recorded per fold"
        ),
    ),
    SignalSpec(
        name="alea_candidate_ambiguity",
        block="aleatoric",
        tags=(),
        source="frozen candidate provenance, `prov_site_candidate_count`",
        definition="log1p(prov_site_candidate_count)",
        rationale=(
            "how many competing readings the frozen generator proposed at this site; more "
            "competitors is more genuine ambiguity about what the span says"
        ),
    ),
    SignalSpec(
        name="alea_edit_magnitude",
        block="aleatoric",
        tags=(),
        source="the proposed edit, `text_normalized_distance`",
        definition="text_normalized_distance (Levenshtein O->Y over max length)",
        rationale=(
            "a larger proposed change is a larger leap away from what the engine actually "
            "read, and carries more of the edit's outcome on the model's judgement alone"
        ),
    ),
    SignalSpec(
        name="alea_language_ambiguity",
        block="aleatoric",
        tags=("language",),
        source="TRAIN-fitted character n-gram LM, `plaus_lm_delta`",
        definition="-|plaus_lm_delta|",
        rationale=(
            "the LM separates O from Y only when |delta| is large; a delta near zero says "
            "the language model cannot tell which reading is the plausible one"
        ),
    ),
    SignalSpec(
        name="alea_visual_ambiguity",
        block="aleatoric",
        tags=("visual",),
        source="page crop glyph template match, `vis_glyph_gain`",
        definition="-|vis_glyph_gain|",
        rationale=(
            "the same argument on the image channel: a glyph-match gain near zero means the "
            "pixels do not favour either reading, which is aleatoric and not fixable by more "
            "training data"
        ),
    ),
    SignalSpec(
        name="epis_ensemble_std",
        block="epistemic",
        tags=("disagreement",),
        source=(
            "12 bagged harm models, each on a seeded 50% row resample and a 75% feature "
            "subset of the fold's fit rows"
        ),
        definition="standard deviation across members of P(harmful)",
        rationale=(
            "disagreement among models that saw different halves of the same fit data is "
            "uncertainty about the model rather than about the span, and it is the signal "
            "SGV2 did not test: SGV2's detectors all asked whether the row was unusual, none "
            "asked whether the fitted decision boundary was settled there"
        ),
    ),
    SignalSpec(
        name="epis_feature_distance",
        block="epistemic",
        tags=(),
        source="Mahalanobis distance to the fit-row centroid in scaled feature space",
        definition="sqrt((x - mu)' S^-1 (x - mu)) on fit-fitted mu and ridged S",
        rationale=(
            "the classical epistemic proxy, and deliberately the SAME quantity SGV2 measured "
            "as its `mahalanobis` cold-start detector. Including it means the ablation that "
            "removes the epistemic block re-tests SGV2's finding inside this stage's "
            "protocol rather than citing it"
        ),
    ),
    SignalSpec(
        name="epis_instability",
        block="epistemic",
        tags=("disagreement",),
        source="12 seeded feature-dropout masks applied to the single fit harm model",
        definition="standard deviation of P(harmful) over masks that zero 25% of features",
        rationale=(
            "how much the prediction depends on any particular part of the representation; a "
            "decision that survives dropping a quarter of the evidence is a decision the "
            "model is not balancing on one feature"
        ),
    ),
    SignalSpec(
        name="deci_harm_entropy",
        block="decision",
        tags=(),
        source="the calibrated binary harm head",
        definition="binary Shannon entropy of P(harmful), in nats",
        rationale=(
            "peaks at P = 0.5, so it is NOT monotone in the score and can genuinely reorder "
            "a ranking -- unlike a temperature or isotonic recalibration of the same "
            "probability, which cannot"
        ),
    ),
    SignalSpec(
        name="deci_action_margin",
        block="decision",
        tags=(),
        source="the three-class outcome head (harm / neutral / benefit)",
        definition="1 - |P(beneficial) - P(harmful)|",
        rationale=(
            "the margin between the two actions the utility is choosing between; a small "
            "margin means the decision, not the probability, is the uncertain part"
        ),
    ),
    SignalSpec(
        name="deci_class_entropy",
        block="decision",
        tags=(),
        source="the three-class outcome head",
        definition="Shannon entropy of (P(harm), P(neutral), P(benefit)) divided by log 3",
        rationale=(
            "the margin ignores the neutral mass; the entropy does not, and a row the model "
            "thinks is probably a no-op is a different kind of unsure"
        ),
    ),
)

SIGNAL_NAMES = tuple(s.name for s in SIGNALS)

# The ablation variants. Each is a SUBSET of the signal set, aggregated and percentile-mapped
# exactly as the full set is, and each gets its own (lambda1, lambda2) chosen by the same
# inner leave-one-engine-out. Comparing variants at a shared hyperparameter would confound
# "this block carries no signal" with "this block needed a different weight".
VARIANTS: dict[str, tuple[str, ...]] = {
    "full": SIGNAL_NAMES,
    "no_epistemic": tuple(s.name for s in SIGNALS if s.block != "epistemic"),
    "no_aleatoric": tuple(s.name for s in SIGNALS if s.block != "aleatoric"),
    "no_visual": tuple(s.name for s in SIGNALS if "visual" not in s.tags),
    "no_disagreement": tuple(s.name for s in SIGNALS if "disagreement" not in s.tags),
    "confidence_only": tuple(s.name for s in SIGNALS if "confidence" in s.tags),
    "model_uncertainty_only": tuple(s.name for s in SIGNALS if s.block == "epistemic"),
}
HEADLINE_VARIANT = "full"

# Which brief-named ablation each variant answers, kept next to the definition so the
# artifact and the document cannot drift apart on what "ablation C" meant.
ABLATION_LABELS = {
    "no_epistemic": "A - remove epistemic uncertainty",
    "confidence_only": "A (literal) - confidence features only",
    "no_aleatoric": "B - remove aleatoric uncertainty",
    "model_uncertainty_only": "B (literal) - model uncertainty only",
    "no_visual": "C - remove visual uncertainty",
    "no_disagreement": "D - remove disagreement features",
    "no_penalty": "E - remove the uncertainty penalty (lambda2 = 0)",
}


# ------------------------------------------------------------------ the fitted machinery


def _feature_index(design: dg.Design, name: str) -> int:
    try:
        return design.names.index(name)
    except ValueError as error:
        raise PhaseError(f"the design matrix has no feature {name!r}") from error


def _binary_entropy(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, 1e-12, 1.0 - 1e-12)
    return -(q * np.log(q) + (1.0 - q) * np.log1p(-q))


@dataclass(slots=True)
class Heads:
    """The harm head, the three-outcome head, their calibrators, and the ensemble.

    Everything here is fitted on ``fold.fit`` except the two calibrators, which are fitted on
    ``fold.source_cal`` -- fit engines, CALIBRATION documents. No evaluation row and no
    held-out engine row is seen by any of it, which `build_fold` has already guaranteed by
    the time this is called.
    """

    scaler: Any
    binary: Any
    three: Any
    harm_calibrator: Any
    benefit_calibrator: Any
    columns: tuple[int, ...]
    ensemble: tuple[tuple[tuple[int, ...], Any], ...]
    dropout_masks: np.ndarray

    def _scaled(self, design: dg.Design, index: np.ndarray) -> np.ndarray:
        return np.asarray(
            self.scaler.transform(design.matrix[np.ix_(index, list(self.columns))]), dtype=float
        )

    @staticmethod
    def _column(model: Any, scaled: np.ndarray, klass: int) -> np.ndarray:
        classes = list(model.classes_)
        if klass not in classes:
            return np.zeros(len(scaled))
        return np.asarray(model.predict_proba(scaled)[:, classes.index(klass)], dtype=float)

    def raw_probabilities(self, design: dg.Design, index: np.ndarray) -> dict[str, np.ndarray]:
        """Uncalibrated head outputs. Separate from `probabilities` because the calibrators
        are fitted FROM this, and a single method would have to be called before it is
        fitted -- which `Calibrator` refuses, correctly."""
        scaled = self._scaled(design, index)
        three = np.column_stack(
            [
                self._column(self.three, scaled, policy.CLASS_HARM),
                self._column(self.three, scaled, policy.CLASS_NEUTRAL),
                self._column(self.three, scaled, policy.CLASS_BENEFIT),
            ]
        )
        return {
            "scaled": scaled,
            "harm_raw": self._column(self.binary, scaled, 1),
            "benefit_raw": three[:, 2],
            "three": three,
        }

    def probabilities(self, design: dg.Design, index: np.ndarray) -> dict[str, np.ndarray]:
        """Calibrated P(harm) and P(benefit), plus the raw three-way distribution."""
        out = self.raw_probabilities(design, index)
        out["harm"] = np.asarray(self.harm_calibrator.transform(out["harm_raw"]), dtype=float)
        out["benefit"] = np.asarray(
            self.benefit_calibrator.transform(out["benefit_raw"]), dtype=float
        )
        return out

    def ensemble_spread(self, scaled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Mean and standard deviation of P(harm) across the bagged members."""
        stack = np.vstack(
            [self._column(model, scaled[:, list(cols)], 1) for cols, model in self.ensemble]
        )
        return stack.mean(axis=0), stack.std(axis=0)

    def dropout_spread(self, scaled: np.ndarray) -> np.ndarray:
        """Standard deviation of P(harm) when a quarter of the evidence is removed.

        Zeroing a *scaled* column substitutes the fit-row mean of that feature, which is the
        right sense of "remove this evidence" -- it is what the model would have seen from an
        average row rather than an arbitrary constant.
        """
        stack = np.vstack(
            [self._column(self.binary, scaled * mask, 1) for mask in self.dropout_masks]
        )
        return stack.std(axis=0)


def fit_heads(design: dg.Design, fold: Fold, columns: list[int]) -> Heads:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    block = design.matrix[np.ix_(fold.fit, columns)]
    scaler = StandardScaler().fit(block)
    scaled = np.asarray(scaler.transform(block), dtype=float)
    harmful = design.harmful[fold.fit].astype(int)
    outcome = np.full(fold.fit.size, policy.CLASS_NEUTRAL, dtype=np.int64)
    outcome[design.harmful[fold.fit]] = policy.CLASS_HARM
    outcome[design.beneficial[fold.fit]] = policy.CLASS_BENEFIT

    def logistic() -> Any:
        return LogisticRegression(
            C=1.0, max_iter=2000, class_weight="balanced", random_state=pilot.FIT_SEED
        )

    binary = logistic().fit(scaled, harmful)
    three = logistic().fit(scaled, outcome)

    rng = np.random.default_rng(pilot.FIT_SEED)
    n_rows = round(ENSEMBLE_ROW_FRACTION * len(scaled))
    n_cols = round(ENSEMBLE_FEATURE_FRACTION * len(columns))
    members: list[tuple[tuple[int, ...], Any]] = []
    for _ in range(ENSEMBLE_MEMBERS):
        rows = rng.choice(len(scaled), size=n_rows, replace=False)
        cols = np.sort(rng.choice(len(columns), size=n_cols, replace=False))
        members.append(
            (tuple(int(c) for c in cols), logistic().fit(scaled[np.ix_(rows, cols)], harmful[rows]))
        )

    masks = np.ones((DROPOUT_DRAWS, len(columns)), dtype=float)
    for draw in range(DROPOUT_DRAWS):
        dropped = rng.choice(len(columns), size=round(DROPOUT_RATE * len(columns)), replace=False)
        masks[draw, dropped] = 0.0

    heads = Heads(
        scaler=scaler,
        binary=binary,
        three=three,
        harm_calibrator=None,
        benefit_calibrator=None,
        columns=tuple(columns),
        ensemble=tuple(members),
        dropout_masks=masks,
    )
    calibration = heads.raw_probabilities(design, fold.source_cal)
    harm_calibrator = build_calibrator(pilot.CALIBRATION_METHOD)
    harm_calibrator.fit(calibration["harm_raw"], design.harmful[fold.source_cal].astype(float))
    benefit_calibrator = build_calibrator(pilot.CALIBRATION_METHOD)
    benefit_calibrator.fit(
        calibration["benefit_raw"], design.beneficial[fold.source_cal].astype(float)
    )
    heads.harm_calibrator = harm_calibrator
    heads.benefit_calibrator = benefit_calibrator
    return heads


@dataclass(slots=True)
class UncertaintyModel:
    """Every signal, its frozen percentile map, and the composite for each ablation variant.

    Percentiles rather than z-scores, for the reason SGV2 used percentiles: an ensemble
    standard deviation and a Mahalanobis distance are not on comparable axes, and a mean of
    z-scores over skewed signals is a mean of whichever signal has the longest tail. Each
    signal is mapped to where it falls in the CALIBRATION distribution first, so every signal
    contributes a uniform [0, 1] and the composite weights them equally by construction.
    """

    heads: Heads
    centre: np.ndarray
    inverse: np.ndarray
    conf_median: float
    conf_index: int
    plain: dict[str, int]
    signal_reference: dict[str, np.ndarray]
    composite_reference: dict[str, np.ndarray]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def probabilities(self, design: dg.Design, index: np.ndarray) -> dict[str, np.ndarray]:
        return self.heads.probabilities(design, index)

    def raw_signals(self, design: dg.Design, index: np.ndarray) -> dict[str, np.ndarray]:
        """Every signal on its own scale, already oriented so higher means more uncertain."""
        probabilities = self.heads.probabilities(design, index)
        scaled = probabilities["scaled"]
        matrix = design.matrix
        conf = matrix[index, self.conf_index].copy()
        missing = matrix[index, self.plain["conf_missing"]] > 0.5
        conf[missing] = self.conf_median
        delta = scaled - self.centre
        _, ensemble_std = self.heads.ensemble_spread(scaled)
        three = np.clip(probabilities["three"], 1e-12, 1.0)
        three = three / three.sum(axis=1, keepdims=True)
        return {
            "alea_ocr_confidence": 1.0 - conf,
            "alea_candidate_ambiguity": np.log1p(
                matrix[index, self.plain["prov_site_candidate_count"]]
            ),
            "alea_edit_magnitude": matrix[index, self.plain["text_normalized_distance"]],
            "alea_language_ambiguity": -np.abs(matrix[index, self.plain["plaus_lm_delta"]]),
            "alea_visual_ambiguity": -np.abs(matrix[index, self.plain["vis_glyph_gain"]]),
            "epis_ensemble_std": ensemble_std,
            "epis_feature_distance": np.sqrt(
                np.maximum(np.einsum("ij,jk,ik->i", delta, self.inverse, delta), 0.0)
            ),
            "epis_instability": self.heads.dropout_spread(scaled),
            "deci_harm_entropy": _binary_entropy(probabilities["harm"]),
            "deci_action_margin": 1.0 - np.abs(probabilities["benefit"] - probabilities["harm"]),
            "deci_class_entropy": -(three * np.log(three)).sum(axis=1) / float(np.log(3.0)),
        }

    @staticmethod
    def _percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
        return np.searchsorted(reference, values, side="right") / float(reference.size)

    def signal_percentiles(self, design: dg.Design, index: np.ndarray) -> dict[str, np.ndarray]:
        raw = self.raw_signals(design, index)
        return {name: self._percentile(self.signal_reference[name], raw[name]) for name in raw}

    def composites(self, design: dg.Design, index: np.ndarray) -> dict[str, np.ndarray]:
        """One uncertainty score per ablation variant, each a fit percentile in [0, 1]."""
        percentiles = self.signal_percentiles(design, index)
        out = {}
        for variant, members in VARIANTS.items():
            mean = np.mean([percentiles[name] for name in members], axis=0)
            out[variant] = self._percentile(self.composite_reference[variant], mean)
        return out


def fit_uncertainty(design: dg.Design, fold: Fold, columns: list[int]) -> UncertaintyModel:
    """Fit the heads on ``fold.fit`` and freeze every percentile map on ``fold.source_cal``."""
    heads = fit_heads(design, fold, columns)
    scaled_fit = heads._scaled(design, fold.fit)
    centre = scaled_fit.mean(axis=0)
    covariance = np.cov(scaled_fit - centre, rowvar=False) + MAHALANOBIS_RIDGE * np.eye(
        len(columns)
    )
    conf_index = _feature_index(design, "conf_normalized")
    missing_fit = design.matrix[fold.fit, _feature_index(design, "conf_missing")] > 0.5
    present = design.matrix[fold.fit, conf_index][~missing_fit]
    model = UncertaintyModel(
        heads=heads,
        centre=centre,
        inverse=np.linalg.pinv(covariance),
        conf_median=float(np.median(present)) if present.size else 0.0,
        conf_index=conf_index,
        plain={
            name: _feature_index(design, name)
            for name in (
                "conf_missing",
                "prov_site_candidate_count",
                "text_normalized_distance",
                "plaus_lm_delta",
                "vis_glyph_gain",
            )
        },
        signal_reference={name: np.zeros(1) for name in SIGNAL_NAMES},
        composite_reference={variant: np.zeros(1) for variant in VARIANTS},
    )
    reference_raw = model.raw_signals(design, fold.source_cal)
    model.signal_reference = {name: np.sort(values) for name, values in reference_raw.items()}
    percentiles = model.signal_percentiles(design, fold.source_cal)
    model.composite_reference = {
        variant: np.sort(np.mean([percentiles[name] for name in members], axis=0))
        for variant, members in VARIANTS.items()
    }
    model.diagnostics = {
        "confidence_missing_fraction_fit": float(missing_fit.mean()),
        "confidence_median_substituted": model.conf_median,
        "ensemble_members": ENSEMBLE_MEMBERS,
        "dropout_draws": DROPOUT_DRAWS,
        "reference_rows": int(fold.source_cal.size),
        "reference_note": (
            "the percentile maps are fitted transformers frozen on the fit engines' "
            "CALIBRATION documents. The two decision signals read the CALIBRATED head "
            "probabilities, and those calibrators were themselves fitted on these same rows, "
            "so the reference is in-sample for those two signals. It is a monotone map to a "
            "percentile and touches no evaluation row and no label of the held-out engine; "
            "it is recorded here rather than hidden because it makes the reference slightly "
            "optimistic about how spread those two signals are"
        ),
    }
    return model


# ------------------------------------------------------------------ conformal prediction

# Split-conformal in its PREDICTION-SET form, which is the only one of the four calibration
# methods that maps onto the brief's third action. For each row it asks how plausible each of
# the two labels is against the calibration distribution of that label. A row whose set holds
# both labels is one the model cannot resolve; a row whose set is empty is one that fits
# neither calibration population. Both are "cannot estimate reliability", which is ABSTAIN --
# and neither is a threshold on the score, so this is a truncation, not a reordering.


@dataclass(slots=True)
class Conformal:
    """Calibration nonconformity distributions for the two labels, frozen."""

    harmful_scores: np.ndarray
    safe_scores: np.ndarray

    @staticmethod
    def _p_value(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
        """Fraction of the calibration class at least as nonconforming, with the +1 term."""
        at_least = reference.size - np.searchsorted(reference, values, side="left")
        return (at_least + 1.0) / (reference.size + 1.0)

    def sets(self, harm_probability: np.ndarray, alpha: float) -> dict[str, np.ndarray]:
        p_harm = self._p_value(self.harmful_scores, 1.0 - harm_probability)
        p_safe = self._p_value(self.safe_scores, harm_probability)
        in_harm = p_harm > alpha
        in_safe = p_safe > alpha
        return {
            "p_harm": p_harm,
            "p_safe": p_safe,
            "ambiguous": in_harm & in_safe,
            "empty": ~in_harm & ~in_safe,
            "singleton_safe": in_safe & ~in_harm,
            "singleton_harm": in_harm & ~in_safe,
        }

    def resolvable(self, harm_probability: np.ndarray, alpha: float) -> np.ndarray:
        sets = self.sets(harm_probability, alpha)
        return ~(sets["ambiguous"] | sets["empty"])


def fit_conformal(harm_probability: np.ndarray, harmful: np.ndarray) -> Conformal:
    return Conformal(
        harmful_scores=np.sort(1.0 - harm_probability[harmful]),
        safe_scores=np.sort(harm_probability[~harmful]),
    )


# ------------------------------------------------------------------ selection, kept inside


@dataclass(slots=True)
class Selection:
    """(lambda1, lambda2) per variant, plus the SGV2 baseline, all chosen without the
    held-out engine."""

    lambdas: dict[str, tuple[float, float]]
    shift: rl.Selection
    shift_cold: tuple[str, float]
    inner: dict[str, Any]


def _cold_start_from_inner(shift: rl.Selection) -> tuple[str, float]:
    """SGV2's arm restricted to detectors that use no engine label.

    SGV2's inner selection could pick `domain_classifier`, which is fitted against engine
    identity. A baseline is allowed to use what the proposed method may not -- but a
    comparison in which only the baseline may see engine labels answers a different question,
    so the label-free restriction is run beside it. It is re-derived from SGV2's own recorded
    inner scores rather than refitted, so the two baselines differ in exactly one respect.
    """
    means = shift.inner["mean_repair_recall_at_primary_epsilon"]
    cold = {k: v for k, v in means.items() if k.split("|kappa_")[0] in rl.COLD_START_DETECTORS}
    if not cold:
        raise PhaseError("no cold-start detector survived SGV2's inner grid")
    best = max(cold, key=lambda k: (cold[k], -abs(float(k.split("|kappa_")[1])), k))
    kind, kappa = best.split("|kappa_")
    return kind, float(kappa)


def select_lambdas(design: dg.Design, outer: Fold, columns: list[int]) -> Selection:
    """Pick (lambda1, lambda2) per variant by an inner leave-one-engine-out over fit engines.

    The inner folds are SGV2's, imported rather than rebuilt: fit engines only, CALIBRATION
    documents, and the CALIBRATION documents split in half so the frozen percentile reference
    and the inner evaluation rows never share a page. The outer held-out engine appears in no
    block of any inner fold, which `_inner_fold` raises on and `tests/leakage` re-checks.

    lambda2 = 0 is on the grid on purpose. A selection that cannot decline to use uncertainty
    is not a selection, and the resulting arm would be the harm-aware baseline wearing a new
    name.
    """
    key = f"epsilon_{int(PRIMARY_EPSILON * 100)}"
    grid: dict[str, list[float]] = {
        f"{variant}|l1_{l1:g}|l2_{l2:g}": []
        for variant in VARIANTS
        for l1 in LAMBDA1_GRID
        for l2 in LAMBDA2_GRID
    }
    for inner_held in outer.train_engines:
        fold = rl._inner_fold(design, outer, inner_held)
        model = fit_uncertainty(design, fold, columns)
        probabilities = model.probabilities(design, fold.eval)
        composites = model.composites(design, fold.eval)
        harmful = design.harmful[fold.eval]
        beneficial = design.beneficial[fold.eval]
        for variant in VARIANTS:
            uncertainty = composites[variant]
            for l1 in LAMBDA1_GRID:
                utility = probabilities["benefit"] - l1 * probabilities["harm"]
                for l2 in LAMBDA2_GRID:
                    measured = achievable_repair_recall(
                        utility - l2 * uncertainty, harmful, beneficial
                    )
                    grid[f"{variant}|l1_{l1:g}|l2_{l2:g}"].append(
                        float(measured[key]["repair_recall"])
                    )

    means = {name: float(np.mean(values)) for name, values in grid.items()}

    def best_for(variant: str, l2_values: tuple[float, ...]) -> tuple[float, float]:
        names = [(l1, l2) for l1 in LAMBDA1_GRID for l2 in l2_values]
        return max(
            names,
            key=lambda pair: (
                means[f"{variant}|l1_{pair[0]:g}|l2_{pair[1]:g}"],
                -abs(pair[1]),
                -pair[0],
            ),
        )

    lambdas = {variant: best_for(variant, LAMBDA2_GRID) for variant in VARIANTS}
    # The harm-aware baseline is the lambda2 = 0 slice of the same grid, selected by the same
    # rule on the same inner folds. Anything else would compare the proposed method against a
    # baseline that had a different amount of tuning spent on it.
    lambdas["harm_aware"] = best_for(HEADLINE_VARIANT, (0.0,))
    shift = rl.select_policy(design, outer)
    return Selection(
        lambdas=lambdas,
        shift=shift,
        shift_cold=_cold_start_from_inner(shift),
        inner={
            "inner_engines": list(outer.train_engines),
            "inner_evaluation_role": "CALIBRATION",
            "lambda1_grid": list(LAMBDA1_GRID),
            "lambda2_grid": list(LAMBDA2_GRID),
            "variants": {v: list(members) for v, members in VARIANTS.items()},
            "mean_repair_recall_at_primary_epsilon": means,
            "per_inner_fold": grid,
            "selected": {
                name: {"lambda1": pair[0], "lambda2": pair[1]} for name, pair in lambdas.items()
            },
            "lambda2_at_grid_boundary": {
                name: bool(pair[1] in (LAMBDA2_GRID[0], LAMBDA2_GRID[-1]))
                for name, pair in lambdas.items()
            },
            "tie_break": "highest inner mean, then smallest |lambda2|, then smallest lambda1",
            "shift_baseline": {
                "sgv2_selected_detector": shift.detector,
                "sgv2_selected_kappa": shift.kappa,
                "cold_start_restricted": {
                    "detector": _cold_start_from_inner(shift)[0],
                    "kappa": _cold_start_from_inner(shift)[1],
                },
            },
        },
    )


# ------------------------------------------------------------------ the acceptance rules

# Every rule is a score vector on the same rows, with rejection encoded as -inf. Expressing
# abstention as a score rather than as a separate code path is what lets an arm that rejects
# rows and an arm that does not be pushed through identical ranking, threshold and endpoint
# code -- SGV2's construction, kept because the alternative is two comparison paths that can
# disagree for reasons that have nothing to do with the science.

BASELINE_ARMS = ("no_correction", "confidence_only", "harm_only", "harm_aware", "shift_aware")
PROPOSED_ARM = "selfaware"
CEILING_ARMS = ("oracle_selfaware",)


@dataclass(slots=True)
class FoldModel:
    """Everything one fold fits, so cross-engine and in-domain rows go through one object."""

    held_out: str
    fold: Fold
    columns: tuple[int, ...]
    uncertainty: UncertaintyModel
    selection: Selection
    conformal: Conformal
    ensemble_calibrator: Any
    detectors: dict[str, Any]
    diagnostics: dict[str, Any]

    def arms(self, design: dg.Design, index: np.ndarray) -> dict[str, np.ndarray]:
        model = self.uncertainty
        probabilities = model.probabilities(design, index)
        composites = model.composites(design, index)
        harm = probabilities["harm"]
        benefit = probabilities["benefit"]
        safety = 1.0 - probabilities["harm_raw"]
        conf = design.matrix[index, model.conf_index].copy()
        conf[design.matrix[index, model.plain["conf_missing"]] > 0.5] = model.conf_median

        def utility(l1: float, l2: float, variant: str) -> np.ndarray:
            return benefit - l1 * harm - l2 * composites[variant]

        scores: dict[str, np.ndarray] = {
            # R0: the do-nothing reference. Rejecting every row is the honest encoding -- it
            # gives coverage 0, repair recall 0 and an undefined selective harm rate, which is
            # exactly what "apply no correction" delivers.
            "no_correction": np.full(index.size, -np.inf),
            "confidence_only": conf,
            "harm_only": safety,
        }
        for name in ("harm_aware", *VARIANTS):
            l1, l2 = self.selection.lambdas[name]
            variant = HEADLINE_VARIANT if name == "harm_aware" else name
            key = "harm_aware" if name == "harm_aware" else f"selfaware__{name}"
            scores[key] = utility(l1, l2, variant)
        scores[PROPOSED_ARM] = scores[f"selfaware__{HEADLINE_VARIANT}"]
        # Ablation E is not a separate model: removing the uncertainty penalty leaves the
        # harm-aware baseline exactly. Aliasing it rather than recomputing it is what makes
        # `tests/leakage` able to assert the identity bit-for-bit.
        scores["ablate_penalty"] = scores["harm_aware"]

        for name, (kind, kappa) in (
            ("shift_aware", (self.selection.shift.detector, self.selection.shift.kappa)),
            ("shift_aware_cold", self.selection.shift_cold),
        ):
            batch = rl._batch_rank(self.detectors[kind].raw(design, index))
            scores[name] = safety - kappa * batch

        l1, l2 = self.selection.lambdas[HEADLINE_VARIANT]
        ensemble_mean, _ = self.uncertainty.heads.ensemble_spread(probabilities["scaled"])
        scores["selfaware_ensemble"] = (
            benefit
            - l1 * np.asarray(self.ensemble_calibrator.transform(ensemble_mean), dtype=float)
            - l2 * composites[HEADLINE_VARIANT]
        )
        scores["selfaware_conformal"] = np.where(
            self.conformal.resolvable(harm, PRIMARY_EPSILON), scores["harm_aware"], -np.inf
        )
        return scores

    def oracle(self, design: dg.Design, index: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """The ceiling: (lambda1, lambda2) chosen against the held-out engine's own labels.

        Never an arm an operator could run. It exists to separate "the inner selection failed"
        from "the signal is not there", which are different findings and would otherwise be
        reported as the same null.
        """
        probabilities = self.uncertainty.probabilities(design, index)
        uncertainty = self.uncertainty.composites(design, index)[HEADLINE_VARIANT]
        harmful, beneficial = design.harmful[index], design.beneficial[index]
        key = f"epsilon_{int(PRIMARY_EPSILON * 100)}"
        grid: dict[str, float] = {}
        for l1 in LAMBDA1_GRID:
            utility = probabilities["benefit"] - l1 * probabilities["harm"]
            for l2 in LAMBDA2_GRID:
                measured = achievable_repair_recall(utility - l2 * uncertainty, harmful, beneficial)
                grid[f"l1_{l1:g}|l2_{l2:g}"] = float(measured[key]["repair_recall"])
        best = max(grid, key=lambda name: grid[name])
        l1 = float(best.split("|")[0][len("l1_") :])
        l2 = float(best.split("|")[1][len("l2_") :])
        score = probabilities["benefit"] - l1 * probabilities["harm"] - l2 * uncertainty
        return score, {
            "note": (
                "chosen on the held-out engine's own labels; a ceiling on what perfect "
                "selection of (lambda1, lambda2) could deliver, never a deployable arm"
            ),
            "selected": best,
            "selected_repair_recall": grid[best],
            "deployable_selection_repair_recall": grid[
                "l1_{:g}|l2_{:g}".format(*self.selection.lambdas[HEADLINE_VARIANT])
            ],
            "grid": grid,
        }


def fit_fold(design: dg.Design, held_out: str) -> FoldModel:
    columns = design.columns(tuple(dg.FAMILIES))
    fold = build_fold(design, held_out)
    selection = select_lambdas(design, fold, columns)
    model = fit_uncertainty(design, fold, columns)
    calibration = model.probabilities(design, fold.source_cal)
    conformal = fit_conformal(calibration["harm"], design.harmful[fold.source_cal])
    ensemble_calibrator = build_calibrator(pilot.CALIBRATION_METHOD)
    ensemble_mean, _ = model.heads.ensemble_spread(calibration["scaled"])
    ensemble_calibrator.fit(ensemble_mean, design.harmful[fold.source_cal].astype(float))
    detectors = {
        kind: rl.fit_shift_detector(design, fold, kind)
        for kind in {selection.shift.detector, selection.shift_cold[0]}
    }
    fitted = FoldModel(
        held_out=held_out,
        fold=fold,
        columns=tuple(columns),
        uncertainty=model,
        selection=selection,
        conformal=conformal,
        ensemble_calibrator=ensemble_calibrator,
        detectors=detectors,
        diagnostics={},
    )
    evaluation = model.probabilities(design, fold.eval)
    composite = model.composites(design, fold.eval)[HEADLINE_VARIANT]
    utility = evaluation["benefit"] - selection.lambdas[HEADLINE_VARIANT][0] * evaluation["harm"]
    fitted.diagnostics = {
        "uncertainty": model.diagnostics,
        "composite_mean_evaluation": float(composite.mean()),
        "composite_sd_evaluation": float(composite.std()),
        "composite_saturation_fraction": float(np.mean(composite >= rl.SATURATION_THRESHOLD)),
        "spearman_uncertainty_vs_utility": rl._spearman(composite, utility),
        "collapse_note": (
            "The composite is built from unsupervised aggregation of percentile-mapped "
            "signals, but three of its eleven members are functions of the same head "
            "probabilities the utility is built from. A rank correlation near -1 or +1 would "
            "mean the penalty is a monotone re-expression of the utility and cannot reorder "
            "it; the measured value is reported per fold rather than assumed away."
        ),
        "conformal_calibration_rows": {
            "harmful": int(conformal.harmful_scores.size),
            "safe": int(conformal.safe_scores.size),
        },
    }
    return fitted


# ------------------------------------------------------------------ stage 1: the artifact


def _distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "sd": float(values.std()),
        "p05": float(np.percentile(values, 5)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
    }


def run_features() -> int:
    """The uncertainty feature framework, documented signal by signal and measured per fold.

    This is the stage-1 deliverable and it is deliberately descriptive: what each signal is,
    where it comes from, how it is oriented and aggregated, and what it actually looks like
    on fit, calibration and held-out rows. Whether any of it *helps* is stages 3-5; a signal
    table that reported only the signals that turned out to work would be a table assembled
    after the answer.
    """
    started = time.monotonic()
    design = dg.load_design()
    columns = design.columns(tuple(dg.FAMILIES))
    folds: dict[str, Any] = {}
    for held_out in design.engines:
        fold = build_fold(design, held_out)
        model = fit_uncertainty(design, fold, columns)
        raw_eval = model.raw_signals(design, fold.eval)
        again = model.raw_signals(design, fold.eval)
        if not all(np.array_equal(raw_eval[name], again[name]) for name in raw_eval):
            raise PhaseError(f"{held_out}: a signal is not deterministic across two calls")
        percentiles = model.signal_percentiles(design, fold.eval)
        harmful = design.harmful[fold.eval]
        beneficial = design.beneficial[fold.eval]
        # How much of a signal is just the harm score wearing a different name. Three of the
        # eleven are functions of the same head probabilities the utility is built from, and a
        # signal whose rank correlation with P(harm) is near -1 cannot be independent evidence
        # about anything: its association with the harm LABEL is inherited, not measured.
        harm_probability = model.probabilities(design, fold.eval)["harm"]
        stack = np.column_stack([percentiles[name] for name in SIGNAL_NAMES])
        folds[held_out] = {
            "held_out_engine": held_out,
            "evaluation_rows": int(fold.eval.size),
            "signals": {
                name: {
                    "fit": _distribution(model.raw_signals(design, fold.fit)[name]),
                    "calibration_reference": _distribution(model.signal_reference[name]),
                    "held_out": _distribution(raw_eval[name]),
                    "held_out_percentile": _distribution(percentiles[name]),
                    # Document-clustered, because candidate rows inside one page share a
                    # scan, a font and usually a systematic engine failure; a row-level
                    # interval here would be dishonestly narrow.
                    "spearman_vs_harmful": rl._spearman_interval(
                        percentiles[name], harmful, design.documents[fold.eval]
                    ),
                    "auroc_vs_harmful": float(roc_auc(percentiles[name], harmful.astype(float))),
                    "auroc_vs_beneficial": float(
                        roc_auc(percentiles[name], beneficial.astype(float))
                    ),
                    "spearman_vs_harm_probability": rl._spearman(
                        percentiles[name], harm_probability
                    ),
                }
                for name in SIGNAL_NAMES
            },
            "signal_rank_correlation": {
                left: {
                    right: rl._spearman(percentiles[left], percentiles[right])
                    for right in SIGNAL_NAMES
                }
                for left in SIGNAL_NAMES
            },
            "composites": {
                variant: _distribution(values)
                for variant, values in model.composites(design, fold.eval).items()
            },
            "mean_absolute_off_diagonal_rank_correlation": float(
                np.abs(
                    np.corrcoef(
                        np.apply_along_axis(
                            lambda column: pd.Series(column).rank().to_numpy(), 0, stack
                        ),
                        rowvar=False,
                    )[~np.eye(len(SIGNAL_NAMES), dtype=bool)]
                ).mean()
            ),
            "diagnostics": model.diagnostics,
        }
        print(f"  fold {held_out:11s} {len(SIGNAL_NAMES)} signals, {fold.eval.size} held-out rows")

    cc._write_json_once(
        FEATURE_SPEC,
        {
            "schema_version": "sgv3-uncertainty-features-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV3-U1",
            "framework": {
                "orientation": "every signal is oriented so that a larger value is more uncertain",
                "aggregation": (
                    "each signal is mapped to its percentile in the fit engines' CALIBRATION "
                    "distribution, the percentiles are averaged with equal weight, and the "
                    "average is mapped to a percentile again so that lambda2 has one meaning "
                    "across variants"
                ),
                "why_percentiles_not_z_scores": (
                    "an ensemble standard deviation and a Mahalanobis distance are not on "
                    "comparable axes, and a mean of z-scores over skewed signals is a mean of "
                    "whichever signal has the longest tail"
                ),
                "redundancy_note": (
                    "`spearman_vs_harm_probability` says how much each signal is a "
                    "re-expression of the harm score itself. A signal near -1 or +1 there "
                    "inherits its association with the harm LABEL from the model it was "
                    "derived from, and its sign is definitional rather than evidence"
                ),
                "why_unsupervised": (
                    "an aggregator fitted against the harm label would be a second harm model, "
                    "and subtracting a harm model from a harm-aware utility measures nothing "
                    "about uncertainty. The weights are therefore fixed and equal; only "
                    "lambda1 and lambda2 are chosen, and they are chosen on calibration rows "
                    "of the fit engines by an inner leave-one-engine-out"
                ),
                "engine_identity_used": False,
                "domain_labels_used": False,
                "ground_truth_used": False,
            },
            "signals": [
                {
                    "name": s.name,
                    "block": s.block,
                    "tags": list(s.tags),
                    "source": s.source,
                    "definition": s.definition,
                    "rationale": s.rationale,
                }
                for s in SIGNALS
            ],
            "variants": {
                variant: {
                    "members": list(members),
                    "ablation": ABLATION_LABELS.get(variant, "the full signal set"),
                }
                for variant, members in VARIANTS.items()
            },
            "constants": {
                "ensemble_members": ENSEMBLE_MEMBERS,
                "ensemble_row_fraction": ENSEMBLE_ROW_FRACTION,
                "ensemble_feature_fraction": ENSEMBLE_FEATURE_FRACTION,
                "dropout_draws": DROPOUT_DRAWS,
                "dropout_rate": DROPOUT_RATE,
                "mahalanobis_ridge": MAHALANOBIS_RIDGE,
                "fit_seed": pilot.FIT_SEED,
            },
            "determinism_checked": True,
            "folds": folds,
            "confirmatory_accessed": False,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(f"features: {len(SIGNAL_NAMES)} signals -> {cc._relative(FEATURE_SPEC)}")
    return 0


# ------------------------------------------------------------------ scores, fitted once

# Every fold is fitted exactly once and the row-level result is a write-once table. Each later
# stage is then a pure function of that table, so a reviewer can recompute any headline
# without running a model, and two stages cannot silently have scored different rows.

CROSS_ENGINE = "cross_engine"
IN_DOMAIN = "in_domain"
SOURCE_CALIBRATION = "source_calibration"


def run_scores() -> int:
    started = time.monotonic()
    design = dg.load_design()
    base_class, mapped_class = rl.error_classes(design)
    frames: list[pd.DataFrame] = []
    record: dict[str, Any] = {}

    for held_out in design.engines:
        fitted = fit_fold(design, held_out)
        fold = fitted.fold
        thresholds: dict[str, Any] = {}
        calibration_arms = fitted.arms(design, fold.source_cal)
        for name, score in calibration_arms.items():
            finite = np.isfinite(score)
            if not finite.any():
                continue
            thresholds[name] = {
                controller: select_threshold(
                    score[finite],
                    design.harmful[fold.source_cal][finite],
                    PRIMARY_EPSILON,
                    delta=DELTA,
                    controller=controller,
                ).as_dict()
                for controller in CONTROLLERS
            }

        for mode, index in (
            (CROSS_ENGINE, fold.eval),
            (IN_DOMAIN, fold.seen_eval),
            (SOURCE_CALIBRATION, fold.source_cal),
        ):
            arms = fitted.arms(design, index)
            probabilities = fitted.uncertainty.probabilities(design, index)
            composites = fitted.uncertainty.composites(design, index)
            percentiles = fitted.uncertainty.signal_percentiles(design, index)
            frame = pd.DataFrame(
                {
                    "candidate_id": design.meta["candidate_id"].astype(str).to_numpy()[index],
                    "document_id": design.documents[index],
                    "engine_id": design.meta["engine_id"].to_numpy(str)[index],
                    "held_out_engine": held_out,
                    "evaluation_mode": mode,
                    "is_harmful": design.harmful[index],
                    "beneficial": design.beneficial[index],
                    "base_error_class": base_class[index].astype(str),
                    "error_class": mapped_class[index].astype(str),
                    "p_harm": probabilities["harm"],
                    "p_harm_raw": probabilities["harm_raw"],
                    "p_harm_ensemble_raw": fitted.uncertainty.heads.ensemble_spread(
                        probabilities["scaled"]
                    )[0],
                    "p_benefit": probabilities["benefit"],
                    "p_benefit_raw": probabilities["benefit_raw"],
                }
            )
            for name, values in percentiles.items():
                frame[f"signal__{name}"] = values
            for variant, values in composites.items():
                frame[f"uncertainty__{variant}"] = values
            for alpha in CONFORMAL_ALPHAS:
                sets = fitted.conformal.sets(probabilities["harm"], alpha)
                frame[f"conformal_ambiguous__{alpha:g}"] = sets["ambiguous"]
                frame[f"conformal_empty__{alpha:g}"] = sets["empty"]
                frame[f"conformal_singleton_harm__{alpha:g}"] = sets["singleton_harm"]
            for name, values in arms.items():
                frame[f"arm__{name}"] = values
            if mode == CROSS_ENGINE:
                score, oracle = fitted.oracle(design, index)
                frame[f"arm__{CEILING_ARMS[0]}"] = score
                fitted.diagnostics["oracle"] = oracle
            else:
                frame[f"arm__{CEILING_ARMS[0]}"] = np.nan
            frames.append(frame)

        record[held_out] = {
            "held_out_engine": held_out,
            "train_engines": list(fold.train_engines),
            "rows": {
                "fit": int(fold.fit.size),
                "source_calibration": int(fold.source_cal.size),
                "in_domain_development": int(fold.seen_eval.size),
                "cross_engine_development": int(fold.eval.size),
            },
            "documents": {
                "fit": len(set(design.documents[fold.fit].tolist())),
                "source_calibration": len(set(design.documents[fold.source_cal].tolist())),
                "cross_engine_development": len(set(design.documents[fold.eval].tolist())),
            },
            "selected_lambdas": {
                name: {"lambda1": pair[0], "lambda2": pair[1]}
                for name, pair in fitted.selection.lambdas.items()
            },
            "inner_selection": fitted.selection.inner,
            "diagnostics": fitted.diagnostics,
            "calibration": {
                "method": pilot.CALIBRATION_METHOD,
                "fitted_on": "fit engines, CALIBRATION documents",
                "rows": int(fold.source_cal.size),
            },
            "certified_thresholds": thresholds,
        }
        l1, l2 = fitted.selection.lambdas[HEADLINE_VARIANT]
        print(
            f"  fold {held_out:11s} lambda1={l1:g} lambda2={l2:+g} "
            f"rho(u,U)={fitted.diagnostics['spearman_uncertainty_vs_utility']:+.3f} "
            f"sgv2={fitted.selection.shift.detector}"
        )

    cc._write_parquet_once(SCORES, pd.concat(frames, ignore_index=True))
    cc._write_json_once(
        SELECTION_RECORD,
        {
            "schema_version": "sgv3-selfaware-selection-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV3-U1",
            "selection_scope": (
                "lambda1 and lambda2 are chosen per variant by an inner leave-one-engine-out "
                "over the FIT engines, evaluated on CALIBRATION documents with the "
                "calibration pages split in half so the frozen percentile reference and the "
                "inner evaluation rows never share a page. The held-out engine contributes "
                "nothing to any choice; tests/leakage re-derives every selection from the "
                "recorded inner scores."
            ),
            "engine_identity_used_by_method": False,
            "engine_identity_used_by_sgv2_baseline": True,
            "baseline_note": (
                "SGV2's arm is reproduced with the detector its own inner selection chose, "
                "which may be the supervised domain classifier over engine labels. A baseline "
                "may use what the proposed method may not; the label-free restriction is run "
                "beside it as shift_aware_cold so the comparison exists in both forms."
            ),
            "folds": record,
            "confirmatory_accessed": False,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(f"scores: {sum(len(f) for f in frames)} rows -> {cc._relative(SCORES)}")
    return 0


@dataclass(slots=True)
class Slice:
    """One (fold, evaluation mode) block, read back from the write-once score table."""

    held_out: str
    mode: str
    frame: pd.DataFrame
    harmful: np.ndarray
    beneficial: np.ndarray
    documents: np.ndarray

    def arm(self, name: str) -> np.ndarray:
        return self.frame[f"arm__{name}"].to_numpy(dtype=np.float64)

    def column(self, name: str) -> np.ndarray:
        return self.frame[name].to_numpy(dtype=np.float64)

    @property
    def arm_names(self) -> list[str]:
        return [c[len("arm__") :] for c in self.frame.columns if c.startswith("arm__")]


def load_scores() -> tuple[dict[tuple[str, str], Slice], dict[str, Any]]:
    if not SCORES.is_file() or not SELECTION_RECORD.is_file():
        raise PhaseError("run --scores before any experiment that reads them")
    table = pd.read_parquet(SCORES)
    record = cc._read_json(SELECTION_RECORD)
    slices: dict[tuple[str, str], Slice] = {}
    for (held_out, mode), group in table.groupby(["held_out_engine", "evaluation_mode"], sort=True):
        frame = group.reset_index(drop=True)
        engines = set(frame["engine_id"].astype(str))
        if mode == CROSS_ENGINE and engines != {str(held_out)}:
            raise PhaseError(f"{held_out}: a non-held-out engine is in its cross-engine rows")
        if mode != CROSS_ENGINE and str(held_out) in engines:
            raise PhaseError(f"{held_out}: the held-out engine is in its {mode} rows")
        slices[(str(held_out), str(mode))] = Slice(
            held_out=str(held_out),
            mode=str(mode),
            frame=frame,
            harmful=frame["is_harmful"].to_numpy(dtype=bool),
            beneficial=frame["beneficial"].to_numpy(dtype=bool),
            documents=frame["document_id"].astype(str).to_numpy(),
        )
    return slices, record


# ------------------------------------------------------------------ stage 3: calibration

# The brief asks for four calibration methods and says not to judge them by ECE alone. Two of
# the four cannot change the primary endpoint at all, and saying so up front is more useful
# than measuring it four times and leaving the reader to notice:
#
#   temperature and isotonic are MONOTONE maps of the input probability, so they leave
#   argsort(-score) fixed and every ranking-derived quantity -- repair recall at bounded harm,
#   AURC, coverage at risk -- is identical to the uncalibrated one, up to ties isotonic
#   introduces. They can still move a DEPLOYED threshold, which is a different question and is
#   reported separately.
#
#   ensemble averaging is not a monotone map of any single member's output, so it can reorder.
#   conformal prediction does not produce a score at all: it removes rows it cannot resolve,
#   which is a truncation, and a truncation can move the frontier without reordering anything
#   (SGV2 established the point; `tests/leakage` re-asserts it on constructed data).
#
# The invariance is measured rather than assumed, because an implementation bug would look
# exactly like a scientific finding here.


def _probability_report(
    probability: np.ndarray, harmful: np.ndarray, beneficial: np.ndarray
) -> dict[str, Any]:
    score = 1.0 - probability
    curve = risk_coverage_curve(score, harmful)
    report: dict[str, Any] = {
        "brier": float(brier_score(probability, harmful.astype(float))),
        # ECE under BOTH binning schemes with n_bins recorded, because equal-width ECE is
        # binning-biased and a single number would be misleading (.claude/rules/metrics.md).
        "expected_calibration_error": {
            binning: dict(
                zip(
                    ("ece", "max_calibration_error"),
                    expected_calibration_error(
                        probability, harmful.astype(float), n_bins=ECE_BINS, binning=binning
                    ),
                    strict=True,
                )
            )
            for binning in BINNINGS
        },
        "murphy_decomposition": {
            binning: dict(
                zip(
                    ("calibration", "refinement", "uncertainty"),
                    murphy_decomposition(
                        probability, harmful.astype(float), n_bins=ECE_BINS, binning=binning
                    ),
                    strict=True,
                )
            )
            for binning in BINNINGS
        },
        "ece_bins": ECE_BINS,
        "aurc": float(aurc(curve)),
        "frontier": achievable_repair_recall(score, harmful, beneficial),
        "ranking_fingerprint": _ranking_fingerprint(score),
    }
    for epsilon in EPSILONS:
        point = coverage_at_risk(curve, epsilon)
        report[f"coverage_at_risk_{int(epsilon * 100)}"] = (
            None if point is None else {"coverage": point.coverage, "risk": point.risk}
        )
    return report


def _label_covered(block: Slice, alpha: float) -> np.ndarray:
    """Does the conformal prediction set contain the row's true label?

    An ambiguous set holds both labels and therefore always covers; an empty set holds
    neither and never does; a singleton covers exactly when it names the right label.
    """
    ambiguous = block.frame[f"conformal_ambiguous__{alpha:g}"].to_numpy(dtype=bool)
    empty = block.frame[f"conformal_empty__{alpha:g}"].to_numpy(dtype=bool)
    singleton_harm = block.frame[f"conformal_singleton_harm__{alpha:g}"].to_numpy(dtype=bool)
    singleton_safe = ~ambiguous & ~empty & ~singleton_harm
    return ambiguous | (singleton_harm & block.harmful) | (singleton_safe & ~block.harmful)


def _ranking_fingerprint(score: np.ndarray) -> str:
    """A hash of the accept ORDER, so "did this method reorder" is a fact, not an eyeball."""
    import hashlib

    order = np.argsort(-score, kind="stable").astype(np.int64)
    return hashlib.sha256(order.tobytes()).hexdigest()[:16]


def _inverts_strict_order(reference: np.ndarray, mapped: np.ndarray) -> bool:
    """Does ``mapped`` ever rank a row above one that ``reference`` ranked strictly below?

    A first version of this check compared accept-order fingerprints and reported isotonic as
    reordering on all four folds. It does not: isotonic is WEAKLY monotone, it maps many
    inputs onto one output, and the stable sort then orders the newly tied rows by index. That
    is a coarsening, not an inversion, and the two have to be told apart -- a coarsening can
    still move the frontier, because tie-breaking decides which rows enter a prefix, but it
    cannot put a safer-looking row below a riskier one.

    Sorting by reference with ties broken by ``mapped`` makes the test exact: within a
    reference tie ``mapped`` is ascending by construction, so any decrease that remains
    crosses a strict reference step and is a genuine inversion.
    """
    order = np.lexsort((mapped, reference))
    return bool(np.any(np.diff(mapped[order]) < 0.0))


def run_calibration() -> int:
    started = time.monotonic()
    slices, _ = load_scores()
    engines = sorted({key[0] for key in slices})
    folds: dict[str, Any] = {}

    for held_out in engines:
        calibration = slices[(held_out, SOURCE_CALIBRATION)]
        fitted_calibrators: dict[str, Any] = {}
        for method in ("temperature", "isotonic"):
            calibrator = build_calibrator(method)
            calibrator.fit(calibration.column("p_harm_raw"), calibration.harmful.astype(float))
            fitted_calibrators[method] = calibrator
        ensemble_calibrator = build_calibrator(pilot.CALIBRATION_METHOD)
        ensemble_calibrator.fit(
            calibration.column("p_harm_ensemble_raw"), calibration.harmful.astype(float)
        )

        modes: dict[str, Any] = {}
        for mode in (CROSS_ENGINE, IN_DOMAIN, SOURCE_CALIBRATION):
            block = slices[(held_out, mode)]
            raw = block.column("p_harm_raw")
            probabilities = {
                "identity": raw,
                "temperature": np.asarray(
                    fitted_calibrators["temperature"].transform(raw), dtype=float
                ),
                "isotonic": np.asarray(fitted_calibrators["isotonic"].transform(raw), dtype=float),
                "ensemble": np.asarray(
                    ensemble_calibrator.transform(block.column("p_harm_ensemble_raw")), dtype=float
                ),
            }
            reports = {
                method: _probability_report(values, block.harmful, block.beneficial)
                for method, values in probabilities.items()
            }
            identity_fingerprint = reports["identity"]["ranking_fingerprint"]
            identity_key = f"epsilon_{int(PRIMARY_EPSILON * 100)}"
            for method, report in reports.items():
                report["accept_order_differs_from_identity"] = bool(
                    report["ranking_fingerprint"] != identity_fingerprint
                )
                report["inverts_a_strictly_ordered_pair"] = _inverts_strict_order(
                    probabilities["identity"], probabilities[method]
                )
                report["distinct_values"] = int(np.unique(probabilities[method]).size)
                report["distinct_values_identity"] = int(np.unique(probabilities["identity"]).size)
                report["monotone_by_construction"] = method in MONOTONE_METHODS
                report["repair_recall_shift_vs_identity"] = float(
                    report["frontier"][identity_key]["repair_recall"]
                    - reports["identity"]["frontier"][identity_key]["repair_recall"]
                )

            conformal: dict[str, Any] = {}
            for alpha in CONFORMAL_ALPHAS:
                ambiguous = block.frame[f"conformal_ambiguous__{alpha:g}"].to_numpy(dtype=bool)
                empty = block.frame[f"conformal_empty__{alpha:g}"].to_numpy(dtype=bool)
                # Marginal validity: does the prediction set contain the TRUE label at least
                # (1 - alpha) of the time? Split conformal guarantees this only under
                # exchangeability between calibration and test rows, which an unseen engine
                # breaks by construction. In-domain and calibration rows are the reference
                # against which the cross-engine number is read.
                covered = _label_covered(block, alpha)
                conformal[f"alpha_{alpha:g}"] = {
                    "nominal_coverage": 1.0 - alpha,
                    "empirical_label_coverage": float(covered.mean()),
                    "coverage_shortfall": float((1.0 - alpha) - covered.mean()),
                    "guarantee_held": bool(covered.mean() >= 1.0 - alpha),
                    "ambiguous_rate": float(ambiguous.mean()),
                    "empty_rate": float(empty.mean()),
                    "abstention_rate": float((ambiguous | empty).mean()),
                    "harm_rate_among_abstained": float(block.harmful[ambiguous | empty].mean())
                    if (ambiguous | empty).any()
                    else float("nan"),
                    "harm_rate_among_resolved": float(block.harmful[~(ambiguous | empty)].mean())
                    if (~(ambiguous | empty)).any()
                    else float("nan"),
                }
            modes[mode] = {
                "rows": len(block.frame),
                "harm_base_rate": float(block.harmful.mean()),
                "probability_methods": reports,
                "conformal": conformal,
            }
        folds[held_out] = {
            "held_out_engine": held_out,
            "calibrators_fitted_on": "fit engines, CALIBRATION documents",
            "modes": modes,
        }
        cross = folds[held_out]["modes"][CROSS_ENGINE]["probability_methods"]
        print(
            f"  fold {held_out:11s} brier identity={cross['identity']['brier']:.4f} "
            f"isotonic={cross['isotonic']['brier']:.4f} "
            f"inverts: isotonic={cross['isotonic']['inverts_a_strictly_ordered_pair']} "
            f"ensemble={cross['ensemble']['inverts_a_strictly_ordered_pair']}"
        )

    cc._write_json_once(
        CALIBRATION_RESULTS,
        {
            "schema_version": "sgv3-calibration-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV3-U1",
            "methods": list(CALIBRATION_METHODS),
            "monotone_methods": list(MONOTONE_METHODS),
            "monotone_note": (
                "temperature and isotonic are monotone in their input, so neither can invert a "
                "strictly ordered pair -- `inverts_a_strictly_ordered_pair` is the exact test "
                "and is False for both on every fold. Isotonic is only WEAKLY monotone: it "
                "collapses many inputs onto one value, and the resulting ties are broken by row "
                "order, so its accept order differs from the identity's without any inversion. "
                "A coarsening can still move the frontier, because tie-breaking decides which "
                "rows enter a prefix; `repair_recall_shift_vs_identity` records by how much. "
                "Ensemble averaging is not a monotone map of any single member and can invert"
            ),
            "conformal_note": (
                "split conformal over the two labels {harmful, safe}. Its marginal validity "
                "assumes calibration and test rows are exchangeable; an unseen engine breaks "
                "that assumption by construction, so the empirical label coverage is reported "
                "against the nominal level on calibration, in-domain and cross-engine rows and "
                "the three are read together"
            ),
            "conformal_alphas": list(CONFORMAL_ALPHAS),
            "ece_bins": ECE_BINS,
            "ece_binnings": list(BINNINGS),
            "folds": folds,
            "confirmatory_accessed": False,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(f"calibration: {len(folds)} folds -> {cc._relative(CALIBRATION_RESULTS)}")
    return 0


# ------------------------------------------------------------------ curves and metrics

# A rejected row must never be accepted, and Phase 5's endpoint has no way to know that: it
# ranks whatever vector it is handed. Substituting a very negative finite score for a rejected
# row is not enough -- the achievable frontier chooses its threshold with hindsight, so if the
# harm bound still holds after the real rows run out, the largest feasible prefix walks into
# the rejected block and credits an arm with repairs it refused to make.
#
# The fix reuses the one definition rather than restating it: the endpoint is computed on the
# ELIGIBLE rows and the two ratios are rescaled back onto the full site set, which is exact
# (`repair_recall` and `coverage` are both linear in their denominators) and keeps arms that
# reject different amounts on one comparable axis. `tests/leakage` constructs the case where
# the two differ and asserts this function, not the raw one, is what the stage reports.


def _restricted_frontier(
    score: np.ndarray, harmful: np.ndarray, beneficial: np.ndarray
) -> dict[str, Any]:
    eligible = np.isfinite(score)
    total_beneficial = int(beneficial.sum())
    empty = {
        f"epsilon_{int(e * 100)}": {
            "repair_recall": 0.0,
            "coverage": 0.0,
            "n_accepted": 0,
            "realized_harm_rate": 0.0,
        }
        for e in EPSILONS
    }
    base = {
        "total_beneficial": total_beneficial,
        "n": int(score.size),
        "eligible_rows": int(np.count_nonzero(eligible)),
    }
    if not eligible.any() or total_beneficial == 0:
        return {**base, **empty, "eligible_beneficial": 0}
    inner = achievable_repair_recall(score[eligible], harmful[eligible], beneficial[eligible])
    recall_scale = float(inner["total_beneficial"]) / float(total_beneficial)
    coverage_scale = float(np.count_nonzero(eligible)) / float(score.size)
    out: dict[str, Any] = {**base, "eligible_beneficial": int(inner["total_beneficial"])}
    for epsilon in EPSILONS:
        key = f"epsilon_{int(epsilon * 100)}"
        cell = dict(inner[key])
        cell["repair_recall"] = float(cell["repair_recall"] * recall_scale)
        cell["coverage"] = float(cell["coverage"] * coverage_scale)
        out[key] = cell
    return out


def _restricted_interval(
    score: np.ndarray,
    harmful: np.ndarray,
    beneficial: np.ndarray,
    documents: np.ndarray,
    epsilon: float,
) -> dict[str, float]:
    key = f"epsilon_{int(epsilon * 100)}"

    def statistic(index: np.ndarray) -> float:
        if index.size == 0 or beneficial[index].sum() == 0:
            return float("nan")
        return float(
            _restricted_frontier(score[index], harmful[index], beneficial[index])[key][
                "repair_recall"
            ]
        )

    result = cluster_bootstrap_indices(
        list(documents),
        statistic,
        n_resamples=BOOTSTRAP_RESAMPLES,
        seed=pilot.BOOTSTRAP_SEED,
        bounds=(0.0, 1.0),
    )
    return {
        "estimate": result.estimate,
        "ci_lower": result.lower,
        "ci_upper": result.upper,
        "degenerate_interval": bool(result.degenerate_interval),
        "n_documents": result.n_clusters,
    }


def _arm_summary(
    score: np.ndarray,
    harmful: np.ndarray,
    beneficial: np.ndarray,
    documents: np.ndarray,
    *,
    with_interval: bool = True,
) -> dict[str, Any]:
    """Everything one acceptance rule delivers, abstention reported as abstention."""
    eligible = np.isfinite(score)
    summary = _restricted_frontier(score, harmful, beneficial)
    summary["rejected_rows"] = int(score.size - np.count_nonzero(eligible))
    summary["abstention_rate_before_threshold"] = float(1.0 - eligible.mean())
    if eligible.any():
        kept, kept_harm, kept_benefit = score[eligible], harmful[eligible], beneficial[eligible]
        summary["aurc_over_eligible_rows"] = float(aurc(risk_coverage_curve(kept, kept_harm)))
        summary["auc_safe"] = (
            float(roc_auc(kept, (~kept_harm).astype(float)))
            if kept_harm.any() and (~kept_harm).any()
            else float("nan")
        )
        summary["auc_beneficial"] = (
            float(roc_auc(kept, kept_benefit.astype(float)))
            if kept_benefit.any() and (~kept_benefit).any()
            else float("nan")
        )
    else:
        summary["aurc_over_eligible_rows"] = float("nan")
        summary["auc_safe"] = float("nan")
        summary["auc_beneficial"] = float("nan")
    summary["matched_coverage"] = {
        f"coverage_{int(c * 100)}": _selective_metrics(score, harmful, beneficial, c)
        for c in MATCHED_COVERAGES
    }
    summary["matched_repair_recall"] = {
        f"recall_{int(r * 100)}": harm_at_matched_repair_recall(score, harmful, beneficial, r)
        for r in rl.MATCHED_RECALLS
    }
    if with_interval:
        summary["interval_primary_epsilon"] = _restricted_interval(
            score, harmful, beneficial, documents, PRIMARY_EPSILON
        )
    return summary


HEADLINE_ARMS = (
    "no_correction",
    "confidence_only",
    "harm_only",
    "harm_aware",
    "shift_aware",
    "shift_aware_cold",
    "selfaware",
    "selfaware_ensemble",
    "selfaware_conformal",
    "oracle_selfaware",
)


def _selective_metrics(
    score: np.ndarray, harmful: np.ndarray, beneficial: np.ndarray, coverage: float
) -> dict[str, Any]:
    """Everything the brief asks to be reported at a fixed accept volume, in one place."""
    cell = harm_at_matched_coverage(score, harmful, beneficial, coverage)
    if cell["n_accepted"] == 0:
        return {**cell, "selective_accuracy": float("nan"), "beneficial_precision": 0.0}
    order = rl._ranked(score)[: cell["n_accepted"]]
    return {
        **cell,
        # The brief's "selective accuracy". Defined explicitly because it is ambiguous: here
        # an accepted edit counts as correct when accepting it did not increase the distance
        # to ground truth, so it is exactly 1 - selective risk and is reported as such rather
        # than as an independent third number.
        "selective_accuracy": float(1.0 - cell["realized_harm_rate"]),
        "beneficial_precision": float(beneficial[order].mean()),
    }


def _curve(
    score: np.ndarray, harmful: np.ndarray, beneficial: np.ndarray, points: int
) -> dict[str, list[float]]:
    """Coverage, selective risk and repair recall along the accept order."""
    order = rl._ranked(score)
    total = int(beneficial.sum())
    if order.size == 0:
        return {"coverage": [0.0], "selective_risk": [float("nan")], "repair_recall": [0.0]}
    harm_cum = np.cumsum(harmful[order])
    ben_cum = np.cumsum(beneficial[order])
    accepted = np.arange(1, order.size + 1)
    keep = np.unique(np.linspace(0, order.size - 1, min(points, order.size)).astype(int))
    return {
        "coverage": [float(v) for v in accepted[keep] / score.size],
        "selective_risk": [float(v) for v in harm_cum[keep] / accepted[keep]],
        "repair_recall": [float(v) for v in ben_cum[keep] / max(total, 1)],
    }


def run_curves() -> int:
    """The risk-coverage curves, in-domain and cross-engine, for every arm."""
    started = time.monotonic()
    slices, _ = load_scores()
    engines = sorted({key[0] for key in slices})
    folds: dict[str, Any] = {}
    for held_out in engines:
        modes: dict[str, Any] = {}
        for mode in (CROSS_ENGINE, IN_DOMAIN):
            block = slices[(held_out, mode)]
            arms: dict[str, Any] = {}
            for name in block.arm_names:
                score = block.arm(name)
                if np.isnan(score).all():
                    continue
                eligible = np.isfinite(score)
                arms[name] = {
                    "eligible_rows": int(np.count_nonzero(eligible)),
                    "abstention_rate_before_threshold": float(1.0 - eligible.mean()),
                    "frontier": _restricted_frontier(score, block.harmful, block.beneficial),
                    "aurc_over_eligible_rows": float(
                        aurc(risk_coverage_curve(score[eligible], block.harmful[eligible]))
                    )
                    if eligible.any()
                    else float("nan"),
                    "selective": {
                        f"coverage_{int(c * 100)}": _selective_metrics(
                            score, block.harmful, block.beneficial, c
                        )
                        for c in MATCHED_COVERAGES
                    },
                    "curve": (
                        _curve(score, block.harmful, block.beneficial, CURVE_POINTS)
                        if name in HEADLINE_ARMS
                        else None
                    ),
                }
            modes[mode] = {"rows": len(block.frame), "arms": arms}
        folds[held_out] = {"held_out_engine": held_out, "modes": modes}
        cross = folds[held_out]["modes"][CROSS_ENGINE]["arms"]
        key = f"epsilon_{int(PRIMARY_EPSILON * 100)}"
        print(
            f"  fold {held_out:11s} "
            + " ".join(
                f"{n}={cross[n]['frontier'][key]['repair_recall']:.3f}"
                for n in ("harm_only", "harm_aware", "shift_aware", "selfaware")
            )
        )

    cc._write_json_once(
        CURVE_RESULTS,
        {
            "schema_version": "sgv3-risk-coverage-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV3-U1",
            "epsilon_grid": list(EPSILONS),
            "primary_epsilon": PRIMARY_EPSILON,
            "matched_coverages": list(MATCHED_COVERAGES),
            "curve_points": CURVE_POINTS,
            "definitions": {
                "coverage": "accepted edits / all candidate sites considered",
                "selective_risk": "harmful accepted / accepted",
                "selective_accuracy": "1 - selective risk",
                "repair_recall": "beneficial accepted / all beneficial candidates available",
                "aurc": "area under the risk-coverage curve, from ocr_risk.metrics.selective",
                "abstention_rate_before_threshold": (
                    "rows the rule refused outright, before any threshold; an arm that "
                    "abstains everywhere is recorded as abstaining, never as safe"
                ),
            },
            "folds": folds,
            "confirmatory_accessed": False,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(f"curves: {len(folds)} folds -> {cc._relative(CURVE_RESULTS)}")
    return 0


# ------------------------------------------------------------------ stage 4: three actions

# CORRECT applies the edit. PRESERVE keeps the OCR output because the model believes it is
# already right. ABSTAIN keeps the OCR output because the model cannot say. On TEXT the last
# two are the same event -- both leave `O` in place, both give `d_after = d_before`, and
# `edits/outcome.py` scores neither as harmful nor beneficial -- so no benchmark number can
# separate them here and none is reported as if it could.
#
# What separates them is composition. A three-way split earns its review budget only if the
# rows it sends to a human are richer in recoverable repairs than the rows it silently leaves
# alone. That difference IS measurable, it is the claim this stage tests, and it is reported
# with a document-clustered interval like every other contrast.

ACTIONS = ("CORRECT", "PRESERVE", "ABSTAIN")


def _action_labels(
    block: Slice, tau: float, alpha: float, uncertainty_cut: float, variant: str
) -> np.ndarray:
    unresolvable = block.frame[f"conformal_ambiguous__{alpha:g}"].to_numpy(
        dtype=bool
    ) | block.frame[f"conformal_empty__{alpha:g}"].to_numpy(dtype=bool)
    uncertain = block.column(f"uncertainty__{variant}") >= uncertainty_cut
    score = block.arm(PROPOSED_ARM)
    labels = np.full(len(block.frame), "PRESERVE", dtype=object)
    labels[np.isfinite(score) & (score >= tau)] = "CORRECT"
    labels[unresolvable | uncertain] = "ABSTAIN"
    return labels


def _rate_difference(
    left: np.ndarray, right: np.ndarray, outcome: np.ndarray, documents: np.ndarray
) -> dict[str, float]:
    """Difference of two subgroup rates, resampled by document and recomputed inside."""

    def statistic(index: np.ndarray) -> float:
        a, b = left[index], right[index]
        if not a.any() or not b.any():
            return float("nan")
        return float(outcome[index][a].mean() - outcome[index][b].mean())

    result = cluster_bootstrap_indices(
        list(documents),
        statistic,
        n_resamples=BOOTSTRAP_RESAMPLES,
        seed=pilot.BOOTSTRAP_SEED,
        bounds=(-1.0, 1.0),
    )
    return {
        "estimate": result.estimate,
        "ci_lower": result.lower,
        "ci_upper": result.upper,
        "p_value_two_sided": result.p_value_two_sided,
        "degenerate_interval": bool(result.degenerate_interval),
    }


def run_actions() -> int:
    started = time.monotonic()
    slices, record = load_scores()
    engines = sorted({key[0] for key in slices})
    folds: dict[str, Any] = {}

    for held_out in engines:
        thresholds = record["folds"][held_out]["certified_thresholds"][PROPOSED_ARM]
        tau = float(thresholds["ltt_bentkus"]["tau"])
        modes: dict[str, Any] = {}
        for mode in (CROSS_ENGINE, IN_DOMAIN):
            block = slices[(held_out, mode)]
            cuts: dict[str, Any] = {}
            for cut in ABSTAIN_PERCENTILES:
                labels = _action_labels(block, tau, PRIMARY_EPSILON, cut, HEADLINE_VARIANT)
                masks = {action: labels == action for action in ACTIONS}
                counts = {
                    action: {
                        "n": int(mask.sum()),
                        "share_of_sites": float(mask.mean()),
                        "harm_rate": float(block.harmful[mask].mean())
                        if mask.any()
                        else float("nan"),
                        "beneficial_rate": float(block.beneficial[mask].mean())
                        if mask.any()
                        else float("nan"),
                    }
                    for action, mask in masks.items()
                }
                cuts[f"uncertainty_cut_{int(cut * 100)}"] = {
                    "uncertainty_percentile_cut": cut,
                    "conformal_alpha": PRIMARY_EPSILON,
                    "tau": tau,
                    "actions": counts,
                    # The claim the three-way split has to earn. Positive means the rows sent
                    # to a human carry more recoverable repairs than the rows left silently
                    # alone, which is the only sense in which ABSTAIN differs from PRESERVE in
                    # this benchmark.
                    "abstain_minus_preserve_beneficial_rate": _rate_difference(
                        masks["ABSTAIN"],
                        masks["PRESERVE"],
                        block.beneficial.astype(float),
                        block.documents,
                    ),
                    "abstain_minus_preserve_harm_rate": _rate_difference(
                        masks["ABSTAIN"],
                        masks["PRESERVE"],
                        block.harmful.astype(float),
                        block.documents,
                    ),
                    "perfect_reviewer_bound": {
                        "assumption": "a reviewer who is always correct and always available",
                        "review_load_fraction_of_sites": float(masks["ABSTAIN"].mean()),
                        "beneficial_edits_recoverable": int(
                            block.beneficial[masks["ABSTAIN"]].sum()
                        ),
                        "harmful_edits_the_reviewer_would_reject": int(
                            block.harmful[masks["ABSTAIN"]].sum()
                        ),
                        "note": (
                            "an upper bound under a stated assumption, never a measured "
                            "result; there is no human-review channel in this benchmark"
                        ),
                    },
                }
            modes[mode] = {
                "rows": len(block.frame),
                "harm_base_rate": float(block.harmful.mean()),
                "beneficial_base_rate": float(block.beneficial.mean()),
                "cuts": cuts,
            }
        folds[held_out] = {
            "held_out_engine": held_out,
            "tau_controller": "ltt_bentkus",
            "tau": tau,
            "tau_feasible": bool(thresholds["ltt_bentkus"]["feasible"]),
            "modes": modes,
        }
        cell = modes[CROSS_ENGINE]["cuts"]["uncertainty_cut_90"]
        print(
            f"  fold {held_out:11s} "
            f"CORRECT={cell['actions']['CORRECT']['share_of_sites']:.3f} "
            f"PRESERVE={cell['actions']['PRESERVE']['share_of_sites']:.3f} "
            f"ABSTAIN={cell['actions']['ABSTAIN']['share_of_sites']:.3f} "
            f"benefit(A-P)={cell['abstain_minus_preserve_beneficial_rate']['estimate']:+.3f}"
        )

    cc._write_json_once(
        ACTION_RESULTS,
        {
            "schema_version": "sgv3-action-selection-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV3-U1",
            "actions": list(ACTIONS),
            "decision_rule": {
                "ABSTAIN": (
                    "the conformal prediction set is ambiguous or empty at alpha, OR the "
                    "composite uncertainty percentile is at or above the cut"
                ),
                "CORRECT": (
                    "not abstaining, and the self-aware utility is at or above the certified "
                    "threshold chosen on calibration rows of the fit engines"
                ),
                "PRESERVE": "not abstaining, and the utility is below that threshold",
            },
            "text_outcome_note": (
                "PRESERVE and ABSTAIN leave `O` in place and are the same event under "
                "edits/outcome.py: both give d_after = d_before and neither is harmful nor "
                "beneficial. The benchmark has no human-review channel, so the split is "
                "reported by COMPOSITION -- what each set contains -- and by a perfect-reviewer "
                "bound, never as a measured text-quality difference"
            ),
            "uncertainty_cuts": list(ABSTAIN_PERCENTILES),
            "uncertainty_variant": HEADLINE_VARIANT,
            "folds": folds,
            "confirmatory_accessed": False,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(f"actions: {len(folds)} folds -> {cc._relative(ACTION_RESULTS)}")
    return 0


# ------------------------------------------------------------------ stage 5: per engine

# The brief's primary objective is repair coverage under a bounded harmful-correction risk, so
# the primary contrast is the PAIRED difference in repair recall at the bounded harm level.
# SGV2's matched-coverage contrast is reported beside it as a co-primary, because the two fail
# in different ways: repair recall at bounded harm can be won by an arm that simply accepts
# more when it happens to be safe to, and harm at matched coverage cannot be won by abstaining.
# Neither alone is enough; agreement between them is what a claim would need.


def _paired_repair_recall_delta(
    arm: np.ndarray,
    baseline: np.ndarray,
    harmful: np.ndarray,
    beneficial: np.ndarray,
    documents: np.ndarray,
    epsilon: float,
) -> dict[str, float]:
    """Paired by construction: both rankings are rebuilt inside the SAME resampled rows."""

    key = f"epsilon_{int(epsilon * 100)}"

    def recall(score: np.ndarray, index: np.ndarray) -> float:
        return float(
            _restricted_frontier(score[index], harmful[index], beneficial[index])[key][
                "repair_recall"
            ]
        )

    def statistic(index: np.ndarray) -> float:
        if index.size == 0 or beneficial[index].sum() == 0:
            return float("nan")
        return recall(arm, index) - recall(baseline, index)

    result = cluster_bootstrap_indices(
        list(documents),
        statistic,
        n_resamples=BOOTSTRAP_RESAMPLES,
        seed=pilot.BOOTSTRAP_SEED,
        bounds=(-1.0, 1.0),
    )
    return {
        "delta_repair_recall": result.estimate,
        "ci_lower": result.lower,
        "ci_upper": result.upper,
        "p_value_two_sided": result.p_value_two_sided,
        # A degenerate interval is a grafted one-sided bound, not a resampled one: every draw
        # returned the same value, which happens when the two arms are the same vector. It is
        # flagged so such a cell can never be read as evidence in either direction.
        "degenerate_interval": bool(result.degenerate_interval),
        "favours_arm": bool(result.lower > 0.0 and not result.degenerate_interval),
        "n_documents": result.n_clusters,
    }


def run_transfer() -> int:
    started = time.monotonic()
    slices, record = load_scores()
    engines = sorted({key[0] for key in slices})
    key = f"epsilon_{int(PRIMARY_EPSILON * 100)}"
    folds: dict[str, Any] = {}

    for held_out in engines:
        thresholds = record["folds"][held_out]["certified_thresholds"]
        modes: dict[str, Any] = {}
        for mode in (CROSS_ENGINE, IN_DOMAIN):
            block = slices[(held_out, mode)]
            arms: dict[str, Any] = {}
            for name in block.arm_names:
                score = block.arm(name)
                if np.isnan(score).all():
                    continue
                summary = _arm_summary(
                    score,
                    block.harmful,
                    block.beneficial,
                    block.documents,
                    with_interval=name in HEADLINE_ARMS,
                )
                if name in thresholds:
                    tau = float(thresholds[name]["ltt_bentkus"]["tau"])
                    summary["deployed_point_certified"] = dg.deployed_point(
                        tau,
                        np.where(np.isfinite(score), score, -np.inf),
                        block.harmful,
                        block.beneficial,
                        PRIMARY_EPSILON,
                    )
                    summary["deployed_point_certified"]["tau"] = tau
                    summary["deployed_point_certified"]["controller"] = "ltt_bentkus"
                arms[name] = summary
            per_engine = {}
            if mode == IN_DOMAIN:
                for engine in sorted(set(block.frame["engine_id"].astype(str))):
                    rows = (block.frame["engine_id"].astype(str) == engine).to_numpy()
                    per_engine[engine] = {
                        "rows": int(rows.sum()),
                        "arms": {
                            name: _restricted_frontier(
                                block.arm(name)[rows],
                                block.harmful[rows],
                                block.beneficial[rows],
                            )[key]
                            for name in HEADLINE_ARMS
                            if not np.isnan(block.arm(name)).all()
                        },
                    }
            modes[mode] = {
                "rows": len(block.frame),
                "harm_base_rate": float(block.harmful.mean()),
                "beneficial_available": int(block.beneficial.sum()),
                "arms": arms,
                "per_fit_engine": per_engine,
            }

        cross = slices[(held_out, CROSS_ENGINE)]
        proposed = cross.arm(PROPOSED_ARM)
        contrasts: dict[str, Any] = {}
        for baseline in BASELINE_ARMS:
            reference = cross.arm(baseline)
            contrasts[baseline] = {
                "primary_paired_repair_recall": _paired_repair_recall_delta(
                    proposed,
                    reference,
                    cross.harmful,
                    cross.beneficial,
                    cross.documents,
                    PRIMARY_EPSILON,
                ),
                "co_primary_matched_coverage": rl._matched_coverage_contrast(
                    reference, proposed, cross.harmful, cross.beneficial, cross.documents
                ),
            }

        transfer_gap = {
            name: {
                "in_domain": modes[IN_DOMAIN]["arms"][name][key]["repair_recall"]
                if name in modes[IN_DOMAIN]["arms"]
                else float("nan"),
                "cross_engine": modes[CROSS_ENGINE]["arms"][name][key]["repair_recall"]
                if name in modes[CROSS_ENGINE]["arms"]
                else float("nan"),
            }
            for name in HEADLINE_ARMS
        }
        for cell in transfer_gap.values():
            cell["drop"] = float(cell["in_domain"] - cell["cross_engine"])

        folds[held_out] = {
            "held_out_engine": held_out,
            "train_engines": record["folds"][held_out]["train_engines"],
            "selected_lambdas": record["folds"][held_out]["selected_lambdas"],
            "modes": modes,
            "contrasts_vs_baselines": contrasts,
            "transfer_gap": transfer_gap,
        }
        print(
            f"  fold {held_out:11s} selfaware={transfer_gap['selfaware']['cross_engine']:.3f} "
            f"(in-domain {transfer_gap['selfaware']['in_domain']:.3f}) "
            f"vs harm_aware={transfer_gap['harm_aware']['cross_engine']:.3f} "
            f"delta CI=[{contrasts['harm_aware']['primary_paired_repair_recall']['ci_lower']:+.3f},"
            f"{contrasts['harm_aware']['primary_paired_repair_recall']['ci_upper']:+.3f}]"
        )

    cc._write_json_once(
        TRANSFER_RESULTS,
        {
            "schema_version": "sgv3-engine-transfer-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV3-U1",
            "mandatory_baselines": list(BASELINE_ARMS),
            "proposed_arm": PROPOSED_ARM,
            "ceiling_arms": list(CEILING_ARMS),
            "primary_endpoint": (
                "paired difference in repair recall at bounded harmful-accept risk, "
                "epsilon = " + f"{PRIMARY_EPSILON}"
            ),
            "co_primary_endpoint": "harm reduction at matched site coverage",
            "evaluation_modes": {
                CROSS_ENGINE: "the held-out engine, DEVELOPMENT documents",
                IN_DOMAIN: "the three fit engines, DEVELOPMENT documents (documents held out, "
                "engines not)",
            },
            "pooling_note": (
                "no result in this file pools engines. The in-domain block additionally "
                "reports each fit engine separately, so an in-domain number is never an "
                "average over engines that behave differently"
            ),
            "epsilon_grid": list(EPSILONS),
            "primary_epsilon": PRIMARY_EPSILON,
            "delta": DELTA,
            "bootstrap": {
                "n_resamples": BOOTSTRAP_RESAMPLES,
                "seed": pilot.BOOTSTRAP_SEED,
                "unit": "document",
            },
            "folds": folds,
            "confirmatory_accessed": False,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(f"transfer: {len(folds)} folds -> {cc._relative(TRANSFER_RESULTS)}")
    return 0


# ------------------------------------------------------------------ the ablations

# Ablation E is not a model: removing the uncertainty penalty leaves the harm-aware baseline
# exactly, and it is reported as the identity it is rather than as a fifth measurement. The
# other four each drop a block or a tag from the composite and then re-select (lambda1,
# lambda2) on the SAME inner folds, so a variant that needed a different weight is not
# recorded as a variant that carried no signal.


def run_ablation() -> int:
    started = time.monotonic()
    slices, record = load_scores()
    engines = sorted({key[0] for key in slices})
    key = f"epsilon_{int(PRIMARY_EPSILON * 100)}"
    folds: dict[str, Any] = {}

    for held_out in engines:
        block = slices[(held_out, CROSS_ENGINE)]
        full = block.arm(PROPOSED_ARM)
        variants: dict[str, Any] = {}
        for variant in VARIANTS:
            arm = f"selfaware__{variant}"
            score = block.arm(arm)
            selected = record["folds"][held_out]["selected_lambdas"][variant]
            variants[variant] = {
                "ablation": ABLATION_LABELS.get(variant, "the full signal set"),
                "signals_kept": list(VARIANTS[variant]),
                "signals_dropped": [n for n in SIGNAL_NAMES if n not in VARIANTS[variant]],
                "selected_lambda1": selected["lambda1"],
                "selected_lambda2": selected["lambda2"],
                "declined_to_use_uncertainty": bool(selected["lambda2"] == 0.0),
                "frontier": _restricted_frontier(score, block.harmful, block.beneficial),
                "repair_recall_at_primary_epsilon": float(
                    _restricted_frontier(score, block.harmful, block.beneficial)[key][
                        "repair_recall"
                    ]
                ),
                "matched_coverage_vs_full": rl._matched_coverage_contrast(
                    full, score, block.harmful, block.beneficial, block.documents
                )
                if variant != HEADLINE_VARIANT
                else None,
            }
        harm_aware = block.arm("harm_aware")
        penalty = block.arm("ablate_penalty")
        variants["no_penalty"] = {
            "ablation": ABLATION_LABELS["no_penalty"],
            "signals_kept": [],
            "signals_dropped": list(SIGNAL_NAMES),
            "selected_lambda1": record["folds"][held_out]["selected_lambdas"]["harm_aware"][
                "lambda1"
            ],
            "selected_lambda2": 0.0,
            "declined_to_use_uncertainty": True,
            "identical_to_harm_aware_baseline": bool(np.array_equal(penalty, harm_aware)),
            "identity_note": (
                "lambda2 = 0 removes the penalty term from the utility, which leaves the "
                "harm-aware baseline itself. This is an algebraic identity, asserted here and "
                "in tests/leakage rather than measured as if it could have come out otherwise"
            ),
            "frontier": _restricted_frontier(penalty, block.harmful, block.beneficial),
            "repair_recall_at_primary_epsilon": float(
                _restricted_frontier(penalty, block.harmful, block.beneficial)[key]["repair_recall"]
            ),
            "matched_coverage_vs_full": rl._matched_coverage_contrast(
                full, penalty, block.harmful, block.beneficial, block.documents
            ),
        }
        folds[held_out] = {"held_out_engine": held_out, "variants": variants}
        print(
            f"  fold {held_out:11s} "
            + " ".join(
                f"{v.split('_')[-1] if v != 'full' else 'full'}="
                f"{variants[v]['repair_recall_at_primary_epsilon']:.3f}"
                for v in ("full", "no_epistemic", "no_aleatoric", "no_visual", "no_disagreement")
            )
        )

    contribution = {
        variant: {
            "mean_repair_recall": float(
                np.mean(
                    [
                        folds[e]["variants"][variant]["repair_recall_at_primary_epsilon"]
                        for e in engines
                    ]
                )
            ),
            "engines_where_dropping_it_hurts": sorted(
                e
                for e in engines
                if folds[e]["variants"][variant]["repair_recall_at_primary_epsilon"]
                < folds[e]["variants"][HEADLINE_VARIANT]["repair_recall_at_primary_epsilon"]
            ),
            "engines_where_dropping_it_helps": sorted(
                e
                for e in engines
                if folds[e]["variants"][variant]["repair_recall_at_primary_epsilon"]
                > folds[e]["variants"][HEADLINE_VARIANT]["repair_recall_at_primary_epsilon"]
            ),
        }
        for variant in (*VARIANTS, "no_penalty")
    }

    cc._write_json_once(
        ABLATION_RESULTS,
        {
            "schema_version": "sgv3-ablation-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV3-U1",
            "reference_variant": HEADLINE_VARIANT,
            "labels": ABLATION_LABELS,
            "reselection_note": (
                "every variant re-selects (lambda1, lambda2) on the same inner "
                "leave-one-engine-out folds, so a block that needed a different weight is not "
                "recorded as a block that carried no signal"
            ),
            "primary_epsilon": PRIMARY_EPSILON,
            "contribution_summary": contribution,
            "folds": folds,
            "confirmatory_accessed": False,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(f"ablation: {len(folds)} folds -> {cc._relative(ABLATION_RESULTS)}")
    return 0


# ------------------------------------------------------------------ the finding

# The support rule is Phase 6's and SGV2's, unchanged: an arm is supported against a baseline
# only if it beats that baseline on EVERY held-out engine AND the paired interval excludes
# zero in its favour on every one. Requiring unanimity is what stops "it worked on Tesseract"
# from becoming the finding when four folds were run, and it is the rule that produced both
# preceding negatives -- not a criterion built to fit this stage's outcome.
#
# `no_correction` is on the baseline list because the brief names it, and it is beaten by
# anything with positive repair recall. It is recorded as the trivial reference it is and is
# not allowed to carry the verdict on its own.

SUBSTANTIVE_BASELINES = ("confidence_only", "harm_only", "harm_aware", "shift_aware")


def run_decide() -> int:
    """The machine-readable finding, computed from the artifacts and not from memory."""
    started = time.monotonic()
    features = cc._read_json(FEATURE_SPEC)
    calibration = cc._read_json(CALIBRATION_RESULTS)
    curves = cc._read_json(CURVE_RESULTS)
    actions = cc._read_json(ACTION_RESULTS)
    transfer = cc._read_json(TRANSFER_RESULTS)
    ablation = cc._read_json(ABLATION_RESULTS)
    selection = cc._read_json(SELECTION_RECORD)
    key = f"epsilon_{int(PRIMARY_EPSILON * 100)}"
    engines = sorted(transfer["folds"])

    def recall(engine: str, arm: str, mode: str = CROSS_ENGINE) -> float:
        arms = curves["folds"][engine]["modes"][mode]["arms"]
        if arm not in arms:
            # The oracle arm exists only cross-engine: it is chosen against the held-out
            # engine's labels, and there is no held-out engine in the in-domain block.
            return float("nan")
        return float(arms[arm]["frontier"][key]["repair_recall"])

    def contrast(engine: str, baseline: str) -> dict[str, Any]:
        return transfer["folds"][engine]["contrasts_vs_baselines"][baseline]

    support = {
        baseline: {
            "engines_where_proposed_beats_it": sorted(
                e for e in engines if recall(e, PROPOSED_ARM) > recall(e, baseline)
            ),
            "engines_with_favourable_primary_ci": sorted(
                e
                for e in engines
                if contrast(e, baseline)["primary_paired_repair_recall"]["favours_arm"]
            ),
            "engines_with_favourable_matched_coverage_ci": sorted(
                e
                for e in engines
                if contrast(e, baseline)["co_primary_matched_coverage"][
                    f"coverage_{int(PRIMARY_EPSILON * 100)}"
                ]["ci_lower"]
                > 0.0
            ),
            "per_engine_delta_repair_recall": {
                e: float(recall(e, PROPOSED_ARM) - recall(e, baseline)) for e in engines
            },
            "mean_delta_repair_recall": float(
                np.mean([recall(e, PROPOSED_ARM) - recall(e, baseline) for e in engines])
            ),
        }
        for baseline in BASELINE_ARMS
    }

    unanimous = sorted(
        baseline
        for baseline in SUBSTANTIVE_BASELINES
        if len(support[baseline]["engines_where_proposed_beats_it"]) == len(engines)
        and len(support[baseline]["engines_with_favourable_primary_ci"]) == len(engines)
    )
    supported = sorted(SUBSTANTIVE_BASELINES) == unanimous

    lambda2 = {
        e: selection["folds"][e]["selected_lambdas"][HEADLINE_VARIANT]["lambda2"] for e in engines
    }
    oracle_gap = {
        e: {
            "deployable": recall(e, PROPOSED_ARM),
            "oracle": recall(e, CEILING_ARMS[0]),
            "gap": float(recall(e, CEILING_ARMS[0]) - recall(e, PROPOSED_ARM)),
        }
        for e in engines
    }
    conformal_validity = {
        e: {
            mode: calibration["folds"][e]["modes"][mode]["conformal"][f"alpha_{PRIMARY_EPSILON:g}"]
            for mode in (SOURCE_CALIBRATION, IN_DOMAIN, CROSS_ENGINE)
        }
        for e in engines
    }
    reordering = {
        e: {
            method: {
                "accept_order_differs": calibration["folds"][e]["modes"][CROSS_ENGINE][
                    "probability_methods"
                ][method]["accept_order_differs_from_identity"],
                "inverts_strict_order": calibration["folds"][e]["modes"][CROSS_ENGINE][
                    "probability_methods"
                ][method]["inverts_a_strictly_ordered_pair"],
                "repair_recall_shift": calibration["folds"][e]["modes"][CROSS_ENGINE][
                    "probability_methods"
                ][method]["repair_recall_shift_vs_identity"],
            }
            for method in ("identity", "temperature", "isotonic", "ensemble")
        }
        for e in engines
    }
    signal_quality = {
        e: {
            name: features["folds"][e]["signals"][name]["auroc_vs_harmful"] for name in SIGNAL_NAMES
        }
        for e in engines
    }
    three_way = {
        e: actions["folds"][e]["modes"][CROSS_ENGINE]["cuts"]["uncertainty_cut_90"][
            "abstain_minus_preserve_beneficial_rate"
        ]
        for e in engines
    }
    three_way_support = {
        "engines_where_abstain_is_richer_in_repairs": sorted(
            e for e in engines if three_way[e]["ci_lower"] > 0.0
        ),
        "unanimous": bool(all(three_way[e]["ci_lower"] > 0.0 for e in engines)),
        "note": (
            "ABSTAIN and PRESERVE leave the same text behind, so this is a claim about what "
            "each set CONTAINS and about a review budget, never about measured text quality"
        ),
    }

    # Would a selection that never saw the held-out engine have preferred a different variant?
    # The ablation is read on held-out labels, so a variant that looks better there cannot
    # carry a claim. The inner means are the leakage-free version of the same question, and
    # the two are reported side by side rather than letting the held-out ranking stand alone.
    inner_variant_ranking = {}
    for e in engines:
        inner = selection["folds"][e]["inner_selection"]
        means = inner["mean_repair_recall_at_primary_epsilon"]
        at_selected = {
            variant: means[
                f"{variant}|l1_{inner['selected'][variant]['lambda1']:g}"
                f"|l2_{inner['selected'][variant]['lambda2']:g}"
            ]
            for variant in VARIANTS
        }
        inner_variant_ranking[e] = {
            "inner_mean_repair_recall_at_selected_lambdas": at_selected,
            "inner_best_variant": max(at_selected, key=lambda v: (at_selected[v], v)),
            "held_out_best_variant": max(
                VARIANTS,
                key=lambda v: (
                    ablation["folds"][e]["variants"][v]["repair_recall_at_primary_epsilon"],
                    v,
                ),
            ),
        }

    payload = {
        "schema_version": "sgv3-selfaware-decision-v1",
        "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "issued_head": pilot._git_head(),
        "hypothesis_id": "SGV3-U1",
        "hypothesis": (
            "a self-aware uncertainty model combining aleatoric, epistemic and decision "
            "uncertainty improves selective OCR correction -- more repairs captured under a "
            "bounded harmful-accept risk -- over OCR confidence alone, over the harm-aware "
            "utility model, and over shift-based rejection, on an engine held out of fitting"
        ),
        "id_note": (
            "the brief names this hypothesis H1; docs/sgv1/protocol.md binds SGV1-H1 to a "
            "different preregistered claim and states that a frozen ID is never reused, so "
            "the family restarts under this stage's own root as SGV2 did"
        ),
        "verdict": "SUPPORTED" if supported else "NOT SUPPORTED",
        "support_rule": (
            "the proposed arm must beat a baseline on EVERY held-out engine AND its paired "
            "repair-recall interval must exclude zero in its favour on every one. Unanimity "
            "across all four substantive baselines is required for SUPPORTED"
        ),
        "substantive_baselines": list(SUBSTANTIVE_BASELINES),
        "baselines_unanimously_beaten": unanimous,
        "support": support,
        "per_engine_repair_recall": {
            e: {arm: recall(e, arm) for arm in HEADLINE_ARMS} for e in engines
        },
        "in_domain_repair_recall": {
            e: {arm: recall(e, arm, IN_DOMAIN) for arm in HEADLINE_ARMS} for e in engines
        },
        "transfer_gap": {e: transfer["folds"][e]["transfer_gap"] for e in engines},
        "selected_lambda2": lambda2,
        "lambda2_sign": {
            "penalises_uncertainty": sorted(e for e in engines if lambda2[e] > 0),
            "prefers_uncertainty": sorted(e for e in engines if lambda2[e] < 0),
            "declined_to_use_it": sorted(e for e in engines if lambda2[e] == 0),
            "note": (
                "the grid is symmetric about zero, so a negative selection is a measurement "
                "that the anomalous rows were the better ones and not an artefact of a grid "
                "that could only express the hypothesis's direction"
            ),
        },
        "oracle_gap": oracle_gap,
        "conformal_validity_at_primary_epsilon": conformal_validity,
        "calibration_reordering": reordering,
        "signal_auroc_vs_harmful": signal_quality,
        "ablation_contribution": ablation["contribution_summary"],
        "three_way_split_value": three_way,
        "three_way_split_support": three_way_support,
        "inner_variant_ranking": inner_variant_ranking,
        "headline_variant": HEADLINE_VARIANT,
        "headline_variant_note": (
            "the headline variant is fixed at the full signal set before any held-out number "
            "is read. A variant that scores better on the held-out engines was chosen by "
            "looking at them and cannot carry a claim; `inner_variant_ranking` gives the "
            "leakage-free version of the same comparison"
        ),
        "confirmatory_accessed": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    cc._write_json_once(DECISION, payload)
    print(f"decide: {payload['verdict']} -> {cc._relative(DECISION)}")
    return 0


# ------------------------------------------------------------------ figures


def run_figures() -> int:
    started = time.monotonic()
    features = cc._read_json(FEATURE_SPEC)
    curves = cc._read_json(CURVE_RESULTS)
    transfer = cc._read_json(TRANSFER_RESULTS)
    ablation = cc._read_json(ABLATION_RESULTS)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    note = "SGV3 DEVELOPMENT -- not a confirmatory result"
    engines = sorted(transfer["folds"])
    key = f"epsilon_{int(PRIMARY_EPSILON * 100)}"
    written: list[Path] = []

    def finish(figure: Any, path: Path, title: str) -> None:
        figure.suptitle(f"{title}\n{note}", fontsize=9)
        figure.tight_layout()
        figure.savefig(path, dpi=140)
        plt.close(figure)
        written.append(path)

    block_colour = {"aleatoric": "#3a5f9e", "epistemic": "#c87a2b", "decision": "#2e7d5b"}

    # --- uncertainty_signal_map.png -------------------------------------------------------
    # Discrimination of each signal against the harm label on the held-out engine. 0.5 is the
    # line a signal carrying no information about harm sits on; BELOW it is not "no signal",
    # it is signal with the opposite sign, which is what SGV2 found for its shift scores.
    figure, panel = plt.subplots(figsize=(12.5, 5.6))
    width = 0.8 / len(engines)
    positions = np.arange(len(SIGNAL_NAMES))
    for offset, engine in enumerate(engines):
        values = [features["folds"][engine]["signals"][n]["auroc_vs_harmful"] for n in SIGNAL_NAMES]
        panel.bar(positions + offset * width, values, width, label=f"held out: {engine}")
    panel.axhline(0.5, color="#a33", linestyle=":", linewidth=1.2)
    panel.text(len(SIGNAL_NAMES) - 0.4, 0.505, "chance", fontsize=7, color="#a33")
    panel.set_xticks(positions + 0.4 - width / 2)
    panel.set_xticklabels(
        [n.replace("alea_", "").replace("epis_", "").replace("deci_", "") for n in SIGNAL_NAMES],
        rotation=30,
        ha="right",
        fontsize=8,
    )
    for tick, name in zip(panel.get_xticklabels(), SIGNAL_NAMES, strict=True):
        tick.set_color(block_colour[next(s.block for s in SIGNALS if s.name == name)])
    panel.set_ylabel("AUROC of the signal against the harmful label")
    panel.set_ylim(0.0, 1.0)
    panel.legend(fontsize=7, ncol=4, loc="upper left")
    finish(
        figure,
        FIGURE_DIR / "uncertainty_signal_map.png",
        "Does each uncertainty signal predict harm on an unseen engine? "
        "(label colour: blue aleatoric, orange epistemic, green decision)",
    )

    # --- coverage_risk_curve.png ----------------------------------------------------------
    figure, panels = plt.subplots(1, len(engines), figsize=(4.1 * len(engines), 4.4), sharey=True)
    drawn = ("harm_only", "harm_aware", "shift_aware", "selfaware", "oracle_selfaware")
    styles = {
        "harm_only": ("#888", "-"),
        "harm_aware": ("#3a5f9e", "-"),
        "shift_aware": ("#c87a2b", "--"),
        "selfaware": ("#2e7d5b", "-"),
        "oracle_selfaware": ("#a33", ":"),
    }
    for panel, engine in zip(np.atleast_1d(panels), engines, strict=True):
        arms = curves["folds"][engine]["modes"][CROSS_ENGINE]["arms"]
        for name in drawn:
            curve = arms[name]["curve"]
            colour, style = styles[name]
            panel.plot(
                curve["coverage"],
                curve["selective_risk"],
                style,
                color=colour,
                linewidth=1.6,
                label=name,
            )
        panel.axhline(PRIMARY_EPSILON, color="#a33", linestyle=":", linewidth=1)
        panel.set_title(f"held out: {engine}", fontsize=10)
        panel.set_xlabel("coverage (fraction of sites accepted)")
        panel.set_xlim(0.0, 1.0)
    np.atleast_1d(panels)[0].set_ylabel("selective risk (harmful accepted / accepted)")
    np.atleast_1d(panels)[0].legend(fontsize=7, loc="lower right")
    finish(
        figure,
        FIGURE_DIR / "coverage_risk_curve.png",
        "Risk against coverage on the held-out engine; the dotted line is the nominal bound",
    )

    # --- ablation_contributions.png -------------------------------------------------------
    figure, panel = plt.subplots(figsize=(12.0, 5.0))
    order = ("full", "no_epistemic", "no_aleatoric", "no_visual", "no_disagreement", "no_penalty")
    width = 0.8 / len(order)
    positions = np.arange(len(engines))
    for offset, variant in enumerate(order):
        values = [
            ablation["folds"][e]["variants"][variant]["repair_recall_at_primary_epsilon"]
            for e in engines
        ]
        panel.bar(positions + offset * width, values, width, label=variant)
    for offset, engine in enumerate(engines):
        baseline = curves["folds"][engine]["modes"][CROSS_ENGINE]["arms"]["harm_aware"]["frontier"][
            key
        ]["repair_recall"]
        panel.plot(
            [offset - 0.06, offset + 0.8],
            [baseline, baseline],
            color="#a33",
            linestyle="--",
            linewidth=1.2,
            label="harm-aware baseline" if offset == 0 else None,
        )
    panel.set_xticks(positions + 0.4 - width / 2)
    panel.set_xticklabels([f"held out: {e}" for e in engines], fontsize=9)
    panel.set_ylabel(f"repair recall at harm <= {PRIMARY_EPSILON:g}")
    panel.legend(fontsize=7, ncol=4, loc="upper left")
    panel.set_ylim(0.0, max(0.9, panel.get_ylim()[1] * 1.25))
    finish(
        figure,
        FIGURE_DIR / "ablation_contributions.png",
        "Which uncertainty block carries the effect; the dashed line is the baseline "
        "each variant has to beat",
    )

    # --- transfer_gap.png -----------------------------------------------------------------
    # The same arm measured on documents held out from engines it HAS seen, and on an engine
    # it has not. The distance between the pair is what "cross-engine transfer" costs, and it
    # is the quantity every preceding phase of this project has failed to close.
    figure, panel = plt.subplots(figsize=(11.0, 5.0))
    drawn = ("harm_only", "harm_aware", "shift_aware", "selfaware")
    width = 0.8 / len(drawn)
    for offset, name in enumerate(drawn):
        gaps = [transfer["folds"][e]["transfer_gap"][name] for e in engines]
        panel.bar(
            positions + offset * width,
            [g["in_domain"] for g in gaps],
            width,
            color="#c9d6e8",
            edgecolor="#3a5f9e",
            label="in-domain" if offset == 0 else None,
        )
        panel.bar(
            positions + offset * width,
            [g["cross_engine"] for g in gaps],
            width * 0.55,
            color="#2e7d5b",
            label="cross-engine" if offset == 0 else None,
        )
        for index in range(len(gaps)):
            # Anchored at the bottom rather than above the taller bar: an arm whose
            # cross-engine value is ~0 has no bar to label, and a label placed above the
            # in-domain bar collides with the frame wherever in-domain is near 1.
            panel.text(
                positions[index] + offset * width,
                0.02,
                name.replace("selfaware", "self"),
                fontsize=6,
                rotation=90,
                ha="center",
                va="bottom",
            )
    panel.set_xticks(positions + 0.4 - width / 2)
    panel.set_xticklabels([f"held out: {e}" for e in engines], fontsize=9)
    panel.set_ylabel(f"repair recall at harm <= {PRIMARY_EPSILON:g}")
    panel.set_ylim(0.0, 1.18)
    panel.legend(fontsize=8, loc="upper left", ncol=2)
    finish(
        figure,
        FIGURE_DIR / "transfer_gap.png",
        "In-domain against cross-engine, per arm: what holding out the engine costs",
    )

    cc._write_json_once(
        FIGURE_MANIFEST,
        {
            "schema_version": "sgv3-selfaware-figures-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV3-U1",
            "synthetic": False,
            "development_only": True,
            "sources": {
                cc._relative(path): file_sha256(path)
                for path in (
                    FEATURE_SPEC,
                    CURVE_RESULTS,
                    TRANSFER_RESULTS,
                    ABLATION_RESULTS,
                    SCORES,
                )
            },
            "figures": {cc._relative(path): file_sha256(path) for path in written},
            "confirmatory_accessed": False,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(f"figures: {len(written)} -> {cc._relative(FIGURE_DIR)}")
    return 0


# ------------------------------------------------------------------ provenance


def run_record() -> int:
    """What produced every artifact in this stage, with hashes."""
    started = time.monotonic()
    design = dg.load_design()
    produced = [
        path
        for path in (
            SCORES,
            SELECTION_RECORD,
            FEATURE_SPEC,
            CALIBRATION_RESULTS,
            CURVE_RESULTS,
            ACTION_RESULTS,
            TRANSFER_RESULTS,
            ABLATION_RESULTS,
            DECISION,
            FIGURE_MANIFEST,
        )
        if path.is_file()
    ]
    cc._write_json_once(
        FIT_RECORD,
        {
            "schema_version": "sgv3-selfaware-fit-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV3-U1",
            "representation": (
                "Phase 3's candidate-conditioned R1 design matrix, reused byte-for-byte from "
                "Phase 6 and SGV2 rather than rebuilt, so all three stages score the same rows"
            ),
            "n_features": len(design.names),
            "feature_names": list(design.names),
            "families": {
                family: [n for n in design.names if n.startswith(prefix)]
                for family, prefix in dg.FAMILIES.items()
            },
            "models": {
                "harm_head": "sklearn LogisticRegression on is_harmful",
                "outcome_head": "sklearn LogisticRegression on (harm / neutral / benefit)",
                "ensemble": f"{ENSEMBLE_MEMBERS} bagged harm heads, "
                f"{ENSEMBLE_ROW_FRACTION:g} of rows and {ENSEMBLE_FEATURE_FRACTION:g} of features",
                "random_state": pilot.FIT_SEED,
                "calibration_method": pilot.CALIBRATION_METHOD,
            },
            "uncertainty_signals": list(SIGNAL_NAMES),
            "uncertainty_variants": {v: list(m) for v, m in VARIANTS.items()},
            "selection": {
                "lambda1_grid": list(LAMBDA1_GRID),
                "lambda2_grid": list(LAMBDA2_GRID),
                "scope": "inner leave-one-engine-out over the fit engines, CALIBRATION documents",
                "engine_identity_used_by_method": False,
            },
            "endpoints": {
                "primary": "repair_recall_at_bounded_harm (imported from Phase 5)",
                "co_primary": "harm reduction at matched site coverage (imported from SGV2)",
                "epsilon_grid": list(EPSILONS),
                "primary_epsilon": PRIMARY_EPSILON,
                "delta": DELTA,
            },
            "bootstrap": {
                "n_resamples": BOOTSTRAP_RESAMPLES,
                "seed": pilot.BOOTSTRAP_SEED,
                "unit": "document",
            },
            "ground_truth_used_for_features": False,
            "ground_truth_used_for_uncertainty": False,
            "confirmatory_accessed": False,
            "inputs": {
                cc._relative(path): file_sha256(path)
                for path in (
                    pilot.CANDIDATE_TABLE,
                    pilot.LABEL_TABLE,
                    cc.FEATURES,
                    dg.DESIGN_MATRIX,
                )
            },
            "artifacts": {cc._relative(path): file_sha256(path) for path in produced},
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(f"record: {len(produced)} artifacts -> {cc._relative(FIT_RECORD)}")
    return 0


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    stages = (
        ("features", run_features),
        ("scores", run_scores),
        ("calibration", run_calibration),
        ("curves", run_curves),
        ("actions", run_actions),
        ("transfer", run_transfer),
        ("ablation", run_ablation),
        ("figures", run_figures),
        ("decide", run_decide),
        ("record", run_record),
    )
    for name, _ in stages:
        parser.add_argument(f"--{name}", action="store_true")
    args = parser.parse_args()
    selected = [function for name, function in stages if getattr(args, name)]
    if not selected:
        parser.print_help()
        return 2
    for stage in selected:
        code = stage()
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
