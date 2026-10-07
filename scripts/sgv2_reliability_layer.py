#!/usr/bin/env python3
"""SGV2 development phase 1: can a reliability layer tell when the risk estimate is invalid?

Phase 6 (`docs/sgv1/domain_generalized_risk_calibration.md`) ended on a negative sharper
than "it did not transfer": the harm ranking itself does not survive an unseen engine, no
monotone recalibration can move the achievable frontier, and the certified threshold missed
its nominal bound on one engine of four. This stage stops trying to make the estimate
transfer and asks whether its *failure* is detectable before an edit is applied.

**The hypothesis is recorded as SGV2-R1, not H5.** `docs/sgv1/protocol.md` binds SGV1-H1
through SGV1-H4 and says a frozen ID is never reused for a different claim. The brief for
this stage names its hypothesis H5; its artifacts live under an `sgv2` root, so the family
restarts rather than extending a frozen sequence.

    SGV2-R1: a shift-aware abstention mechanism preserves bounded-risk OCR correction on an
    unseen engine by detecting unreliable conditions before the correction is applied.

Four things decide what the numbers below can mean.

**1. Detecting the engine is easy, and easy is not evidence.** Phase 6 measured a supervised
domain classifier between engines at AUC 0.867-0.999. A high detection AUROC here therefore
re-confirms a measurement already in hand and settles nothing about safety. The scientific
content is whether a detector's score predicts harm *within* the held-out engine, and
whether acting on it buys anything at matched repair coverage.

**2. An abstention rule earns nothing unless it reorders.** Repair recall under a harm bound
is a property of a ranking. Any abstention rule that is a monotone function of the risk
score alone traces the risk-only risk-coverage curve exactly -- Phase 6's monotone-invariance
argument, applied to abstention instead of calibration. A shift signal can help only if
combining it with the risk score produces a different order. Every arm here is therefore
compared against risk-only at MATCHED coverage, never against "before abstention" at
whatever coverage it happened to land on.

**3. Abstaining is not being safe.** A policy that accepts nothing has a harm rate of zero
and repairs nothing. Every cell reports coverage and repair recall beside the harm rate, and
a policy that abstains everywhere is recorded as abstaining rather than as safe. Phase 6 had
to repair exactly this defect in its label-budget curve.

**4. CORRECT / ABSTAIN / PRESERVE is a two-way decision in this benchmark.** ABSTAIN (route
to a human) and PRESERVE (keep the OCR output) both leave `O` in place, so both give
`d_after = d_before` and neither is harmful nor beneficial under `edits/outcome.py`. There is
no human-review channel in the benchmark, so the third arm is not measurable here. What is
measurable is an upper bound on what a perfect reviewer could recover from the abstained set,
and it is reported as a bound under a stated assumption, never as a result.

    --shift        experiment 1: label-free detection of an unseen engine
    --reliability  experiment 2: does the shift score predict harmful corrections?
    --abstain      experiment 3: shift-aware abstention at matched coverage
    --transfer     experiment 4: selective transfer, and the certified bound
    --failure      experiment 5: where detection succeeds and where it fails
    --figures      the three required figures
    --decide       the machine-readable finding

DEVELOPMENT ONLY. The CONFIRMATORY reserve stays locked and is absent from every artifact.
Every fold holds out an engine AND holds out documents; neither axis is relaxed anywhere.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sgv1_candidate_conditioned as cc
import sgv1_domain_generalization as dg
import sgv1_risk_policy as policy
import sgv1_verifier_pilot as pilot
from ocr_risk.calibrate.calibrators import build_calibrator
from ocr_risk.io.hashing import file_sha256
from ocr_risk.metrics.discrimination import roc_auc
from ocr_risk.risk.controller import CONTROLLERS, select_threshold
from ocr_risk.stats.bootstrap import cluster_bootstrap_indices

REPO = pilot.REPO
OUT = REPO / "results/generated/sgv2/reliability_layer"
SHIFT_RESULTS = OUT / "shift_detection_results.json"
RELIABILITY_RESULTS = OUT / "reliability_analysis.json"
ABSTENTION_RESULTS = OUT / "abstention_results.json"
TRANSFER_SAFETY = OUT / "transfer_safety_results.json"
FAILURE_RESULTS = OUT / "failure_mode_results.json"
DECISION = OUT / "research_decision.json"
FIT_RECORD = OUT / "fit_record.json"
SCORES = OUT / "reliability_scores.parquet"
SELECTION_RECORD = OUT / "selection_record.json"
FIGURE_DIR = OUT / "figures"
FIGURE_MANIFEST = OUT / "figure_manifest.json"

EPSILONS = dg.EPSILONS
PRIMARY_EPSILON = dg.PRIMARY_EPSILON
DELTA = dg.DELTA
BOOTSTRAP_RESAMPLES = dg.BOOTSTRAP_RESAMPLES

# The detector families. `mahalanobis` and the two learned detectors are COLD START: they see
# only the fit engines and can flag the very first page from an engine nobody has sampled.
# `domain_classifier` is WARM START: it needs a batch of unlabeled pages from the new engine
# before it can score anything. Both are ground-truth-free -- which engine produced a row is
# always known -- so the distinction is about what an operator has on day one, not about
# labels.
COLD_START_DETECTORS = (
    "mahalanobis",
    "mahalanobis_nearest_mode",
    "isolation_forest",
    "one_class_svm",
)
WARM_START_DETECTORS = ("domain_classifier",)
DETECTORS = COLD_START_DETECTORS + WARM_START_DETECTORS

MAHALANOBIS_RIDGE = 1e-3
FOREST_TREES = 150
ONE_CLASS_NU = 0.1
ONE_CLASS_SUBSAMPLE = 4000
# The kappa grid is SYMMETRIC about zero on purpose. SGV2-R1 assumes anomalous rows are the
# dangerous ones, so a grid of non-negative kappas can only ever test the direction the
# hypothesis assumes -- and if the association runs the other way, such a grid cannot express
# the better policy and the ceiling it reports is not a ceiling. Negative kappa means "prefer
# the anomalous rows"; zero means the selection declined to use the shift signal at all.
KAPPA_GRID = (
    -3.20,
    -1.60,
    -0.80,
    -0.40,
    -0.20,
    -0.10,
    -0.05,
    0.0,
    0.05,
    0.10,
    0.20,
    0.40,
    0.80,
    1.60,
    3.20,
)
REJECT_FRACTIONS = (0.10, 0.20, 0.30, 0.50)
# Symmetric by construction, for the same reason the kappa grid is: a family of cuts that
# can only remove the anomalous end can only test the direction SGV2-R1 assumes.
INDUCTIVE_UPPER = (0.90, 0.95, 0.99)
INDUCTIVE_LOWER = (0.10, 0.05, 0.01)
MATCHED_RECALLS = (0.05, 0.10, 0.20, 0.30, 0.50)
MATCHED_COVERAGES = (0.05, 0.10, 0.20, 0.40)
SHIFT_QUANTILE_BINS = 4
# Rejection is carried as -inf, which is what `np.isfinite` tests for and therefore what
# distinguishes "this policy refused to consider the row" from "this policy scored it very
# low". The two are different events and an abstention rate that cannot tell them apart is
# the defect Phase 6 had to repair in its label-budget curve. REJECTED_SCORE is the finite
# stand-in used only where an AUC or a cumulative mean would otherwise see a non-finite
# value; it never appears in a stored score.
REJECTED_SCORE = -1e18
SATURATION_THRESHOLD = 0.99
NULL_SPLITS = dg.NULL_SPLITS


class PhaseError(RuntimeError):
    """A freeze, role, split, or selection invariant failed."""


# The primary endpoint has exactly one implementation in this repository and it is Phase 5's.
# Phase 6 imported it rather than restating it; so does this stage, and `tests/leakage`
# asserts all three names are the same function object.
achievable_repair_recall = dg.achievable_repair_recall


# ------------------------------------------------------------------ folds


@dataclass(slots=True)
class Fold:
    """One leave-one-engine-out fold, with both split axes resolved once.

    Holding the four index blocks in one object is what makes the disjointness assertion a
    single place rather than a line repeated in five experiments -- Phase 6 repeated it and
    the repetition is how a fold ends up guarded in four stages and not the fifth.
    """

    held_out: str
    train_engines: tuple[str, ...]
    fit: np.ndarray
    """Fit engines, TRAIN documents. Everything that learns is fitted here."""
    source_cal: np.ndarray
    """Fit engines, CALIBRATION documents. Frozen transformers and thresholds come from here."""
    seen_eval: np.ndarray
    """Fit engines, DEVELOPMENT documents. The detection negatives."""
    eval: np.ndarray
    """Held-out engine, DEVELOPMENT documents. The only rows any headline is read from."""
    unlabeled_target: np.ndarray
    """Held-out engine, TRAIN documents -- features only, used by warm-start detection."""


def build_fold(design: dg.Design, held_out: str) -> Fold:
    role = design.meta["role"].to_numpy(str)
    engine = design.meta["engine_id"].to_numpy(str)
    fold = Fold(
        held_out=held_out,
        train_engines=tuple(e for e in design.engines if e != held_out),
        fit=np.flatnonzero((role == "TRAIN") & (engine != held_out)),
        source_cal=np.flatnonzero((role == "CALIBRATION") & (engine != held_out)),
        seen_eval=np.flatnonzero((role == "DEVELOPMENT") & (engine != held_out)),
        eval=np.flatnonzero((role == "DEVELOPMENT") & (engine == held_out)),
        unlabeled_target=np.flatnonzero((role == "TRAIN") & (engine == held_out)),
    )
    documents = design.documents
    evaluation_documents = set(documents[fold.eval].tolist())
    for name, block in (
        ("fit", fold.fit),
        ("source_cal", fold.source_cal),
        ("unlabeled_target", fold.unlabeled_target),
    ):
        if set(documents[block].tolist()) & evaluation_documents:
            raise PhaseError(f"{held_out}: {name} shares documents with the evaluation rows")
    for name, block in (("fit", fold.fit), ("source_cal", fold.source_cal)):
        if held_out in set(engine[block].tolist()):
            raise PhaseError(f"{held_out}: the held-out engine reached {name}")
    if not set(documents[fold.seen_eval].tolist()) & evaluation_documents:
        raise PhaseError(f"{held_out}: no development document is read by both sides")
    return fold


def matched_page_detection(design: dg.Design, fold: Fold) -> tuple[np.ndarray, np.ndarray]:
    """Detection rows restricted to pages BOTH sides read, plus the positive mask.

    Engines do not all propose a candidate on every page -- Tesseract reaches 110 of the 133
    development documents where docTR reaches all of them -- so the raw positive and negative
    blocks sit on overlapping but unequal page sets. Left alone, a detection AUROC would be
    part engine separation and part "this page has candidates at all". Intersecting the two
    page sets removes that channel; how many pages it costs is recorded per fold.
    """
    documents = design.documents
    shared = set(documents[fold.eval].tolist()) & set(documents[fold.seen_eval].tolist())
    positives = fold.eval[np.isin(documents[fold.eval], list(shared))]
    negatives = fold.seen_eval[np.isin(documents[fold.seen_eval], list(shared))]
    index = np.concatenate([negatives, positives])
    positive = np.concatenate(
        [np.zeros(negatives.size, dtype=bool), np.ones(positives.size, dtype=bool)]
    )
    return index, positive


# ------------------------------------------------------------------ the shift detector


@dataclass(slots=True)
class ShiftDetector:
    """A fitted density/novelty model plus the frozen map from its score to a percentile.

    `reference` is the sorted score distribution on the fit engines' CALIBRATION rows. It is
    a **fitted transformer** in the sense `.claude/rules/experiment-leakage.md` uses: its
    quantiles come from fit-role data and are frozen before any evaluation row is scored.
    """

    kind: str
    scaler: Any
    model: Any
    columns: tuple[int, ...]
    reference: np.ndarray

    def raw(self, design: dg.Design, index: np.ndarray) -> np.ndarray:
        """Higher is more unlike the fit distribution, whatever the detector's own sign."""
        block = self.scaler.transform(design.matrix[np.ix_(index, list(self.columns))])
        if self.kind in ("mahalanobis", "mahalanobis_nearest_mode"):
            # One Gaussian for the pooled fit engines, or one per engine with the distance to
            # the NEAREST mode. The pair is the test of whether a pooled density is failing
            # because the fit distribution is a mixture: if it is, the per-mode form recovers.
            distances = [
                np.sqrt(
                    np.maximum(
                        np.einsum("ij,jk,ik->i", block - centre, inverse, block - centre), 0.0
                    )
                )
                for centre, inverse in self.model
            ]
            return np.min(np.vstack(distances), axis=0)
        if self.kind == "domain_classifier":
            classes = list(self.model.classes_)
            return np.asarray(self.model.predict_proba(block)[:, classes.index(1)], dtype=float)
        return -np.asarray(self.model.score_samples(block), dtype=float)

    def percentile(self, raw: np.ndarray) -> np.ndarray:
        """Where a score falls in the fit distribution: 1.0 means "past everything seen".

        The percentile is the unit every downstream arm consumes, because a Mahalanobis
        distance and an isolation-forest score are not on comparable axes and a single kappa
        could not mean the same thing for both.
        """
        return np.searchsorted(self.reference, raw, side="right") / float(self.reference.size)


def fit_shift_detector(
    design: dg.Design,
    fold: Fold,
    kind: str,
    columns: list[int] | None = None,
) -> ShiftDetector:
    """Fit one detector on the fold's fit rows and freeze its percentile map.

    Only `domain_classifier` sees any row from the held-out engine, and only its *features*
    on TRAIN documents -- never a label, never a page that will be scored. That is the
    warm-start assumption made explicit: an operator who has run the new engine over a batch
    of pages has this, and an operator meeting the engine for the first time does not.
    """
    from sklearn.ensemble import IsolationForest
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import OneClassSVM

    used = list(range(len(design.names))) if columns is None else list(columns)
    block = design.matrix[np.ix_(fold.fit, used)]
    scaler = StandardScaler().fit(block)
    scaled = scaler.transform(block)

    def gaussian(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        centre = rows.mean(axis=0)
        covariance = np.cov(rows - centre, rowvar=False) + MAHALANOBIS_RIDGE * np.eye(len(used))
        return centre, np.linalg.pinv(covariance)

    model: Any
    if kind == "mahalanobis":
        model = [gaussian(scaled)]
    elif kind == "mahalanobis_nearest_mode":
        engines = design.meta["engine_id"].to_numpy(str)[fold.fit]
        model = [gaussian(scaled[engines == e]) for e in sorted(set(engines.tolist()))]
    elif kind == "isolation_forest":
        model = IsolationForest(
            n_estimators=FOREST_TREES, random_state=pilot.FIT_SEED, n_jobs=1
        ).fit(scaled)
    elif kind == "one_class_svm":
        # A kernel one-class SVM is quadratic in the ~30k fit rows, so it is fitted on a
        # seeded subsample. The linear stochastic solver scales but was measured degenerate
        # here -- score spread of ~1e-3 and AUROC at chance on every fold -- and reporting a
        # method as failing when the configuration failed would be the wrong finding.
        rng = np.random.default_rng(pilot.FIT_SEED)
        subsample = rng.choice(
            len(scaled), size=min(ONE_CLASS_SUBSAMPLE, len(scaled)), replace=False
        )
        model = OneClassSVM(kernel="rbf", nu=ONE_CLASS_NU, gamma="scale").fit(scaled[subsample])
    elif kind == "domain_classifier":
        target = design.matrix[np.ix_(fold.unlabeled_target, used)]
        features = np.vstack([scaled, scaler.transform(target)])
        labels = np.concatenate(
            [np.zeros(len(fold.fit), dtype=int), np.ones(len(fold.unlabeled_target), dtype=int)]
        )
        model = LogisticRegression(
            C=1.0, max_iter=2000, class_weight="balanced", random_state=pilot.FIT_SEED
        ).fit(features, labels)
    else:
        raise PhaseError(f"unknown detector {kind!r}")

    detector = ShiftDetector(
        kind=kind, scaler=scaler, model=model, columns=tuple(used), reference=np.zeros(1)
    )
    detector.reference = np.sort(detector.raw(design, fold.source_cal))
    return detector


# ------------------------------------------------------------------ endpoints

# Two endpoints are reported side by side, because the brief names one and this project
# already owns the other, and they answer different questions:
#
#   repair recall at bounded harm  -- "how many repairs can this policy capture while its
#                                     realized harm stays under epsilon?" One definition,
#                                     imported from Phase 5.
#   harm at matched coverage       -- "at the SAME volume of accepted edits, is this policy
#                                     safer than risk-only?" The brief's primary. It is the
#                                     one that cannot be won by abstaining, because the
#                                     abstention is held fixed on both sides.
#
# A third framing -- harm at matched repair RECALL -- is reported as a point estimate only.
# An arm that rejects half its rows can be unable to reach a target recall at any threshold,
# and a bootstrap over a statistic that is undefined on some draws does not have an interval;
# saying so beats substituting a number for the draws where it does not exist.


def _ranked(score: np.ndarray) -> np.ndarray:
    """Row indices in accept order. A rejected row carries -inf and is never accepted."""
    eligible = np.flatnonzero(np.isfinite(score))
    return eligible[np.argsort(-score[eligible], kind="stable")]


def harm_at_matched_coverage(
    score: np.ndarray, harmful: np.ndarray, beneficial: np.ndarray, coverage: float
) -> dict[str, Any]:
    """Accept exactly ``coverage`` of all sites, best-first, and report what that costs."""
    order = _ranked(score)
    wanted = round(coverage * score.size)
    k = min(wanted, int(order.size))
    if k == 0:
        return {
            "requested_coverage": coverage,
            "n_accepted": 0,
            "coverage": 0.0,
            "coverage_attained": wanted == 0,
            "repair_recall": 0.0,
            "realized_harm_rate": float("nan"),
        }
    accepted = order[:k]
    return {
        "requested_coverage": coverage,
        "n_accepted": int(k),
        "coverage": float(k / score.size),
        "coverage_attained": bool(k == wanted),
        "repair_recall": float(beneficial[accepted].sum() / max(int(beneficial.sum()), 1)),
        "realized_harm_rate": float(harmful[accepted].mean()),
    }


def harm_at_matched_repair_recall(
    score: np.ndarray, harmful: np.ndarray, beneficial: np.ndarray, target: float
) -> dict[str, Any]:
    """Smallest accept set reaching ``target`` repair recall, and the harm it carries."""
    total = int(beneficial.sum())
    order = _ranked(score)
    if total == 0 or order.size == 0:
        return {"requested_repair_recall": target, "attained": False, "reason": "no support"}
    cumulative = np.cumsum(beneficial[order])
    hit = np.flatnonzero(cumulative >= target * total)
    if hit.size == 0:
        return {
            "requested_repair_recall": target,
            "attained": False,
            "reason": "unreachable at full coverage of this policy's eligible rows",
            "max_repair_recall": float(cumulative[-1] / total),
        }
    k = int(hit[0]) + 1
    accepted = order[:k]
    return {
        "requested_repair_recall": target,
        "attained": True,
        "n_accepted": k,
        "coverage": float(k / score.size),
        "repair_recall": float(cumulative[k - 1] / total),
        "realized_harm_rate": float(harmful[accepted].mean()),
    }


def _harm_at_coverage_statistic(
    score_a: np.ndarray,
    score_b: np.ndarray,
    harmful: np.ndarray,
    beneficial: np.ndarray,
    coverage: float,
) -> Any:
    """Paired by construction: both arms are re-ranked inside the SAME resampled rows."""

    def statistic(index: np.ndarray) -> float:
        if index.size == 0:
            return float("nan")
        left = harm_at_matched_coverage(score_a[index], harmful[index], beneficial[index], coverage)
        right = harm_at_matched_coverage(
            score_b[index], harmful[index], beneficial[index], coverage
        )
        if left["n_accepted"] == 0 or right["n_accepted"] == 0:
            return float("nan")
        return float(left["realized_harm_rate"] - right["realized_harm_rate"])

    return statistic


def arm_summary(
    score: np.ndarray,
    harmful: np.ndarray,
    beneficial: np.ndarray,
    documents: np.ndarray,
    *,
    with_interval: bool = True,
) -> dict[str, Any]:
    """Everything one acceptance rule delivers on one fold, abstention reported as abstention."""
    eligible = int(np.count_nonzero(np.isfinite(score)))
    ranked = np.where(np.isfinite(score), score, REJECTED_SCORE)
    summary: dict[str, Any] = dg.frontier(ranked, harmful, beneficial)
    summary["eligible_rows"] = eligible
    summary["rejected_rows"] = int(score.size - eligible)
    summary["abstention_rate_before_threshold"] = float(1.0 - eligible / score.size)
    summary["matched_coverage"] = {
        f"coverage_{int(c * 100)}": harm_at_matched_coverage(score, harmful, beneficial, c)
        for c in MATCHED_COVERAGES
    }
    summary["matched_repair_recall"] = {
        f"recall_{int(r * 100)}": harm_at_matched_repair_recall(score, harmful, beneficial, r)
        for r in MATCHED_RECALLS
    }
    if with_interval:
        summary["interval_primary_epsilon"] = dg.clustered_interval(
            ranked, harmful, beneficial, documents, PRIMARY_EPSILON
        )
    return summary


# ------------------------------------------------------------------ selection, kept inside


@dataclass(slots=True)
class Selection:
    """Which detector and how hard to lean on it -- decided without the held-out engine."""

    detector: str
    kappa: float
    inner: dict[str, Any]


def _inner_fold(design: dg.Design, outer: Fold, inner_held: str) -> Fold:
    """A fold over the fit engines only, on CALIBRATION documents.

    Two properties this has to have and that a careless construction loses. The outer
    held-out engine appears nowhere, so nothing selected here can have been chosen for
    performing well on it. And the CALIBRATION documents are split in half, so the frozen
    percentile reference and the inner evaluation rows do not share a page -- without that
    split they would sit on the same 133 documents and the selection would be reading a
    reference fitted on the pages it is scoring.
    """
    role = design.meta["role"].to_numpy(str)
    engine = design.meta["engine_id"].to_numpy(str)
    inner_train = tuple(e for e in outer.train_engines if e != inner_held)
    calibration_documents = sorted(set(design.documents[role == "CALIBRATION"].tolist()))
    reference_half = set(calibration_documents[::2])
    evaluation_half = set(calibration_documents[1::2])
    in_reference = np.isin(design.documents, list(reference_half))
    in_evaluation = np.isin(design.documents, list(evaluation_half))
    fold = Fold(
        held_out=inner_held,
        train_engines=inner_train,
        fit=np.flatnonzero((role == "TRAIN") & np.isin(engine, list(inner_train))),
        source_cal=np.flatnonzero(
            (role == "CALIBRATION") & np.isin(engine, list(inner_train)) & in_reference
        ),
        seen_eval=np.flatnonzero(
            (role == "CALIBRATION") & np.isin(engine, list(inner_train)) & in_evaluation
        ),
        eval=np.flatnonzero((role == "CALIBRATION") & (engine == inner_held) & in_evaluation),
        unlabeled_target=np.flatnonzero((role == "TRAIN") & (engine == inner_held)),
    )
    for block in (fold.fit, fold.source_cal, fold.eval, fold.unlabeled_target):
        if outer.held_out in set(engine[block].tolist()):
            raise PhaseError(f"inner selection for {outer.held_out} touched the held-out engine")
    if set(design.documents[fold.source_cal].tolist()) & set(design.documents[fold.eval].tolist()):
        raise PhaseError("inner reference and inner evaluation share a document")
    return fold


def select_policy(design: dg.Design, outer: Fold) -> Selection:
    """Pick (detector, kappa) by an inner leave-one-engine-out over the fit engines.

    kappa = 0 is on the grid on purpose: the procedure is allowed to decide that the shift
    signal is not worth acting on, and a selection that cannot decline is not a selection.
    """
    every = design.columns(tuple(dg.FAMILIES))
    grid: dict[str, dict[str, list[float]]] = {
        kind: {f"kappa_{k:g}": [] for k in KAPPA_GRID} for kind in DETECTORS
    }
    for inner_held in outer.train_engines:
        fold = _inner_fold(design, outer, inner_held)
        model = dg.fit_harm_model(design, fold.fit, every)
        safety = model.safety(design, fold.eval)
        harmful = design.harmful[fold.eval]
        beneficial = design.beneficial[fold.eval]
        for kind in DETECTORS:
            detector = fit_shift_detector(design, fold, kind)
            shift = detector.percentile(detector.raw(design, fold.eval))
            for kappa in KAPPA_GRID:
                measured = achievable_repair_recall(safety - kappa * shift, harmful, beneficial)
                grid[kind][f"kappa_{kappa:g}"].append(
                    float(measured[f"epsilon_{int(PRIMARY_EPSILON * 100)}"]["repair_recall"])
                )

    means = {
        (kind, kappa): float(np.mean(grid[kind][f"kappa_{kappa:g}"]))
        for kind in DETECTORS
        for kappa in KAPPA_GRID
    }
    best = max(means, key=lambda key: (means[key], -abs(key[1]), key[0]))
    return Selection(
        detector=best[0],
        kappa=best[1],
        inner={
            "inner_engines": list(outer.train_engines),
            "inner_evaluation_role": "CALIBRATION",
            "kappa_grid": list(KAPPA_GRID),
            "detectors": list(DETECTORS),
            "mean_repair_recall_at_primary_epsilon": {
                f"{kind}|kappa_{kappa:g}": means[(kind, kappa)]
                for kind in DETECTORS
                for kappa in KAPPA_GRID
            },
            "per_inner_fold": {
                f"{kind}|kappa_{kappa:g}": grid[kind][f"kappa_{kappa:g}"]
                for kind in DETECTORS
                for kappa in KAPPA_GRID
            },
            "selected": {"detector": best[0], "kappa": best[1], "score": means[best]},
            "kappa_at_grid_boundary": bool(best[1] in (KAPPA_GRID[0], KAPPA_GRID[-1])),
            "boundary_note": (
                "A kappa selected at the edge of the grid means the grid, not the data, chose "
                "the value. The grid is wide enough that the extremes are close to ranking by "
                "the shift score alone, which is measured separately as the shift_only arms, "
                "so a boundary selection is bracketed rather than open-ended -- but it is "
                "flagged, not smoothed over."
            ),
            "tie_break": "highest mean, then smallest |kappa|, then detector name",
        },
    )


# ------------------------------------------------------------------ the acceptance rules


def _batch_rank(values: np.ndarray) -> np.ndarray:
    """Position of each row inside its own evaluation batch, in [0, 1].

    This is TRANSDUCTIVE and label-free: it needs the whole batch of pages from the new
    engine before it can score any one of them, and it needs no ground truth ever. That is a
    real deployment mode -- an operator re-processing an archive has the batch -- and it is a
    different mode from the frozen percentile, which scores the first page in isolation. The
    two are never pooled into one arm.
    """
    from scipy.stats import rankdata

    return np.asarray(rankdata(values, method="average"), dtype=float) / float(values.size)


def build_arms(
    design: dg.Design,
    fold: Fold,
    selection: Selection,
    detectors: dict[str, ShiftDetector],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Every acceptance rule as a score vector on the same rows, rejection encoded as -inf.

    Expressing abstention as a score rather than as a separate code path is what lets the
    matched-coverage comparison be honest: an arm that rejects rows and an arm that does not
    are pushed through exactly the same ranking, threshold and endpoint code.
    """
    every = design.columns(tuple(dg.FAMILIES))
    model = dg.fit_harm_model(design, fold.fit, every)
    safety = model.safety(design, fold.eval)
    harmful = design.harmful[fold.eval]
    beneficial = design.beneficial[fold.eval]

    chosen = detectors[selection.detector]
    raw = chosen.raw(design, fold.eval)
    inductive = chosen.percentile(raw)
    batch = _batch_rank(raw)

    # Both directions of the shift score alone. One of them is the hypothesis's direction and
    # the other is its mirror; running only the first would measure an assumption.
    arms: dict[str, np.ndarray] = {
        "risk_only": safety,
        "shift_only_typical_first": -inductive,
        "shift_only_anomalous_first": inductive,
    }
    diagnostics: dict[str, Any] = {
        "selected_detector": selection.detector,
        "selected_kappa": selection.kappa,
        "saturation_fraction": float(np.mean(inductive >= SATURATION_THRESHOLD)),
        "inductive_percentile_iqr": float(np.subtract(*np.percentile(inductive, [75, 25]))),
        "inductive_percentile_mean": float(inductive.mean()),
        "rejection": {},
        "saturation_note": (
            "The frozen percentile saturates when nearly every row of the unseen engine "
            "scores past everything the fit engines produced. A saturated percentile is "
            "nearly constant on the evaluation batch, and subtracting a near-constant from "
            "the safety score cannot reorder it -- so a high saturation fraction predicts, "
            "mechanically, that the inductive arms will not move the frontier."
        ),
    }

    for fraction in REJECT_FRACTIONS:
        for end, keep in (
            ("top", batch < float(np.quantile(batch, 1.0 - fraction))),
            ("bottom", batch > float(np.quantile(batch, fraction))),
        ):
            name = f"reject_{end}{int(fraction * 100)}_batch"
            arms[name] = np.where(keep, safety, -np.inf)
            diagnostics["rejection"][name] = {
                "mode": "batch quantile (transductive, label-free)",
                "end_rejected": (
                    "the most anomalous rows -- the hypothesis's direction"
                    if end == "top"
                    else "the most typical rows -- the mirror, measured rather than assumed"
                ),
                "requested_rejection_fraction": fraction,
                "realized_rejection_fraction": float(1.0 - keep.mean()),
            }

    for quantile in INDUCTIVE_UPPER:
        keep = inductive <= quantile
        name = f"reject_above_p{int(quantile * 100)}_inductive"
        arms[name] = np.where(keep, safety, -np.inf)
        diagnostics["rejection"][name] = {
            "mode": "frozen fit percentile (inductive)",
            "threshold_percentile": quantile,
            "end_rejected": "the most anomalous rows -- the hypothesis's direction",
            "realized_rejection_fraction": float(1.0 - keep.mean()),
        }
    for quantile in INDUCTIVE_LOWER:
        keep = inductive >= quantile
        name = f"reject_below_p{int(quantile * 100)}_inductive"
        arms[name] = np.where(keep, safety, -np.inf)
        diagnostics["rejection"][name] = {
            "mode": "frozen fit percentile (inductive)",
            "threshold_percentile": quantile,
            "end_rejected": "the most typical rows -- the mirror",
            "realized_rejection_fraction": float(1.0 - keep.mean()),
        }

    arms["adaptive_inductive"] = safety - selection.kappa * inductive
    arms["adaptive_batch"] = safety - selection.kappa * batch

    warm = detectors["domain_classifier"]
    warm_batch = _batch_rank(warm.raw(design, fold.eval))
    arms["warm_start_domain_batch"] = safety - selection.kappa * warm_batch

    # Non-deployable ceilings. Both are chosen against the held-out engine's own labels, so
    # neither is a result: they bound how much of any shortfall is the SELECTION of kappa and
    # of the detector rather than the shift signal itself.
    key = f"epsilon_{int(PRIMARY_EPSILON * 100)}"
    oracle: dict[str, float] = {}
    for kind, detector in detectors.items():
        candidate_batch = _batch_rank(detector.raw(design, fold.eval))
        for kappa in KAPPA_GRID:
            measured = achievable_repair_recall(
                safety - kappa * candidate_batch, harmful, beneficial
            )
            oracle[f"{kind}|kappa_{kappa:g}"] = float(measured[key]["repair_recall"])
    best = max(oracle, key=lambda name: oracle[name])
    best_kind, best_kappa = best.split("|kappa_")
    arms["oracle_selected_batch"] = safety - float(best_kappa) * _batch_rank(
        detectors[best_kind].raw(design, fold.eval)
    )
    diagnostics["oracle_selection"] = {
        "note": (
            "chosen on the held-out engine's own labels; a ceiling on what perfect selection "
            "of (detector, kappa) could deliver, never an arm an operator could run"
        ),
        "grid": oracle,
        "selected": best,
        "selected_repair_recall": oracle[best],
        "deployable_selection_repair_recall": oracle.get(
            f"{selection.detector}|kappa_{selection.kappa:g}", float("nan")
        ),
    }
    return arms, diagnostics


# ------------------------------------------------------------------ experiment 1: detection


def _auc_interval(
    score: np.ndarray, positive: np.ndarray, documents: np.ndarray
) -> dict[str, float]:
    def statistic(index: np.ndarray) -> float:
        labels = positive[index]
        if labels.all() or not labels.any():
            return float("nan")
        return float(roc_auc(score[index], labels.astype(float)))

    result = cluster_bootstrap_indices(
        list(documents),
        statistic,
        n_resamples=BOOTSTRAP_RESAMPLES,
        seed=pilot.BOOTSTRAP_SEED,
    )
    return {"estimate": result.estimate, "ci_lower": result.lower, "ci_upper": result.upper}


def _document_level(
    design: dg.Design, index: np.ndarray, score: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean detector score per (document, engine): the unit an operator actually gates on."""
    frame = pd.DataFrame(
        {
            "document_id": design.documents[index],
            "engine_id": design.meta["engine_id"].to_numpy(str)[index],
            "score": score,
        }
    )
    grouped = frame.groupby(["document_id", "engine_id"], sort=True)["score"].mean().reset_index()
    return (
        grouped["score"].to_numpy(dtype=float),
        grouped["document_id"].to_numpy(str),
        grouped["engine_id"].to_numpy(str),
    )


def _within_engine_null(design: dg.Design, detector: ShiftDetector, fold: Fold) -> dict[str, float]:
    """How much separation appears between two random halves of the SAME engines' pages.

    Without this the detection AUROC has no scale. A detector that scores unseen documents
    as anomalous regardless of engine would look like an engine detector, and the only way to
    tell the difference is to ask it to separate pages that differ by document alone.
    """
    documents = design.documents[fold.seen_eval]
    score = detector.raw(design, fold.seen_eval)
    unique = np.array(sorted(set(documents.tolist())))
    values: list[float] = []
    for split in range(NULL_SPLITS):
        order = np.random.default_rng(pilot.BOOTSTRAP_SEED + split).permutation(unique.size)
        left = set(unique[order[: unique.size // 2]].tolist())
        positive = np.isin(documents, list(left))
        if positive.all() or not positive.any():
            continue
        values.append(float(roc_auc(score, positive.astype(float))))
    array = np.array(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
        "n_splits": int(array.size),
    }


def run_shift() -> int:
    """Experiment 1: can an unseen OCR engine be identified without ground truth?"""
    started = time.monotonic()
    design = dg.load_design()
    folds: dict[str, Any] = {}

    for held_out in design.engines:
        fold = build_fold(design, held_out)
        detection_index, positive = matched_page_detection(design, fold)
        documents = design.documents[detection_index]

        per_detector: dict[str, Any] = {}
        for kind in DETECTORS:
            detector = fit_shift_detector(design, fold, kind)
            raw = detector.raw(design, detection_index)
            doc_score, doc_documents, doc_engine = _document_level(design, detection_index, raw)
            doc_positive = doc_engine == held_out
            inductive = detector.percentile(detector.raw(design, fold.eval))
            per_detector[kind] = {
                "start": "warm" if kind in WARM_START_DETECTORS else "cold",
                "row_level_auroc": _auc_interval(raw, positive, documents),
                "document_level_auroc": _auc_interval(doc_score, doc_positive, doc_documents),
                "within_engine_null": _within_engine_null(design, detector, fold),
                "evaluation_percentile": {
                    "mean": float(inductive.mean()),
                    "median": float(np.median(inductive)),
                    "iqr": float(np.subtract(*np.percentile(inductive, [75, 25]))),
                    "saturated_fraction": float(np.mean(inductive >= SATURATION_THRESHOLD)),
                },
            }

        family_auroc: dict[str, Any] = {}
        for family in dg.FAMILIES:
            columns = design.columns((family,))
            if not columns:
                continue
            detector = fit_shift_detector(design, fold, "mahalanobis", columns=columns)
            family_auroc[family] = {
                "n_features": len(columns),
                "row_level_auroc": float(
                    roc_auc(detector.raw(design, detection_index), positive.astype(float))
                ),
            }

        engine_of = design.meta["engine_id"].to_numpy(str)
        folds[held_out] = {
            "held_out_engine": held_out,
            "train_engines": list(fold.train_engines),
            "detection_positives": int(positive.sum()),
            "detection_negatives": int((~positive).sum()),
            "detection_documents": len(set(documents.tolist())),
            "development_documents_reached_by_held_out_engine": len(
                set(design.documents[fold.eval].tolist())
            ),
            "development_documents_dropped_as_unmatched": len(
                set(design.documents[fold.eval].tolist())
                | set(design.documents[fold.seen_eval].tolist())
            )
            - len(set(documents.tolist())),
            "detectors": per_detector,
            "mahalanobis_by_feature_family": family_auroc,
            "candidates_per_document": {
                engine: float(
                    np.count_nonzero((engine_of == engine) & (design.meta["role"] == "DEVELOPMENT"))
                    / max(
                        len(
                            set(
                                design.documents[
                                    (engine_of == engine)
                                    & (design.meta["role"].to_numpy(str) == "DEVELOPMENT")
                                ].tolist()
                            )
                        ),
                        1,
                    )
                )
                for engine in design.engines
            },
        }
        best = max(per_detector, key=lambda k: per_detector[k]["row_level_auroc"]["estimate"])
        print(
            f"  fold {held_out:11s} best={best:17s} "
            f"row AUROC={per_detector[best]['row_level_auroc']['estimate']:.3f} "
            f"doc AUROC={per_detector[best]['document_level_auroc']['estimate']:.3f}"
        )

    cc._write_json_once(
        SHIFT_RESULTS,
        {
            "schema_version": "sgv2-reliability-shift-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV2-R1",
            "question": "can an unseen OCR engine be identified without ground truth?",
            "design_note": (
                "Positives and negatives sit on the SAME 133 development documents and differ "
                "only in which engine read them, so a high AUROC is engine separation and not "
                "page novelty. The within-engine null re-runs the same measurement on pages "
                "that differ by document alone."
            ),
            "prior_measurement": (
                "Phase 6 measured a supervised domain classifier between engines at AUC "
                "0.867-0.999 (results/generated/sgv1/domain_generalization/"
                "engine_shift_analysis.json). A high detection AUROC here therefore confirms "
                "an existing measurement; it is not evidence about safety."
            ),
            "detector_configuration": {
                "mahalanobis": {"ridge": MAHALANOBIS_RIDGE, "covariance": "pooled, pseudo-inverse"},
                "isolation_forest": {"n_estimators": FOREST_TREES, "random_state": pilot.FIT_SEED},
                "mahalanobis_nearest_mode": {
                    "ridge": MAHALANOBIS_RIDGE,
                    "covariance": "one Gaussian per fit engine, distance to the nearest",
                },
                "one_class_svm": {
                    "estimator": "sklearn OneClassSVM",
                    "kernel": "rbf",
                    "gamma": "scale",
                    "nu": ONE_CLASS_NU,
                    "subsample": ONE_CLASS_SUBSAMPLE,
                    "subsample_seed": pilot.FIT_SEED,
                    "rejected_configuration": (
                        "SGDOneClassSVM (linear) was measured first and was degenerate: score "
                        "spread ~1e-3 and AUROC at chance on all four folds. It is recorded "
                        "because 'the one-class SVM failed' and 'this configuration of it "
                        "failed' are different findings."
                    ),
                },
                "domain_classifier": {
                    "estimator": "sklearn LogisticRegression",
                    "requires": "a batch of UNLABELED pages from the new engine",
                },
            },
            "family_auroc_has_no_interval": (
                "The per-family decomposition is a point estimate. It is descriptive -- it says "
                "which block of features carries the separation -- and eight more document "
                "bootstraps per fold would not change what it is used for."
            ),
            "auroc_uses_raw_scores": (
                "Detection AUROC is computed on raw detector scores. The frozen percentile is "
                "a monotone map but it saturates at 1.0, and the ties that creates would lower "
                "the AUROC for a reason that has nothing to do with detection."
            ),
            "folds": folds,
            "confirmatory_accessed": False,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(f"shift: {len(folds)} folds -> {cc._relative(SHIFT_RESULTS)}")
    return 0


# ------------------------------------------------------------------ experiment 2: reliability


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    from scipy.stats import spearmanr

    if left.size < 3 or np.all(left == left[0]) or np.all(right == right[0]):
        return float("nan")
    return float(spearmanr(left, right).statistic)


def _spearman_interval(
    shift: np.ndarray, harmful: np.ndarray, documents: np.ndarray
) -> dict[str, float]:
    """Coefficient with a DOCUMENT-clustered interval.

    scipy's own p-value is not reported anywhere in this stage: it assumes independent
    observations, and candidate rows inside one page share a scan, a font and usually a
    systematic engine failure. That p would be a statement about a dataset this is not.
    """
    result = cluster_bootstrap_indices(
        list(documents),
        lambda index: _spearman(shift[index], harmful[index].astype(float)),
        n_resamples=BOOTSTRAP_RESAMPLES,
        seed=pilot.BOOTSTRAP_SEED,
        bounds=(-1.0, 1.0),
    )
    return {"estimate": result.estimate, "ci_lower": result.lower, "ci_upper": result.upper}


def _quantile_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 3:
        return np.zeros(values.size, dtype=int)
    return np.clip(np.searchsorted(edges[1:-1], values, side="right"), 0, edges.size - 2)


def _harm_by_bin(
    values: np.ndarray, harmful: np.ndarray, documents: np.ndarray, n_bins: int
) -> dict[str, Any]:
    """Prevalence of a binary outcome by quantile of a score, with a top-minus-bottom contrast.

    Called twice per detector: once for harm, which is the quantity SGV2-R1 is about, and once
    for benefit. Reporting only the first would leave the obvious alternative explanation --
    that the anomalous rows are where the repairs are rather than where the harm is --
    unmeasured, and an unmeasured alternative is not a rejected one.
    """
    bins = _quantile_bins(values, n_bins)
    distinct = sorted(set(bins.tolist()))
    cells: dict[str, Any] = {}
    for b in distinct:
        mask = bins == b
        interval = cluster_bootstrap_indices(
            list(documents[mask]),
            lambda index, m=mask: float(harmful[m][index].mean()) if index.size else float("nan"),
            n_resamples=BOOTSTRAP_RESAMPLES,
            seed=pilot.BOOTSTRAP_SEED,
        )
        cells[f"bin_{b + 1}"] = {
            "n": int(mask.sum()),
            "score_min": float(values[mask].min()),
            "score_max": float(values[mask].max()),
            "harm_rate": interval.estimate,
            "ci_lower": interval.lower,
            "ci_upper": interval.upper,
        }
    top, bottom = f"bin_{distinct[-1] + 1}", "bin_1"
    contrast = cluster_bootstrap_indices(
        list(documents),
        lambda index: (
            float(
                harmful[index][bins[index] == distinct[-1]].mean()
                - harmful[index][bins[index] == distinct[0]].mean()
            )
            if (bins[index] == distinct[-1]).any() and (bins[index] == distinct[0]).any()
            else float("nan")
        ),
        n_resamples=BOOTSTRAP_RESAMPLES,
        seed=pilot.BOOTSTRAP_SEED,
        bounds=None,
    )
    return {
        "bins": cells,
        "n_bins_realized": len(distinct),
        "top_minus_bottom": {
            "estimate": contrast.estimate,
            "ci_lower": contrast.lower,
            "ci_upper": contrast.upper,
            "p_value_two_sided": contrast.p_value_two_sided,
            "compared": [top, bottom],
        },
    }


def _conditional_on_risk(
    shift: np.ndarray, safety: np.ndarray, harmful: np.ndarray, n_bins: int = 10
) -> dict[str, Any]:
    """Does the shift score still track harm once the risk score is held roughly fixed?

    This is the question the marginal correlation cannot answer. The risk model consumes the
    same 96 features the detector does, so a marginal association between shift and harm can
    be entirely re-expression of what the risk score already says -- and a reliability layer
    that only restates the risk score is not a layer, it is a rename.
    """
    bins = _quantile_bins(safety, n_bins)
    per_bin: list[dict[str, float]] = []
    for b in sorted(set(bins.tolist())):
        mask = bins == b
        per_bin.append(
            {
                "n": int(mask.sum()),
                "harm_rate": float(harmful[mask].mean()),
                "spearman": _spearman(shift[mask], harmful[mask].astype(float)),
            }
        )
    weights = np.array([cell["n"] for cell in per_bin], dtype=float)
    values = np.array([cell["spearman"] for cell in per_bin], dtype=float)
    usable = np.isfinite(values)
    pooled = (
        float(np.sum(weights[usable] * values[usable]) / np.sum(weights[usable]))
        if usable.any()
        else float("nan")
    )
    return {
        "risk_score_bins": n_bins,
        "per_bin": per_bin,
        "weighted_mean_within_bin_spearman": pooled,
        "marginal_minus_conditional": float(_spearman(shift, harmful.astype(float)) - pooled),
    }


def run_reliability() -> int:
    """Experiment 2: does the shift score predict harmful corrections?"""
    started = time.monotonic()
    scores, record = load_scores()
    folds: dict[str, Any] = {}
    engine_level: dict[str, Any] = {}

    # d_before is ground truth. It enters here as an EXPLANATORY diagnostic and nowhere else:
    # no model, detector, threshold or feature in this stage is a function of it. Its job is to
    # say what "anomalous" turns out to mean, which a correlation between two label-free
    # quantities cannot.
    original_error = policy._pool_with_features().set_index("candidate_id")["d_before"]

    for held_out, fold in scores.items():
        safety = fold.frame["safety"].to_numpy(dtype=np.float64)
        harmful, documents = fold.harmful, fold.documents
        distance_before = original_error.reindex(fold.frame["candidate_id"].astype(str)).to_numpy(
            dtype=float
        )
        if not np.isfinite(distance_before).all():
            raise PhaseError(f"{held_out}: a scored row has no original edit distance")

        per_detector: dict[str, Any] = {}
        for kind in DETECTORS:
            raw = fold.frame[f"shift_raw__{kind}"].to_numpy(dtype=np.float64)
            per_detector[kind] = {
                "start": "warm" if kind in WARM_START_DETECTORS else "cold",
                "spearman_shift_vs_harm": _spearman_interval(raw, harmful, documents),
                "auroc_shift_predicts_harm": _auc_interval(raw, harmful, documents),
                "harm_by_shift_quartile": _harm_by_bin(
                    raw, harmful, documents, SHIFT_QUANTILE_BINS
                ),
                "conditional_on_risk_score": _conditional_on_risk(raw, safety, harmful),
                "benefit_by_shift_quartile": _harm_by_bin(
                    raw, fold.beneficial, documents, SHIFT_QUANTILE_BINS
                ),
                "spearman_shift_vs_original_error": _spearman_interval(
                    raw, distance_before, documents
                ),
                "mean_frozen_percentile": float(
                    fold.frame[f"shift_pct__{kind}"].to_numpy(dtype=np.float64).mean()
                ),
            }

        # The control. The risk model is trained to rank harm, so it is the standard any
        # reliability signal has to clear; without it a shift score that merely re-expresses
        # the risk score would read as an independent finding.
        per_detector["risk_score_control"] = {
            "start": "n/a -- the existing risk model, negated so that higher means riskier",
            "spearman_shift_vs_harm": _spearman_interval(-safety, harmful, documents),
            "auroc_shift_predicts_harm": _auc_interval(-safety, harmful, documents),
            "harm_by_shift_quartile": _harm_by_bin(
                -safety, harmful, documents, SHIFT_QUANTILE_BINS
            ),
        }

        folds[held_out] = {
            "held_out_engine": held_out,
            "evaluation_rows": len(fold.frame),
            "evaluation_documents": len(set(documents.tolist())),
            "harm_prevalence": float(harmful.mean()),
            "selected_detector": record["folds"][held_out]["selected_detector"],
            "detectors": per_detector,
        }
        engine_level[held_out] = {
            "mean_mahalanobis_percentile": per_detector["mahalanobis"]["mean_frozen_percentile"],
            "harm_prevalence": float(harmful.mean()),
        }
        control = per_detector["risk_score_control"]["auroc_shift_predicts_harm"]["estimate"]
        best = max(
            (k for k in per_detector if k != "risk_score_control"),
            key=lambda k: per_detector[k]["auroc_shift_predicts_harm"]["estimate"],
        )
        print(
            f"  fold {held_out:11s} best shift->harm AUROC="
            f"{best}:{per_detector[best]['auroc_shift_predicts_harm']['estimate']:.3f}  "
            f"risk control={control:.3f}"
        )

    directions = {
        kind: {
            e: folds[e]["detectors"][kind]["spearman_shift_vs_harm"]["estimate"]
            for e in sorted(folds)
        }
        for kind in DETECTORS
    }

    cc._write_json_once(
        RELIABILITY_RESULTS,
        {
            "schema_version": "sgv2-reliability-analysis-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV2-R1",
            "question": "does a label-free shift score predict harmful corrections?",
            "expected_direction": (
                "SGV2-R1 requires a POSITIVE association: rows a detector finds unlike the fit "
                "distribution should carry more harm. A negative coefficient is not a weak "
                "result, it is the opposite result, and is reported with its sign."
            ),
            "spearman_by_detector_and_engine": directions,
            "mechanism_note": (
                "If the association between shift and harm runs the wrong way, the question is "
                "what the detectors are actually finding. Two diagnostics answer it: benefit "
                "prevalence by shift quartile, from the same outcome taxonomy as harm; and the "
                "rank correlation between the shift score and d_before, the edit distance the "
                "OCR started with. The second uses ground truth and is EXPLANATORY ONLY -- no "
                "model, detector, threshold or feature in this stage is a function of it."
            ),
            "engine_level_note": (
                "The across-engine version of this question has four points -- one per engine -- "
                "and no correlation is fitted to four points. The four pairs are listed so a "
                "reader can see them; they are not a test."
            ),
            "engine_level_pairs": engine_level,
            "conditional_note": (
                "The risk model consumes the same 96 features the detectors do, so a marginal "
                "association between shift and harm can be a re-expression of what the risk "
                "score already says. The within-risk-decile coefficient is the part that is "
                "not; a reliability layer that only restates the risk score is a rename."
            ),
            "folds": folds,
            "confirmatory_accessed": False,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(f"reliability: {len(folds)} folds -> {cc._relative(RELIABILITY_RESULTS)}")
    return 0


# ------------------------------------------------------------------ the error taxonomy

# The base classes are Phase 3's `_error_category`, imported unchanged so this stage cannot
# quietly re-cut them and report a different picture under the same names. The brief asks for
# five classes; the base has no date/time class, so one is added here and everything else is a
# declared MAPPING, recorded in the artifact so a reader can undo it.
DATE_TIME_PATTERN = r"(\d{1,4}[-/.]\d{1,2}[-/.]\d{2,4})|(\b\d{1,2}:\d{2}\b)"
BASE_CLASS_MAP = {
    "numeric_price": "numeric",
    "character_confusion": "character_confusion",
    "gap_insertion": "layout",
    "segmentation": "layout",
    "same_length_other": "language_model",
    "other": "language_model",
    "formatting": "formatting_other",
}


def error_classes(design: dg.Design) -> tuple[np.ndarray, np.ndarray]:
    """Base class and mapped class for every design row, joined on candidate_id."""
    import re

    pool = policy._pool_with_features()
    pattern = re.compile(DATE_TIME_PATTERN)
    base: dict[str, str] = {}
    mapped: dict[str, str] = {}
    for row in pool.itertuples():
        identifier = str(row.candidate_id)
        raw = cc._error_category(row)
        base[identifier] = raw
        if pattern.search(str(row.original_ocr)) or pattern.search(str(row.candidate_text)):
            mapped[identifier] = "datetime"
        else:
            mapped[identifier] = BASE_CLASS_MAP.get(raw, "formatting_other")
    ids = design.meta["candidate_id"].astype(str).to_numpy()
    missing = [i for i in ids if i not in base]
    if missing:
        raise PhaseError(f"{len(missing)} design rows have no candidate row to classify")
    return (
        np.array([base[i] for i in ids], dtype=object),
        np.array([mapped[i] for i in ids], dtype=object),
    )


# ------------------------------------------------------------------ scores, fitted once


def run_scores() -> int:
    """Fit every fold once and write the row-level scores every later stage reads.

    Phase 6 refitted its folds inside each experiment, which is both wasteful and unfalsifiable:
    two stages that refit cannot be shown to have scored the same rows. Here the fitting happens
    once, the result is a write-once table, and every downstream number is a function of that
    table -- so a reviewer can recompute any of them without running a model.
    """
    started = time.monotonic()
    design = dg.load_design()
    base_class, mapped_class = error_classes(design)
    every = design.columns(tuple(dg.FAMILIES))
    frames: list[pd.DataFrame] = []
    record: dict[str, Any] = {}

    for held_out in design.engines:
        fold = build_fold(design, held_out)
        selection = select_policy(design, fold)
        detectors = {kind: fit_shift_detector(design, fold, kind) for kind in DETECTORS}
        arms, diagnostics = build_arms(design, fold, selection, detectors)

        model = dg.fit_harm_model(design, fold.fit, every)
        calibrator = build_calibrator(pilot.CALIBRATION_METHOD)
        source_raw = model.safety(design, fold.source_cal)
        calibrator.fit(source_raw, (~design.harmful[fold.source_cal]).astype(float))
        on_calibration = calibrator.transform(source_raw)
        thresholds = {
            controller: select_threshold(
                on_calibration,
                design.harmful[fold.source_cal],
                PRIMARY_EPSILON,
                delta=DELTA,
                controller=controller,
            ).as_dict()
            for controller in CONTROLLERS
        }

        frame = pd.DataFrame(
            {
                "candidate_id": design.meta["candidate_id"].astype(str).to_numpy()[fold.eval],
                "document_id": design.documents[fold.eval],
                "engine_id": design.meta["engine_id"].to_numpy(str)[fold.eval],
                "held_out_engine": held_out,
                "is_harmful": design.harmful[fold.eval],
                "beneficial": design.beneficial[fold.eval],
                "base_error_class": base_class[fold.eval].astype(str),
                "error_class": mapped_class[fold.eval].astype(str),
                "safety": arms["risk_only"],
                "probability": calibrator.transform(model.safety(design, fold.eval)),
            }
        )
        for kind, detector in detectors.items():
            raw = detector.raw(design, fold.eval)
            frame[f"shift_raw__{kind}"] = raw
            frame[f"shift_pct__{kind}"] = detector.percentile(raw)
            frame[f"shift_batch__{kind}"] = _batch_rank(raw)
        for name, score in arms.items():
            frame[f"arm__{name}"] = score
        frames.append(frame)

        record[held_out] = {
            "held_out_engine": held_out,
            "train_engines": list(fold.train_engines),
            "rows": {
                "fit": int(fold.fit.size),
                "source_calibration": int(fold.source_cal.size),
                "seen_development": int(fold.seen_eval.size),
                "evaluation": int(fold.eval.size),
                "unlabeled_target_pages": int(fold.unlabeled_target.size),
            },
            "documents": {
                "fit": len(set(design.documents[fold.fit].tolist())),
                "source_calibration": len(set(design.documents[fold.source_cal].tolist())),
                "evaluation": len(set(design.documents[fold.eval].tolist())),
            },
            "selected_detector": selection.detector,
            "selected_kappa": selection.kappa,
            "inner_selection": selection.inner,
            "diagnostics": diagnostics,
            "calibration": {
                "method": pilot.CALIBRATION_METHOD,
                "fitted_on": "fit engines, CALIBRATION documents",
                "rows": int(fold.source_cal.size),
            },
            "certified_thresholds": thresholds,
        }
        print(
            f"  fold {held_out:11s} detector={selection.detector:24s} "
            f"kappa={selection.kappa:g} saturation={diagnostics['saturation_fraction']:.3f}"
        )

    cc._write_parquet_once(SCORES, pd.concat(frames, ignore_index=True))
    cc._write_json_once(
        SELECTION_RECORD,
        {
            "schema_version": "sgv2-reliability-selection-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV2-R1",
            "selection_scope": (
                "detector and kappa are chosen by an inner leave-one-engine-out over the FIT "
                "engines, evaluated on CALIBRATION documents. The held-out engine contributes "
                "nothing to the choice; tests/leakage re-derives the choice from the recorded "
                "inner scores."
            ),
            "folds": record,
            "confirmatory_accessed": False,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(f"scores: {sum(len(f) for f in frames)} rows -> {cc._relative(SCORES)}")
    return 0


@dataclass(slots=True)
class FoldScores:
    """One fold's evaluation rows, read back from the write-once score table."""

    held_out: str
    frame: pd.DataFrame
    harmful: np.ndarray
    beneficial: np.ndarray
    documents: np.ndarray

    def arm(self, name: str) -> np.ndarray:
        return self.frame[f"arm__{name}"].to_numpy(dtype=np.float64)

    @property
    def arm_names(self) -> list[str]:
        return [c[len("arm__") :] for c in self.frame.columns if c.startswith("arm__")]


def load_scores() -> tuple[dict[str, FoldScores], dict[str, Any]]:
    if not SCORES.is_file() or not SELECTION_RECORD.is_file():
        raise PhaseError("run --scores before any experiment that reads them")
    table = pd.read_parquet(SCORES)
    record = cc._read_json(SELECTION_RECORD)
    folds = {}
    for held_out, group in table.groupby("held_out_engine", sort=True):
        frame = group.reset_index(drop=True)
        if set(frame["engine_id"].astype(str)) != {str(held_out)}:
            raise PhaseError(f"{held_out}: an engine other than the held-out one is in its fold")
        folds[str(held_out)] = FoldScores(
            held_out=str(held_out),
            frame=frame,
            harmful=frame["is_harmful"].to_numpy(dtype=bool),
            beneficial=frame["beneficial"].to_numpy(dtype=bool),
            documents=frame["document_id"].astype(str).to_numpy(),
        )
    return folds, record


# ------------------------------------------------------------------ experiment 3: abstention


def _matched_coverage_contrast(
    baseline: np.ndarray,
    arm: np.ndarray,
    harmful: np.ndarray,
    beneficial: np.ndarray,
    documents: np.ndarray,
) -> dict[str, Any]:
    """Harm reduction at matched site coverage, paired on documents.

    Positive means the arm accepted the SAME number of edits and fewer of them were harmful.
    Matching the coverage is what makes this a comparison of two rankings rather than of two
    abstention rates: any policy can drive its harm rate to zero by accepting less.
    """
    cells: dict[str, Any] = {}
    for coverage in MATCHED_COVERAGES:
        result = cluster_bootstrap_indices(
            list(documents),
            _harm_at_coverage_statistic(baseline, arm, harmful, beneficial, coverage),
            n_resamples=BOOTSTRAP_RESAMPLES,
            seed=pilot.BOOTSTRAP_SEED,
            bounds=None,
        )
        cells[f"coverage_{int(coverage * 100)}"] = {
            "harm_reduction": result.estimate,
            "ci_lower": result.lower,
            "ci_upper": result.upper,
            "p_value_two_sided": result.p_value_two_sided,
            "favours_arm": bool(result.estimate > 0),
        }
    return cells


def run_abstain() -> int:
    """Experiment 3: shift-aware abstention, judged at matched coverage."""
    started = time.monotonic()
    scores, record = load_scores()
    key = f"epsilon_{int(PRIMARY_EPSILON * 100)}"
    folds: dict[str, Any] = {}

    for held_out, fold in scores.items():
        baseline = fold.arm("risk_only")
        baseline_recall = dg.frontier(baseline, fold.harmful, fold.beneficial)[key]["repair_recall"]
        results: dict[str, Any] = {}
        for name in fold.arm_names:
            score = fold.arm(name)
            summary = arm_summary(score, fold.harmful, fold.beneficial, fold.documents)
            eligible = np.isfinite(score)
            summary["rank_correlation_with_risk_only_on_kept_rows"] = _spearman(
                score[eligible], baseline[eligible]
            )
            summary["frontier_move_vs_risk_only"] = float(
                summary[key]["repair_recall"] - baseline_recall
            )
            if name != "risk_only":
                summary["matched_coverage_contrast"] = _matched_coverage_contrast(
                    baseline, score, fold.harmful, fold.beneficial, fold.documents
                )
            results[name] = summary

        folds[held_out] = {
            "held_out_engine": held_out,
            "evaluation_rows": len(fold.frame),
            "evaluation_documents": len(set(fold.documents.tolist())),
            "beneficial": int(fold.beneficial.sum()),
            "harmful": int(fold.harmful.sum()),
            "harm_prevalence": float(fold.harmful.mean()),
            "selected_detector": record["folds"][held_out]["selected_detector"],
            "selected_kappa": record["folds"][held_out]["selected_kappa"],
            "diagnostics": record["folds"][held_out]["diagnostics"],
            "arms": results,
        }
        best = max(fold.arm_names, key=lambda n: results[n][key]["repair_recall"])
        print(
            f"  fold {held_out:11s} risk_only={baseline_recall:.3f} "
            f"best={best}:{results[best][key]['repair_recall']:.3f} "
            f"oracle={results['oracle_selected_batch'][key]['repair_recall']:.3f}"
        )

    arm_names = next(iter(scores.values())).arm_names
    fold_means = {
        name: float(np.mean([folds[e]["arms"][name][key]["repair_recall"] for e in sorted(folds)]))
        for name in arm_names
    }
    beats_baseline = {
        name: sorted(
            e
            for e in folds
            if folds[e]["arms"][name][key]["repair_recall"]
            > folds[e]["arms"]["risk_only"][key]["repair_recall"]
        )
        for name in arm_names
    }

    cc._write_json_once(
        ABSTENTION_RESULTS,
        {
            "schema_version": "sgv2-reliability-abstention-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV2-R1",
            "primary_endpoint": "repair_recall_at_bounded_harm",
            "co_primary_endpoint": "harm reduction at matched site coverage",
            "endpoint_note": (
                "Two endpoints, because the brief names one and this project already owns the "
                "other. Repair recall under a harm bound is the project's definition, imported "
                "from Phase 5. Harm reduction at matched coverage is the brief's, and is the "
                "one an abstention arm cannot win by abstaining more."
            ),
            "reordering_note": (
                "Re-scoring and rejecting move the frontier by different mechanisms, and the "
                "rank correlation reported per arm only speaks to the first. An arm that "
                "RE-SCORES -- adaptive_*, warm_start_*, oracle_* -- changes the frontier only "
                "by changing the order, so a rank correlation of 1.0 with risk-only would mean "
                "it changed nothing. An arm that REJECTS -- reject_* -- leaves the order of the "
                "rows it keeps untouched, which is why its rank correlation is exactly 1.0, and "
                "can still move the frontier: removing rows changes which prefixes satisfy the "
                "harm bound. Reading a rank correlation of 1.0 as 'this arm cannot have moved "
                "the frontier' is true of the first kind and false of the second."
            ),
            "truncation_moves_the_frontier_note": (
                "Concretely: if the rejected rows are disproportionately harmful and sit high in "
                "the risk ranking, removing them lets the accept prefix extend further before "
                "the bound binds, and repair recall at a fixed epsilon rises with no change to "
                "the ranking at all."
            ),
            "arm_glossary": {
                "risk_only": "Phase 6's transferred harm model; the baseline",
                "shift_only_typical_first": (
                    "rank by the shift score alone, most typical first, no risk model"
                ),
                "shift_only_anomalous_first": (
                    "the same score reversed. The pair brackets what the shift signal can do "
                    "on its own and shows which direction it actually points."
                ),
                "reject_topN_batch": (
                    "drop the N% most anomalous rows of THIS batch, then rank by risk; "
                    "transductive within the batch, ground-truth-free"
                ),
                "reject_bottomN_batch": (
                    "the mirror: drop the N% most TYPICAL rows. Present because the direction "
                    "of the association is a measurement, and a table containing only the "
                    "assumed direction cannot report that it was wrong."
                ),
                "reject_above_pN_inductive": (
                    "drop rows past the Nth percentile of the frozen fit distribution; needs no "
                    "batch, so it can act on the first page from a new engine"
                ),
                "adaptive_inductive": "risk score minus kappa times the frozen percentile",
                "adaptive_batch": "risk score minus kappa times the within-batch rank",
                "warm_start_domain_batch": (
                    "the same rule using the domain classifier, which needs a batch of "
                    "unlabeled pages from the new engine before it can score anything"
                ),
                "oracle_selected_batch": (
                    "detector and kappa chosen on the held-out engine's own labels. A ceiling on "
                    "what perfect selection could deliver, never an arm; its job is to separate "
                    "'the selection failed' from 'the signal is not there'."
                ),
            },
            "fold_mean_repair_recall_at_primary_epsilon": fold_means,
            "engines_where_arm_beats_risk_only": beats_baseline,
            "folds": folds,
            "confirmatory_accessed": False,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(f"abstain: {len(folds)} folds -> {cc._relative(ABSTENTION_RESULTS)}")
    return 0


# ------------------------------------------------------------------ experiment 4: safety


def _review_bound(
    gated: np.ndarray, ungated: np.ndarray, tau: float, harmful: np.ndarray, beneficial: np.ndarray
) -> dict[str, Any]:
    """What the third decision could be worth, under an assumption that is stated, not met.

    ABSTAIN (route to a human) and PRESERVE (keep the OCR output) are the same event in this
    benchmark: both leave `O` in place, so both give `d_after = d_before` and neither is
    harmful nor beneficial. There is no human-review channel to measure, so what is reported
    is a BOUND -- the beneficial edits a reviewer who was always right and always available
    could recover from the rows the gate removed, and the harmful ones it correctly stopped.
    An upper bound under a perfect-reviewer assumption is not a measured result.
    """
    diverted = (ungated >= tau) & ~(gated >= tau)
    return {
        "assumption": "a reviewer who is always correct and always available",
        "rows_diverted_to_review": int(diverted.sum()),
        "review_load_fraction_of_sites": float(diverted.mean()),
        "harmful_edits_stopped": int(harmful[diverted].sum()),
        "beneficial_edits_a_perfect_reviewer_could_recover": int(beneficial[diverted].sum()),
        "share_of_the_diversion_that_was_harmful": float(harmful[diverted].mean())
        if diverted.any()
        else float("nan"),
    }


def run_transfer() -> int:
    """Experiment 4: does a shift gate reduce harmful corrections on an unseen engine?"""
    started = time.monotonic()
    scores, record = load_scores()
    folds: dict[str, Any] = {}

    for held_out, fold in scores.items():
        settings = record["folds"][held_out]
        chosen = settings["selected_detector"]
        probability = fold.frame["probability"].to_numpy(dtype=np.float64)
        inductive = fold.frame[f"shift_pct__{chosen}"].to_numpy(dtype=np.float64)
        batch = fold.frame[f"shift_batch__{chosen}"].to_numpy(dtype=np.float64)
        gates = {
            "inductive_p95": inductive <= 0.95,
            "batch_top20": batch < float(np.quantile(batch, 0.80)),
        }

        deployed: dict[str, Any] = {}
        for controller, threshold in settings["certified_thresholds"].items():
            tau = float(threshold["tau"])
            without = dg.deployed_point(
                tau, probability, fold.harmful, fold.beneficial, PRIMARY_EPSILON
            )
            cell: dict[str, Any] = {
                "threshold": threshold,
                "risk_policy_only": without,
                "with_gate": {},
            }
            for gate_name, keep in gates.items():
                gated = np.where(keep, probability, -np.inf)
                with_gate = dg.deployed_point(
                    tau, gated, fold.harmful, fold.beneficial, PRIMARY_EPSILON
                )
                cell["with_gate"][gate_name] = {
                    **with_gate,
                    "gate_rejection_fraction": float(1.0 - keep.mean()),
                    "harm_reduction": float(
                        without["realized_harm_rate"] - with_gate["realized_harm_rate"]
                    ),
                    "repair_recall_given_up": float(
                        without["repair_recall"] - with_gate["repair_recall"]
                    ),
                    "coverage_given_up": float(without["coverage"] - with_gate["coverage"]),
                    "bound_restored": bool(
                        without["bound_violated"] and not with_gate["bound_violated"]
                    ),
                    "review_bound": _review_bound(
                        gated, probability, tau, fold.harmful, fold.beneficial
                    ),
                }
            deployed[controller] = cell

        matched = {
            name: _matched_coverage_contrast(
                fold.arm("risk_only"),
                fold.arm(name),
                fold.harmful,
                fold.beneficial,
                fold.documents,
            )
            for name in ("adaptive_batch", "adaptive_inductive", "reject_top20_batch")
        }

        folds[held_out] = {
            "held_out_engine": held_out,
            "evaluation_rows": len(fold.frame),
            "evaluation_documents": len(set(fold.documents.tolist())),
            "selected_detector": chosen,
            "selected_kappa": settings["selected_kappa"],
            "calibration": settings["calibration"],
            "deployed_operating_points": deployed,
            "matched_coverage_contrast": matched,
            "saturation_fraction": settings["diagnostics"]["saturation_fraction"],
        }
        first = next(iter(settings["certified_thresholds"]))
        plain = deployed[first]["risk_policy_only"]
        gated_cell = deployed[first]["with_gate"]["batch_top20"]
        print(
            f"  fold {held_out:11s} "
            f"bound {'VIOLATED' if plain['bound_violated'] else 'held'} ungated "
            f"(harm={plain['realized_harm_rate']:.3f}); batch gate "
            f"harm={gated_cell['realized_harm_rate']:.3f} "
            f"coverage {plain['coverage']:.3f}->{gated_cell['coverage']:.3f}"
        )

    cc._write_json_once(
        TRANSFER_SAFETY,
        {
            "schema_version": "sgv2-reliability-transfer-safety-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV2-R1",
            "question": "does adding a shift gate reduce harmful corrections on an unseen engine?",
            "exchangeability_note": (
                "The certified threshold assumes exchangeable calibration and evaluation data. "
                "An unseen engine is not exchangeable with the fit engines by construction, so "
                "the certificate does not apply and every number here measures how far it "
                "misses. Phase 6 established the same caveat and it is not weaker here."
            ),
            "two_ways_to_reduce_harm": (
                "A gate can lower the realized harm rate by removing harmful edits from the "
                "accepted set, or by removing edits. Only the first is safety. Every cell "
                "therefore carries the coverage and the repair recall it gave up, and the "
                "matched-coverage contrast holds the volume fixed so the two cannot be "
                "confused."
            ),
            "folds": folds,
            "confirmatory_accessed": False,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(f"transfer: {len(folds)} folds -> {cc._relative(TRANSFER_SAFETY)}")
    return 0


# ------------------------------------------------------------------ experiment 5: failure modes


def run_failure() -> int:
    """Experiment 5: where does shift detection succeed, and where does it fail?"""
    started = time.monotonic()
    scores, record = load_scores()
    folds: dict[str, Any] = {}

    for held_out, fold in scores.items():
        chosen = record["folds"][held_out]["selected_detector"]
        raw = fold.frame[f"shift_raw__{chosen}"].to_numpy(dtype=np.float64)
        percentile = fold.frame[f"shift_pct__{chosen}"].to_numpy(dtype=np.float64)
        classes = fold.frame["error_class"].astype(str).to_numpy()
        base = fold.frame["base_error_class"].astype(str).to_numpy()

        cells: dict[str, Any] = {}
        for name in sorted(set(classes.tolist())):
            mask = classes == name
            entry: dict[str, Any] = {
                "n": int(mask.sum()),
                "harm_prevalence": float(fold.harmful[mask].mean()),
                "beneficial": int(fold.beneficial[mask].sum()),
                "mean_frozen_percentile": float(percentile[mask].mean()),
                "base_classes": sorted(set(base[mask].tolist())),
            }
            if fold.harmful[mask].any() and (~fold.harmful[mask]).any():
                entry["auroc_shift_predicts_harm"] = _auc_interval(
                    raw[mask], fold.harmful[mask], fold.documents[mask]
                )
            else:
                entry["auroc_shift_predicts_harm"] = {
                    "estimate": float("nan"),
                    "note": "one harm class only in this cell; AUROC undefined",
                }
            if fold.beneficial[mask].any():
                entry["matched_coverage_contrast_at_10"] = _matched_coverage_contrast(
                    fold.arm("risk_only")[mask],
                    fold.arm("adaptive_batch")[mask],
                    fold.harmful[mask],
                    fold.beneficial[mask],
                    fold.documents[mask],
                )["coverage_10"]
            else:
                entry["matched_coverage_contrast_at_10"] = {
                    "note": "no beneficial row in this class; the endpoint is undefined"
                }
            cells[name] = entry

        folds[held_out] = {
            "held_out_engine": held_out,
            "selected_detector": chosen,
            "classes": cells,
        }
        print(f"  fold {held_out:11s} {len(cells)} error classes")

    # A census over the whole labelable pool, not just the folds. It is here because one of
    # the brief's five classes turned out to be empty, and "no row of this kind exists" is a
    # property of the corpus that a per-fold table would leave looking like an omission.
    pool = policy._pool_with_features()
    lengths = pool["original_ocr"].astype(str).str.len()
    design = dg.load_design()
    _, mapped_all = error_classes(design)
    census = {
        name: int(np.count_nonzero(mapped_all == name)) for name in sorted(set(mapped_all.tolist()))
    }

    classes_seen = sorted({c for f in folds.values() for c in f["classes"]})
    detects_in = {
        name: sorted(
            e
            for e in folds
            if name in folds[e]["classes"]
            and folds[e]["classes"][name]["auroc_shift_predicts_harm"]["estimate"] > 0.5
        )
        for name in classes_seen
    }

    cc._write_json_once(
        FAILURE_RESULTS,
        {
            "schema_version": "sgv2-reliability-failure-modes-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV2-R1",
            "taxonomy_note": (
                "The base classes are Phase 3's `_error_category`, imported unchanged. The "
                "brief names five classes and the base has no date/time class, so one is added "
                "and the rest is a mapping. Both the base class and the mapped class are "
                "recorded, so the mapping can be undone. The taxonomy is derived from the "
                "frozen edit and never from ground truth."
            ),
            "class_mapping": BASE_CLASS_MAP,
            "datetime_pattern": DATE_TIME_PATTERN,
            "datetime_precedence": "a date or time match overrides the base class",
            "outside_the_briefs_five": {
                "formatting_other": (
                    "punctuation-only edits. The brief's five classes have no home for them, "
                    "and forcing them into 'layout' or 'language model' would be a decision "
                    "made for the table's benefit rather than the data's."
                )
            },
            "class_census_over_the_whole_pool": census,
            "empty_classes": sorted(
                name
                for name in (
                    "character_confusion",
                    "numeric",
                    "datetime",
                    "layout",
                    "language_model",
                )
                if census.get(name, 0) == 0
            ),
            "original_span_length": {
                "mean": float(lengths.mean()),
                "median": float(lengths.median()),
                "p95": float(lengths.quantile(0.95)),
                "note": (
                    "The candidate generator proposes edits at short spans, not at whole "
                    "fields. A date or time pattern needs several characters and a separator, "
                    "so whether the brief's date/time class can occur at all is a property of "
                    "the span length, which is recorded here rather than inferred."
                ),
            },
            "engines_where_shift_beats_chance_for_this_class": detects_in,
            "folds": folds,
            "confirmatory_accessed": False,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(f"failure: {len(folds)} folds -> {cc._relative(FAILURE_RESULTS)}")
    return 0


# ------------------------------------------------------------------ figures


def run_figures() -> int:
    started = time.monotonic()
    from ocr_risk.metrics.selective import risk_coverage_curve

    shift = cc._read_json(SHIFT_RESULTS)
    reliability = cc._read_json(RELIABILITY_RESULTS)
    safety = cc._read_json(TRANSFER_SAFETY)
    scores, record = load_scores()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    note = "SGV2 DEVELOPMENT -- not a confirmatory result"
    engines = sorted(scores)
    written: list[Path] = []

    def finish(figure: Any, path: Path, title: str) -> None:
        figure.suptitle(f"{title}\n{note}", fontsize=9)
        figure.tight_layout()
        figure.savefig(path, dpi=140)
        plt.close(figure)
        written.append(path)

    colours = {
        "mahalanobis": "#3a5f9e",
        "mahalanobis_nearest_mode": "#2e7d5b",
        "isolation_forest": "#c87a2b",
        "one_class_svm": "#7b52ab",
        "domain_classifier": "#a33",
    }

    # --- shift_distribution.png -----------------------------------------------------------
    figure, (left, right) = plt.subplots(1, 2, figsize=(12.5, 5.2))
    width = 0.15
    positions = np.arange(len(engines))
    for offset, kind in enumerate(DETECTORS):
        values = [
            shift["folds"][e]["detectors"][kind]["row_level_auroc"]["estimate"] for e in engines
        ]
        lower = [
            v - shift["folds"][e]["detectors"][kind]["row_level_auroc"]["ci_lower"]
            for v, e in zip(values, engines, strict=True)
        ]
        upper = [
            shift["folds"][e]["detectors"][kind]["row_level_auroc"]["ci_upper"] - v
            for v, e in zip(values, engines, strict=True)
        ]
        left.bar(
            positions + (offset - 2) * width,
            values,
            width,
            yerr=[lower, upper],
            capsize=2,
            color=colours[kind],
            label=f"{kind} ({'warm' if kind in WARM_START_DETECTORS else 'cold'})",
        )
    # The within-engine null, not 0.5, is the standard each bar has to clear: a detector that
    # scored unseen PAGES as anomalous would look like an engine detector. Drawing it on the
    # bars is what makes "does not clear its own null" visible rather than a claim in prose.
    for offset, kind in enumerate(DETECTORS):
        nulls = [shift["folds"][e]["detectors"][kind]["within_engine_null"]["p95"] for e in engines]
        left.scatter(
            positions + (offset - 2) * width,
            nulls,
            marker="_",
            s=90,
            color="k",
            zorder=4,
            label="within-engine null (p95)" if offset == 0 else None,
        )
    left.axhline(0.5, color="k", linestyle=":", linewidth=1, label="chance")
    left.set_xticks(positions)
    left.set_xticklabels(engines, fontsize=8)
    left.set_ylabel("row-level AUROC, unseen engine vs the three fit engines")
    # Headroom for the legend: the bars reach 1.0 on Tesseract and a legend inside the plot
    # area would sit on top of the warm-start result, which is the one a reader looks for.
    left.set_ylim(0.0, 1.32)
    left.legend(fontsize=7, ncol=3, loc="upper left", framealpha=0.95)
    left.set_title("Detecting the engine, on the same pages", fontsize=10)

    for engine in engines:
        chosen = record["folds"][engine]["selected_detector"]
        percentiles = scores[engine].frame[f"shift_pct__{chosen}"].to_numpy(dtype=float)
        right.hist(
            percentiles,
            bins=25,
            range=(0.0, 1.0),
            histtype="step",
            density=True,
            linewidth=1.6,
            label=f"{engine} ({chosen})",
        )
    right.axhline(1.0, color="k", linestyle=":", linewidth=1, label="uniform: no shift")
    right.set_xlabel("percentile of the frozen fit distribution")
    right.set_ylabel("density of the unseen engine's rows")
    right.legend(fontsize=7)
    right.set_title("Where the unseen engine's rows land", fontsize=10)
    finish(figure, FIGURE_DIR / "shift_distribution.png", "Label-free detection of engine shift")

    # --- risk_vs_shift.png ----------------------------------------------------------------
    figure, (left, right) = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True)
    for panel, source, title in (
        (left, "selected", "harm rate by shift quartile"),
        (right, "risk_score_control", "harm rate by risk-score quartile (control)"),
    ):
        for engine in engines:
            kind = record["folds"][engine]["selected_detector"] if source == "selected" else source
            cell = reliability["folds"][engine]["detectors"][kind]["harm_by_shift_quartile"]
            keys = sorted(cell["bins"], key=lambda k: int(k.split("_")[1]))
            centres = np.arange(len(keys)) + 1
            values = [cell["bins"][k]["harm_rate"] for k in keys]
            lower = [v - cell["bins"][k]["ci_lower"] for v, k in zip(values, keys, strict=True)]
            upper = [cell["bins"][k]["ci_upper"] - v for v, k in zip(values, keys, strict=True)]
            panel.errorbar(
                centres,
                values,
                yerr=[lower, upper],
                marker="o",
                capsize=3,
                label=f"{engine}" + (f" ({kind})" if source == "selected" else ""),
            )
        panel.set_xticks([1, 2, 3, 4])
        panel.set_xlabel(
            "shift quartile, 1 = most typical"
            if source == "selected"
            else "risk quartile, 1 = safest by the risk model"
        )
        panel.set_title(title, fontsize=10)
        panel.legend(fontsize=7)
    left.set_ylabel("harmful fraction of candidates")
    finish(
        figure,
        FIGURE_DIR / "risk_vs_shift.png",
        "Does the shift score track harm? The risk model is the standard it has to clear",
    )

    # --- coverage_risk_curve.png ----------------------------------------------------------
    figure, panels = plt.subplots(1, len(engines), figsize=(4.0 * len(engines), 4.4), sharey=True)
    # Both ends of the rejection family are drawn. Plotting only the hypothesis's direction
    # would leave the figure agreeing with a premise the data does not.
    plotted = (
        "risk_only",
        "reject_top20_batch",
        "reject_bottom20_batch",
        "adaptive_batch",
        "oracle_selected_batch",
    )
    styles = {
        "risk_only": ("#222", "-"),
        "reject_top20_batch": ("#c87a2b", "--"),
        "reject_bottom20_batch": ("#2a8f9e", "--"),
        "adaptive_batch": ("#3a5f9e", "-"),
        "oracle_selected_batch": ("#2e7d5b", ":"),
    }
    for panel, engine in zip(np.atleast_1d(panels), engines, strict=True):
        fold = scores[engine]
        for name in plotted:
            score = fold.arm(name)
            curve = risk_coverage_curve(
                np.where(np.isfinite(score), score, REJECTED_SCORE), fold.harmful
            )
            colour, style = styles[name]
            panel.plot(
                [point.coverage for point in curve.points],
                [point.risk for point in curve.points],
                color=colour,
                linestyle=style,
                linewidth=1.5,
                label=name,
            )
        controller = next(iter(safety["folds"][engine]["deployed_operating_points"]))
        cell = safety["folds"][engine]["deployed_operating_points"][controller]
        panel.plot(
            cell["risk_policy_only"]["coverage"],
            cell["risk_policy_only"]["realized_harm_rate"],
            "kx",
            markersize=9,
            label=f"certified tau ({controller})",
        )
        gated = cell["with_gate"]["batch_top20"]
        panel.plot(
            gated["coverage"], gated["realized_harm_rate"], "k+", markersize=10, label="+ gate"
        )
        panel.axhline(PRIMARY_EPSILON, color="#a33", linestyle=":", linewidth=1)
        panel.set_title(f"held out: {engine}", fontsize=10)
        panel.set_xlabel("coverage (fraction of sites accepted)")
    np.atleast_1d(panels)[0].set_ylabel("realized harm rate among accepted edits")
    np.atleast_1d(panels)[0].legend(fontsize=7)
    finish(
        figure,
        FIGURE_DIR / "coverage_risk_curve.png",
        "Risk against coverage; the dotted line is the nominal bound",
    )

    manifest = {
        "schema_version": "sgv2-reliability-figures-v1",
        "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "issued_head": pilot._git_head(),
        "hypothesis_id": "SGV2-R1",
        "synthetic": False,
        "development_only": True,
        "sources": {
            cc._relative(path): file_sha256(path)
            for path in (SHIFT_RESULTS, RELIABILITY_RESULTS, TRANSFER_SAFETY, SCORES)
        },
        "figures": {cc._relative(path): file_sha256(path) for path in written},
        "confirmatory_accessed": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    cc._write_json_once(FIGURE_MANIFEST, manifest)
    print(f"figures: {len(written)} -> {cc._relative(FIGURE_DIR)}")
    return 0


# ------------------------------------------------------------------ the finding

# The support rule is Phase 6's, transposed from feature families to abstention arms: an arm
# is supported only if it beats the baseline on EVERY held-out engine. Requiring unanimity is
# what stops "it worked on Tesseract" from becoming the finding when four folds were run, and
# it is the same rule that produced Phase 6's negative -- not a criterion built to fit this
# stage's outcome.
DEPLOYABLE_ARMS = (
    "reject_top10_batch",
    "reject_top20_batch",
    "reject_top30_batch",
    "reject_top50_batch",
    "reject_bottom10_batch",
    "reject_bottom20_batch",
    "reject_bottom30_batch",
    "reject_bottom50_batch",
    "reject_above_p90_inductive",
    "reject_above_p95_inductive",
    "reject_above_p99_inductive",
    "reject_below_p10_inductive",
    "reject_below_p5_inductive",
    "reject_below_p1_inductive",
    "adaptive_inductive",
    "adaptive_batch",
    "warm_start_domain_batch",
)


def run_decide() -> int:
    """The machine-readable finding, computed from the artifacts and not from memory."""
    started = time.monotonic()
    shift = cc._read_json(SHIFT_RESULTS)
    reliability = cc._read_json(RELIABILITY_RESULTS)
    abstention = cc._read_json(ABSTENTION_RESULTS)
    safety = cc._read_json(TRANSFER_SAFETY)
    failure = cc._read_json(FAILURE_RESULTS)
    selection_record = cc._read_json(SELECTION_RECORD)
    key = f"epsilon_{int(PRIMARY_EPSILON * 100)}"
    engines = sorted(abstention["folds"])

    def recall(engine: str, arm: str) -> float:
        return float(abstention["folds"][engine]["arms"][arm][key]["repair_recall"])

    support = {
        arm: {
            "engines_beating_risk_only": sorted(
                e for e in engines if recall(e, arm) > recall(e, "risk_only")
            ),
            "engines_with_favourable_matched_coverage_ci": sorted(
                e
                for e in engines
                if abstention["folds"][e]["arms"][arm]["matched_coverage_contrast"][
                    f"coverage_{int(PRIMARY_EPSILON * 100)}"
                ]["ci_lower"]
                > 0.0
            ),
            "fold_mean_repair_recall": float(np.mean([recall(e, arm) for e in engines])),
        }
        for arm in DEPLOYABLE_ARMS
    }
    unanimous = [
        arm
        for arm in DEPLOYABLE_ARMS
        if len(support[arm]["engines_beating_risk_only"]) == len(engines)
        or len(support[arm]["engines_with_favourable_matched_coverage_ci"]) == len(engines)
    ]

    cold = {
        e: {
            kind: shift["folds"][e]["detectors"][kind]["row_level_auroc"]["estimate"]
            for kind in COLD_START_DETECTORS
        }
        for e in engines
    }
    warm = {
        e: shift["folds"][e]["detectors"]["domain_classifier"]["row_level_auroc"]["estimate"]
        for e in engines
    }
    spearman = {
        e: {
            kind: reliability["folds"][e]["detectors"][kind]["spearman_shift_vs_harm"]["estimate"]
            for kind in DETECTORS
        }
        for e in engines
    }
    selected_spearman = {
        e: reliability["folds"][e]["detectors"][abstention["folds"][e]["selected_detector"]][
            "spearman_shift_vs_harm"
        ]
        for e in engines
    }

    payload = {
        "schema_version": "sgv2-reliability-decision-v1",
        "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "issued_head": pilot._git_head(),
        "hypothesis_id": "SGV2-R1",
        "hypothesis": (
            "a shift-aware abstention mechanism preserves bounded-risk OCR correction on an "
            "unseen engine by detecting unreliable conditions before the correction is applied"
        ),
        "id_note": (
            "The brief calls this H5. docs/sgv1/protocol.md binds SGV1-H1..H4 and says a frozen "
            "ID is never reused for a different claim, so the family restarts at SGV2-R1."
        ),
        "support_rule": (
            "an arm is supported only if it beats risk-only on the primary endpoint on EVERY "
            "held-out engine, or its matched-coverage harm reduction has a lower CI bound above "
            "zero on EVERY held-out engine. Phase 6's rule, transposed to abstention arms."
        ),
        "support": support,
        "arms_meeting_the_rule": unanimous,
        "verdict": "SUPPORTED" if unanimous else "NOT SUPPORTED",
        "q1_can_we_know_when_correction_is_unsafe": {
            "answer": "partly, and not in the mode that would matter first",
            "cold_start_row_auroc": cold,
            "cold_start_engines_below_chance": sorted(
                e for e in engines if max(cold[e].values()) < 0.5
            ),
            "cold_start_engines_not_clear_of_their_own_null": sorted(
                e
                for e in engines
                if max(
                    shift["folds"][e]["detectors"][kind]["row_level_auroc"]["estimate"]
                    - shift["folds"][e]["detectors"][kind]["within_engine_null"]["p95"]
                    for kind in COLD_START_DETECTORS
                )
                <= 0.0
            ),
            "null_note": (
                "0.5 is not the right reference. A detector that scores unseen PAGES as "
                "anomalous would look like an engine detector, so the standard each detector "
                "has to clear is its own within-engine null: the same measurement made between "
                "two random halves of the fit engines' own development pages."
            ),
            "warm_start_row_auroc": warm,
            "why": (
                "Detecting THAT the engine changed is not detecting WHICH edits are unsafe. The "
                "warm-start classifier separates engines almost perfectly and needs a batch of "
                "unlabeled pages from the new engine first; the cold-start detectors, which are "
                "the ones that could act on the first page, are at or below chance on part of "
                "the panel."
            ),
        },
        "q2_does_uncertainty_correlate_with_harm": {
            "spearman_by_detector": spearman,
            "selected_detector_spearman": selected_spearman,
            "engines_with_the_sign_the_hypothesis_needs": sorted(
                e for e in engines if selected_spearman[e]["estimate"] > 0.0
            ),
            "answer_note": (
                "The hypothesis needs a positive coefficient: more anomalous, more harmful. The "
                "measured signs are reported as they came out."
            ),
        },
        "q3_can_abstention_recover_safety": {
            "fold_mean_repair_recall": abstention["fold_mean_repair_recall_at_primary_epsilon"],
            "oracle_ceiling_minus_risk_only": {
                e: float(recall(e, "oracle_selected_batch") - recall(e, "risk_only"))
                for e in engines
            },
            "bound_violations_without_gate": sorted(
                e
                for e in engines
                for controller in safety["folds"][e]["deployed_operating_points"]
                if safety["folds"][e]["deployed_operating_points"][controller]["risk_policy_only"][
                    "bound_violated"
                ]
            ),
            "bound_restored_by_gate": sorted(
                f"{e}|{controller}|{gate}"
                for e in engines
                for controller in safety["folds"][e]["deployed_operating_points"]
                for gate in safety["folds"][e]["deployed_operating_points"][controller]["with_gate"]
                if safety["folds"][e]["deployed_operating_points"][controller]["with_gate"][gate][
                    "bound_restored"
                ]
            ),
            "selection_transfer": {
                e: {
                    "oracle_choice": selection_record["folds"][e]["diagnostics"][
                        "oracle_selection"
                    ]["selected"],
                    "oracle_repair_recall": selection_record["folds"][e]["diagnostics"][
                        "oracle_selection"
                    ]["selected_repair_recall"],
                    "inner_selected_choice": (
                        f"{selection_record['folds'][e]['selected_detector']}"
                        f"|kappa_{selection_record['folds'][e]['selected_kappa']:g}"
                    ),
                    "inner_selected_repair_recall": selection_record["folds"][e]["diagnostics"][
                        "oracle_selection"
                    ]["deployable_selection_repair_recall"],
                    "risk_only_repair_recall": recall(e, "risk_only"),
                }
                for e in engines
            },
            "engines_where_the_inner_selected_policy_is_worse_than_no_gate": sorted(
                e
                for e in engines
                if selection_record["folds"][e]["diagnostics"]["oracle_selection"][
                    "deployable_selection_repair_recall"
                ]
                < recall(e, "risk_only")
            ),
            "inductive_rejection_fraction_by_engine": {
                e: {
                    name: cell["realized_rejection_fraction"]
                    for name, cell in selection_record["folds"][e]["diagnostics"][
                        "rejection"
                    ].items()
                    if "inductive" in name
                }
                for e in engines
            },
            "why_the_oracle_matters": (
                "The oracle picks the detector and kappa on the held-out engine's own labels. If "
                "it is close to risk-only, the shortfall is not a selection failure that better "
                "tuning could fix -- the signal an abstention rule would have to act on is not "
                "in the score."
            ),
        },
        "q4_remaining_limitation": {
            "four_engines": (
                "four engines give four folds; every claim here is a claim about this panel, and "
                "the engine axis is the small-sample axis, not the row axis"
            ),
            "development_only": (
                "the confirmatory reserve is untouched, so nothing here is a confirmatory result"
            ),
            "no_review_channel": (
                "ABSTAIN and PRESERVE are the same event in this benchmark; the value of routing "
                "to a human is reported only as an upper bound under a perfect-reviewer "
                "assumption"
            ),
            "detector_family": (
                "four label-free detectors were tried. A detector built on raw page pixels or on "
                "engine-internal state rather than on the 96 candidate features was not tried, "
                "and this stage cannot speak to it"
            ),
        },
        "failure_modes": failure["engines_where_shift_beats_chance_for_this_class"],
        "confirmatory_accessed": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    cc._write_json_once(DECISION, payload)
    print(f"decide: {payload['verdict']} -> {cc._relative(DECISION)}")
    return 0


def run_record() -> int:
    """What produced every artifact in this stage, with hashes."""
    started = time.monotonic()
    design = dg.load_design()
    produced = [
        path
        for path in (
            SCORES,
            SELECTION_RECORD,
            SHIFT_RESULTS,
            RELIABILITY_RESULTS,
            ABSTENTION_RESULTS,
            TRANSFER_SAFETY,
            FAILURE_RESULTS,
            DECISION,
            FIGURE_MANIFEST,
        )
        if path.is_file()
    ]
    cc._write_json_once(
        FIT_RECORD,
        {
            "schema_version": "sgv2-reliability-fit-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV2-R1",
            "representation": (
                "Phase 3's candidate-conditioned R1 design matrix, reused byte-for-byte from "
                "Phase 6 rather than rebuilt, so this stage's baseline is Phase 6's model"
            ),
            "n_features": len(design.names),
            "feature_names": list(design.names),
            "families": {
                family: [n for n in design.names if n.startswith(prefix)]
                for family, prefix in dg.FAMILIES.items()
            },
            "risk_model": {
                "estimator": "sklearn LogisticRegression",
                "source": "sgv1_domain_generalization.fit_harm_model, imported unchanged",
                "target": "is_harmful",
                "random_state": pilot.FIT_SEED,
            },
            "detectors": {
                "cold_start": list(COLD_START_DETECTORS),
                "warm_start": list(WARM_START_DETECTORS),
                "mahalanobis_ridge": MAHALANOBIS_RIDGE,
                "forest_trees": FOREST_TREES,
                "one_class_nu": ONE_CLASS_NU,
                "one_class_subsample": ONE_CLASS_SUBSAMPLE,
            },
            "selection": {
                "kappa_grid": list(KAPPA_GRID),
                "reject_fractions": list(REJECT_FRACTIONS),
                "inductive_upper_quantiles": list(INDUCTIVE_UPPER),
                "inductive_lower_quantiles": list(INDUCTIVE_LOWER),
                "scope": "inner leave-one-engine-out over the fit engines, CALIBRATION documents",
            },
            "endpoints": {
                "primary": "repair_recall_at_bounded_harm (imported from Phase 5)",
                "co_primary": "harm reduction at matched site coverage",
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
            "ground_truth_used_for_detectors": False,
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
        ("scores", run_scores),
        ("shift", run_shift),
        ("reliability", run_reliability),
        ("abstain", run_abstain),
        ("transfer", run_transfer),
        ("failure", run_failure),
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
